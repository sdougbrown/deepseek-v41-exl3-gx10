# TP=3 port plan — DeepSeek-V4.1-Flash EXL3 on the 3× GX10 directed ring

Goal: run this EXL3 kit (vLLM) TP=3 across scenty (head, rank 0), bitey (rank 1),
sparky (rank 2), matching the now-proven GLM-5.3 TP=3 setup
(`sdougbrown/glm53-exl3-gx10`). Two sibling kits inform this port but neither
transplants directly:

- `MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks` (official NVFP4, **sglang**): its
  default profile is a 3-Spark TP=3 config. Supplies *shape decisions* only —
  TP=3 + EP=3, `DSV41_TP_PAD=1` padding baked into its image (not portable to
  vLLM), DSpark SPS/STS profiler tables, mem-fraction notes. Its start-tp4.sh is
  a thin env-profile wrapper worth copying as a file layout (`state-tp3/`,
  `logs-tp3/`, `.env.tp3`).
- `sdougbrown/glm53-exl3-gx10` (GLM EXL3, vLLM): same harness family as this
  repo; its TP=3 campaign is done and its fork divergences are exactly the ones
  this repo needs.

## 1. Fabric / launcher divergences to port from glm53-exl3-gx10

All three of these were built and proven there (see its git log: merge 4833e1c,
port 46fac6f, 61ce69b, 899c634):

1. **Literal model dirs instead of HF cache + rsync.** `WORKER_MODEL_DIR` /
   `MODEL_HOST` paths → a `MODEL_LITERAL` env pointing at the shared NFS-RDMA
   tree: `/mnt/models/DeepSeek-V4.1-Flash-EXL3-2.9bpw` (already on sparky,
   mounted rw on scenty+bitey). Skip `sync_dir_to_worker` when set; every rank
   bind-mounts the literal path. The `.orig`/marker probes in start.sh must not
   attempt writes on a `root_squash` export.
2. **GID `auto` resolution with drift semantics.** Port `preflight_rank_gid` +
   `resolve_gid_rank`: validate a RoCE v2 IPv4-mapped GID exists per listed
   device per rank, then forward an **empty** `NCCL_IB_GID_INDEX` so NCCL 2.30
   auto-selects per port. Pinning one index across two ports breaks when tables
   drift (observed: bitey f0 idx3/f1 idx5; scenty f0 moved 3→4 after re-address).
3. **Caller-export precedence via `compgen -e` snapshot.** Replace the fixed
   `_cli_*` capture list so profile env files can be overridden per-invocation.

Plus TP=3 orchestration itself (`start-tp3.sh` modeled on the GLM one): rank 2
via `WORKER2_*`, per-rank container/cache dirs, per-rank `NCCL_IB_HCA` dual-port
lists, the `serve_env` whitelist pattern so workers need no profile files.

## 2. NCCL / network facts this kit must adopt (GLM campaign receipts)

- Directed-ring wiring: one **/24 per cable** (router uses a single global
  `NCCL_IB_SUBNET_PREFIX_LEN=24`; /16 links are invisible to
  `NCCL_IB_SUBNET_AWARE_ROUTING` and NCCL silently falls back to same-index NIC
  pairing → uncabled QP → `ibv_modify_qp` ETIMEDOUT).
  scenty f0 `10.99.3.1` ↔ sparky f1 `10.99.3.2`; sparky f0 `10.99.2.1` ↔ bitey
  f1 `10.99.2.2`; bitey f0 `10.99.1.3` ↔ scenty f1 `10.99.1.2`.
- Rendezvous + NCCL bootstrap on the management LAN (`NCCL_SOCKET_IFNAME=enP7s7`,
  master = scenty `192.168.0.234`); data plane stays on RoCE via `NCCL_IB_HCA`.
- `NCCL_CROSS_NIC=1`, `NCCL_IB_SUBNET_AWARE_ROUTING=1`, prefix len default 24.
  Verified on this fabric with a 3-rank torchrun allreduce:
  `NET/IB: Subnet-aware routing: overriding dev 0 with dev 1` fires per peer,
  both connector and acceptor sides.
- Don't mount fabric IPs in fstab/profiles except per-link (NFS rides cable B:
  bitey → `10.99.2.1`; scenty → `10.99.3.2`).

## 3. TP=3 shape work — the real unknown (vLLM side)

Model facts (`config.json` text_config): `num_attention_heads=64`,
`num_key_value_heads=1` (MLA), `hidden=5120`, `vocab=129280`,
`moe_intermediate_size=2304` (÷3 ✓), `n_routed_experts=384` (÷3 ✓, EP=3 →
128/rank), `n_shared_experts=1`, 40 layers. Official sglang TP=3 says heads /
hidden / vocab "are not divisible by 3" and pads in-image.

Port the GLM overlay approach (`overlay/tp3/`):

- `patch_tp3_glm.py` analogue for `vllm/models/deepseek*`: head padding 64→66
  (or lcm-based) in the MLA q-head shard + o_proj rows; vocab `padding_size`
  lcm(64, 3); shared-expert intermediate pad / `disable_tp` option.
- EXL3 specifics: `patch_exl3_ep_shard.py` (whole experts under EP — 384/3=128
  whole experts per rank) and `patch_exl3_expert_map.py` port nearly verbatim;
  the fat MoE kernels' row tiles against `2304/3=768` need a divisibility audit
  (`exl3_fat_moe.cu` tile params, `EXL3_MOE_ROW_TILE`).
- DSpark is **in-checkpoint** (`mtp.*`, block_size 5) — no draft repo, no padded
  draft root, nothing to pad-copy like GLM's DFlash2. Verify the draft's own
  head/expert shapes under TP=3; the official TP4 notes mention "draft experts"
  padding, so the mtp module likely needs the same EP-shard treatment.
- Vision: config has `vision_config`. GLM lesson — vision heads may not divide
  by 3; plan `--mm-encoder-tp-mode data` from the start, and note this image
  builds the tower regardless of `--language-model-only`.

**Test-first gate:** before loading ~140 GiB, run a TP=3 dummy-weights probe
(`--load-format dummy`, tiny `max-model-len`) on the three boxes with the real
overlay patches, to enumerate every divisibility assert in one pass (this is the
loop that found GLM's vision-16 trap in one boot).

## 4. Memory / engram

- GB10 is unified memory: watch host `MemAvailable` (sparky previously crashed
  as head at ~3 GiB; cause never found — keep it a worker here).
- Engram tables: upstream ships `ENGRAM_DIR` + `NFS_VOLUME_ENGRAM` options, but
  the official kit insists engram callbacks block the compute stream and wants
  local NVMe. Decide deliberately: local per-rank copy of the engram volume vs
  NFS with `DSV41_CACHE_GIB=0` + IO-thread pool. Start local (disk is plentiful
  on each box), NFS only if size forces it.
- KV pool: GLM TP=3 got 3.0M tokens at 40 GiB/rank cap with 850k ctx; DeepSeek
  MLA KV is far smaller per token — recompute block sizing, don't copy numbers.

## 5. Serve-side (~/Serve)

Mirror the GLM layout once the launcher works: `hosts/scenty/dsv41-tp3-head.env`
+ wrapper `serve-glm`-style trio with a `DSV41_PROFILE` selector (or generalize
the GLM wrappers), profiles carrying the fabric facts from §2, image ship
`SKIP_SHIP` once workers have content.

## Order of attack

1. Fork divergences §1 onto `start.sh` (literal dirs first — bitey runs TP=2 on
   this kit today; keep that path working).
2. `start-tp3.sh` + `.env.tp3.example` (§1 orchestration, §2 defaults).
3. Dummy-weights TP=3 probe → enumerate shape failures → overlay `tp3/` patches (§3).
4. Real load with engram local (§4), then throughput receipts vs TP=2.
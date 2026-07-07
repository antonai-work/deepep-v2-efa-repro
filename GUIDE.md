# GUIDE — how to use this repo

This repo lets you reproduce **DeepEP-V2 MoE inference over AWS EFA** — four
independently verifiable gates — from public upstreams plus the patch files
committed here. `README.md` is the full recipe with the measured numbers; this
page is the map: what to run, in what order, and what "pass" looks like.

## What you need

- 2x AWS p5en.48xlarge (H200, 16 EFA NICs each), same EFA security-group
  rules (self-referencing ingress AND egress), primary-CIDR IPs.
- A container substrate rooted at an official NGC CUDA base with EFA
  userspace 1.48.0 + gdrcopy + torch 2.11/cu130 (`trtllm/Dockerfile` layers
  1-5 build exactly this, or use `trtllm/scripts/00pre-substrate-from-ngc.sh`
  in a bare NGC pod).
- HuggingFace access to `Qwen/Qwen3-30B-A3B-FP8` (~32 GB).
- No artifact from us: every root is NGC, PyPI, pypi.nvidia.com, GitHub
  upstream, or a patch file in `patches/`.

## The four gates, in dependency order

| # | Gate | What it proves | Pass bar (measured 2026-07-06/07) |
|---|---|---|---|
| 1 | micro D+C (`bench/micro/`) | the transport itself: ElasticBuffer over NCCL-GIN CPU-proxy on EFA | D+C p50 <= 740us (we got 443.6us) |
| 2 | vLLM + AIPerf (`bench/vllm/`) | unmodified PyPI engine serves MoE over that transport | 3 cells, 0 errors (c32 ~145 tok/s) |
| 3 | SGLang + AIPerf (`bench/sglang/`) | second engine, needs the seam patch | 3 cells, 0 errors (c32 ~255 tok/s) |
| 4 | TRT-LLM + AIPerf (`trtllm/`) | shim-adapted engine (native EP is an EFA wall) | 3 cells, 0 errors (c32 ~587 tok/s) |

Run them in order: gate 1 validates the substrate every later gate stands on.
If gate 1 fails, nothing downstream is meaningful — fix the transport first
(README section 1-3).

## Common steps for every gate

1. **Assemble DeepEP** (README §1): public `deepseek-ai/DeepEP@b306af06` +
   the PR-#612 trio (`patches/0001..0003`) + the multicomm overlay
   (`patches/private-deltas/deepep-multicomm-*`), build `_C` with
   `TORCH_CUDA_ARCH_LIST=9.0`.
2. **Build the GIN plugin** (README §1): public `aws/aws-ofi-nccl@9c44d34`;
   on gdrdrv **2.4** kernels apply BOTH patch scripts
   (`patch-gdr-pin-v1-fallback.py`, `apply_forced_pcie_copy_bypass.py`) —
   serving faults CUDA 719 without them; a transport-only bench passes either
   way, so do not let gate 1 lull you.
3. **Export the common env** (README §2) — `NCCL_GIN_TYPE=2`,
   `OFI_NCCL_GIN_GDAKI=0`, `FI_PROVIDER=efa`, never IBGDA/GDAKI.
4. **Assert the backend on every serve**: the log must show `ElasticBuffer`
   construction (never legacy `Buffer`) and `Selected provider is efa`.
   `bench/vllm/assert_backend.py` automates this. A benchmark number without
   the assertion is not evidence.

## Per-gate entry points

- **Gate 1**: `bench/micro/test_ep_torchrun_wrapper.py` via torchrun (README
  §3). Use torchrun, NOT mp.spawn — spawn overflows NCCL's 256-node XML cap
  on p5en. The wrapper header documents the two env traps (WORLD_SIZE/RANK
  normalization; MASTER_PORT == rdzv port).
- **Gate 2**: `pip install vllm==0.24.0` (the FIRST release with
  `DeepEPV2All2AllManager`), re-pin nccl/nvshmem, install the
  `bench/vllm/ep_patch_sitecustomize.py` `.pth` hook (kills flashinfer +
  injects 600s ElasticBuffer timeouts), serve leader/worker (README §4),
  then `ARM=deepep-vllm bash bench/aiperf/aiperf_sweep.sh`.
- **Gate 3**: `bench/sglang/install_sglang_v2.sh` (wheel + seam patch +
  re-pin), serve with `bench/sglang/serve_sglang_v2.sh` — the launcher shape
  is load-bearing, do not slim it (README §5). Then the same AIPerf sweep.
- **Gate 4**: `trtllm/INSTRUCTIONS.md` end-to-end. Two bring-up routes:
  from-scratch image (`trtllm/Dockerfile` + `scripts/00*`) or on a live cu13
  pod without touching its serving stack
  (`scripts/09-bringup-venv-on-live-cu13-pod.sh`, then
  `scripts/10-serve-venv-deepep-arm.sh`). The script headers encode every
  wall we hit (cuda-python 12.9 pin, redist nvcc, EP_NCCL_ROOT_DIR, math-lib
  headers, offline tokenizer).

## When something breaks

Work the gotchas index (README §8) plus the per-lane troubleshooting table
(`trtllm/INSTRUCTIONS.md`). The three failure patterns that cost us the most
time, in one place:

1. **"Gin barrier timeout tag:N" during serve is a SYMPTOM.** One rank died
   (import error, JIT failure) and the other 15 are waiting. Find the FIRST
   dead worker's traceback; do not debug the transport.
2. **Silent dependency drift.** Any `pip install` (vllm, sglang, trtllm deps)
   can replace `nvidia-nccl-cu13`/`nvidia-nvshmem-cu13` and break `deep_ep._C`
   with undefined symbols (`ncclTeamWorld`,
   `nvshmem_selected_device_transport`). After EVERY install:
   `pip install --force-reinstall --no-deps nvidia-nccl-cu13==2.30.4
   nvidia-nvshmem-cu13==3.6.5`.
3. **State pollution between attempts.** Reap by explicit PID from
   `nvidia-smi --query-compute-apps=pid` (never `pkill -f`), use a fresh
   dist/rdzv port per attempt, and distrust any latency number from a pod
   with leftover GPU contexts.

## Reading the results

Every gate's as-run artifacts are committed under `results/<gate>-*/`:
AIPerf `profile_export_aiperf.json` (check `error_summary: []` and
`request_count`), plus serve-log evidence (`provider_evidence.log` for the
TRT lane shows the efa-provider x16 + `AlltoallMethodType.DeepEP` lines).
Your numbers should land within ~10% of README §7 on the same instance type;
TTFT is the most environment-sensitive metric, aggregate tok/s the least.

## Scope honesty

- vLLM needs no source patch (0.24.0 has V2 natively); SGLang needs the seam
  patch (upstream PR #24443 lineage, unmerged); TRT-LLM works ONLY through
  the api-shim (its native EP path is MNNVL/IBGDA — dead on EFA) and its
  DeepEP arm loses to its own dense arm (`trtllm/WHY-TRT-SLOW.md`) — the gate
  is "serves + AIPerf clean + backend-asserted", not "faster".
- `EP_NUM_COMMS>1` (the 2.09x vLLM arm) requires the multicomm overlay patch;
  the pure-public arm is `EP_NUM_COMMS=1`.
- On pods without `/sys/class/infiniband/*/ports/*/hw_counters` (efa-direct
  fabric), the EFA-traffic proof substitutes the provider log line + a
  cross-node-only topology, as recorded in the results.

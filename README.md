# DeepEP-V2 MoE on AWS EFA — reproduce all four gates from PUBLIC roots only

> New here? **Start with [GUIDE.md](GUIDE.md)** — what to run, in what order,
> and what "pass" looks like. This page is the full recipe + measured numbers.

Reproduce all four acceptance gates — the **micro D+C benchmark**, the
**vLLM + AIPerf** lane, the **SGLang + AIPerf** lane, and the **TRT-LLM
(api-shim) + AIPerf** lane — starting from **public upstreams only**: official
NVIDIA NGC bases, `deepseek-ai/DeepEP`, PyPI wheels (`vllm`, `sglang`,
`tensorrt_llm`), `aws/aws-ofi-nccl`, and the patch files committed in this
repo. No image, fork, or registry we published is required anywhere.
Measured numbers below are from the 2026-07-06/07 runs on 2x p5en.48xlarge
(H200, EFA, gdrdrv kernel 2.4); the as-run artifacts are under `results/`.

The ONE transport in scope: `deep_ep.ElasticBuffer` over **NCCL-GIN v2
CPU-proxy** (`NCCL_GIN_TYPE=2`, `OFI_NCCL_GIN_GDAKI=0`). Never GDAKI/IBGDA —
those are EFA-dead for this path.

---

## 0. Public roots (everything bottoms out here)

| Component | Public root | Pin |
|---|---|---|
| Base image | `nvcr.io/nvidia/cuda` | `13.0.0-devel-ubuntu22.04` (serving substrate) / `12.9.0-devel-ubuntu24.04` (TRT-LLM lane) |
| EFA userspace | `https://efa-installer.amazonaws.com` | aws-efa-installer 1.48.0 |
| gdrcopy (userspace) | `NVIDIA/gdrcopy` | 2.5.x |
| torch | PyPI cu130 index | `torch==2.11.0+cu130` (+ `nvidia-nccl-cu13==2.30.4`, `nvidia-nvshmem-cu13==3.6.5`) |
| DeepEP V2 | `deepseek-ai/DeepEP` | `b306af06` + patches (below) |
| aws-ofi-nccl (GIN plugin) | `aws/aws-ofi-nccl` | `9c44d34` + 2 patch scripts (below) |
| vLLM | PyPI | `vllm==0.24.0` — the FIRST official release carrying `DeepEPV2All2AllManager` (merge `e2f993dc` is an ancestor; 0.23.0 does NOT have it) |
| SGLang | PyPI | `sglang==0.5.11` + seam patch (below) |
| TRT-LLM | pypi.nvidia.com | `tensorrt_llm==0.21.0` + `tensorrt==10.11.0` + shim (below) |
| AIPerf | PyPI | `aiperf==0.10.0` |
| Model | HuggingFace | `Qwen/Qwen3-30B-A3B-FP8` |

## 1. Patches (all committed in this repo; every one applies onto a public root)

Verified `git apply --check`-clean on fresh public clones/wheels 2026-07-07.

```
patches/
  0001..0003-*.patch                     # DeepEP PR #612 trio -> deepseek-ai/DeepEP@b306af06
  private-deltas/
    deepep-multicomm-overlay.patch       # EP_NUM_COMMS>1 multi-comm (the vLLM 2.09x EPNC=8 win);
    deepep-multicomm-newfile.patch       #   inert when EP_NUM_COMMS unset/1. Apply AFTER the trio.
    sglang-0.5.11-deepep-v2-seam.patch   # SGLANG_DEEPEP_USE_V2=1 -> ElasticBuffer (PR #24443 lineage)
    sglang-v2_compat_buffer.py           #   companion file, copy next to deepep.py
    patch-gdr-pin-v1-fallback.py         # aws-ofi 9c44d34: gdr_pin_buffer_v2 EINVAL/ENOTTY -> v1 ioctl
    apply_forced_pcie_copy_bypass.py     # aws-ofi 9c44d34: stale kernel-probe -> force forced_pcie_copy()
trtllm/   # full TRT-LLM lane (Dockerfile, api-shim, scripts, results)
```

DeepEP assembly (identical to what served):

```bash
git clone https://github.com/deepseek-ai/DeepEP.git /opt/DeepEP && cd /opt/DeepEP
git checkout b306af06
git am  <repo>/patches/0001-*.patch 0002-*.patch 0003-*.patch
git apply <repo>/patches/private-deltas/deepep-multicomm-overlay.patch
git apply <repo>/patches/private-deltas/deepep-multicomm-newfile.patch
TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=12 NVCC_THREADS=2 pip install --no-build-isolation -e .
```

aws-ofi-nccl plugin (the gdrdrv-2.4-safe serving plugin):

```bash
git clone https://github.com/aws/aws-ofi-nccl /opt/aws-ofi-nccl-src && cd /opt/aws-ofi-nccl-src
git checkout 9c44d34
python3 <repo>/patches/private-deltas/patch-gdr-pin-v1-fallback.py   src/nccl_ofi_gdrcopy.cpp
python3 <repo>/patches/private-deltas/apply_forced_pcie_copy_bypass.py src/nccl_ofi_gdrcopy.cpp
./autogen.sh && ./configure --with-libfabric=/opt/amazon/efa --with-cuda=/usr/local/cuda --enable-platform-aws
make -j"$(nproc)" -C src
# plugin at src/.libs/libnccl-net-ofi.so
```

On gdrdrv-**2.5** kernels the two plugin patches are unnecessary (stock
`aws/aws-ofi-nccl@12166be`+forced-pcie also works); on 2.4 kernels BOTH are
required for serving (transport bench passes either way — the 719 combine
fault is serving-path-specific).

## 2. Common serving environment (all lanes)

```bash
export NCCL_NET_PLUGIN=<your-built>/libnccl-net-ofi.so
export NCCL_GIN_TYPE=2 NCCL_GIN_ENABLE=1 OFI_NCCL_GIN_GDAKI=0
export NCCL_CUMEM_ENABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IGNORE_DISABLED_P2P=1
export FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 FI_EFA_ENABLE_SHM_TRANSFER=0
export OFI_NCCL_PROTOCOL=RDMA OFI_NCCL_GIN_MAX_REQUESTS=512
export EP_EFA_MAX_QPS=2 EP_EFA_RDMA_GBS=25.0
unset NVSHMEM_IB_ENABLE_IBGDA
```

Backend assertion is mandatory on every serve: the log must show
`ElasticBuffer` construction (never legacy `Buffer`) and
`Selected provider is efa` — `bench/vllm/assert_backend.py` automates it.

## 3. Gate 1 — micro D+C benchmark (test_ep, 2 nodes, EP16)

Run DeepEP's own `tests/elastic/test_ep.py` cross-node via **torchrun** (NOT
`mp.spawn` — spawn overflows NCCL's 256-node XML topo cap on p5en). Two traps
solved here, wrapper committed at `bench/micro/test_ep_torchrun_wrapper.py`:

1. `test_ep`'s `init_dist` env contract is `WORLD_SIZE=<num NODES>`,
   `RANK=<node rank>` — normalize from torchrun's globals before calling
   `test_loop`.
2. `MASTER_PORT` must EQUAL the torchrun rdzv port (agent-store TCPStore).

```bash
# node 0 and node 1 (bench config = bench/bench_v2_only.sh contract):
EP_BENCH_MASTER=<node0-ip> \
python -m torch.distributed.run --nproc-per-node=8 --nnodes=2 --node-rank=<0|1> \
  --rdzv-backend=c10d --rdzv-endpoint=<node0-ip>:29650 \
  bench/micro/test_ep_torchrun_wrapper.py
# wrapper config: num_tokens=128 hidden=7168 topk=8 experts=256 num_sms=4 num_qps=6, EP_EFA_MAX_QPS=6
```

**Measured (2026-07-06):** dispatch p50 185.4us, combine p50 258.2us, **D+C
443.6us** — 1.67x better than the ~740us contract. 16/16 ranks rc=0.

## 4. Gate 2 — vLLM + AIPerf (PUBLIC wheel)

```bash
pip install "vllm==0.24.0" "aiperf==0.10.0"   # torch must stay +cu130: re-pin
pip install --force-reinstall --no-deps "nvidia-nccl-cu13==2.30.4" "nvidia-nvshmem-cu13==3.6.5"
```

Three serve-killers to neutralize (all root-caused live):
1. **flashinfer must be invisible** (uninstall or gate `has_flashinfer()->False`):
   its FP8-MoE JIT needs `ninja` and version-clashes kill one worker; the
   other 15 ranks then show "Gin barrier timeout tag:8" — a SYMPTOM.
   The proven path is triton MoE (`VLLM_USE_FLASHINFER_MOE_FP8=0`).
2. **ElasticBuffer serve timeouts >= 600s** (vLLM 0.24 passes none; the 100s
   default loses to first-dispatch JIT skew). Inject via the committed
   `.pth` hook `bench/vllm/ep_patch_sitecustomize.py` (install instructions
   in its header; it also handles killer 1 by forcing flashinfer `has_*()`
   gates to False).
3. Wheel dependency drift re-pins nccl/nvshmem (step above).

Serve (leader on node0 / worker on node1; `bench/vllm/serve_ep16_AB.sh` shape):

```bash
vllm serve Qwen/Qwen3-30B-A3B-FP8 --enable-expert-parallel \
  --tensor-parallel-size 1 --data-parallel-size 16 --data-parallel-size-local 8 \
  --all2all-backend deepep_v2 --trust-remote-code --enforce-eager \
  --data-parallel-address <node0-ip> --data-parallel-rpc-port 29570 \
  [leader: --port 8000 --data-parallel-start-rank 0 | worker: --headless --data-parallel-start-rank 8]
# env adds: VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE=1, VLLM_USE_DEEP_GEMM=0, EP_NUM_COMMS=1 (pure public)
# EP_NUM_COMMS=8 (the 2.09x arm) additionally requires the multicomm overlay patch.
```

Gate: `ARM=deepep-vllm bash bench/aiperf/aiperf_sweep.sh` (conc 4/8/32,
ISL120/OSL128, ignore_eos).

**Measured (2026-07-07, EPNC=1 arm):** c4/c8/c32 = **19.5 / 38.8 / 144.6** agg
tok/s, TTFT p50 474/510/899 ms, **0 errors** (232 requests). Log proof:
`Using DeepEPV2PrepareAndFinalize`, ElasticBuffer ctor x16, efa-direct x16 nics.

## 5. Gate 3 — SGLang + AIPerf (PUBLIC wheel + seam patch)

```bash
pip install "sglang[all]==0.5.11" "kernels==0.12.0"
pip install --force-reinstall --no-deps "nvidia-nccl-cu13==2.30.4" "nvidia-nvshmem-cu13==3.6.5"
cd <site-packages> && git apply <repo>/patches/private-deltas/sglang-0.5.11-deepep-v2-seam.patch
cp <repo>/patches/private-deltas/sglang-v2_compat_buffer.py sglang/srt/layers/moe/token_dispatcher/
```

Two mandatory knobs beyond the common env:
1. `SGLANG_DEEPEP_USE_V2=1` (constructs ElasticBuffer).
2. `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024` — the default 128 is
   decode-only-safe; AIPerf conc32 prefill bursts trip
   `buffer.hpp:686 num_tokens <= num_max_tokens_per_rank` on every rank.
   SGLang hard-caps this knob at 1024.

Launcher shape MATTERS (a slimmed launcher faults combine CUDA 719 even with
the right plugin). Use the full proven shape — committed as
`bench/sglang/serve_sglang_v2.sh` (install: `bench/sglang/install_sglang_v2.sh`):

```bash
python3 -m sglang.launch_server --model-path <model> --trust-remote-code \
  --tp-size 16 --ep-size 16 --dp-size 16 --nnodes 2 --node-rank <0|1> \
  --dist-init-addr <node0-ip>:29800 --mem-fraction-static 0.85 \
  --moe-dense-tp-size 1 --chunked-prefill-size 32768 --cuda-graph-bs 256 \
  --page-size 256 --attention-backend fa3 --ep-num-redundant-experts 0 \
  --ep-dispatch-algorithm dynamic --enable-dp-attention --enable-dp-lm-head \
  --moe-a2a-backend deepep --deepep-mode normal --deepep-config <cfg-num_sms24> \
  --host 0.0.0.0 --port 30000
```

Gate: `ARM=deepep-sglang URL=http://127.0.0.1:30000 bash bench/aiperf/aiperf_sweep.sh`.

**Measured (2026-07-07):** c4/c8/c32 = **34.5 / 67.3 / 255.2** agg tok/s,
**0 errors**; c32 reproduces the 2026-06-20 campaign baseline (253) at 1.01x.

## 6. Gate 4 — TRT-LLM + AIPerf (native V2 seam; native EP is an EFA wall)

TRT-LLM's built-in expert-parallel transport is MNNVL/IBGDA — dead on EFA.
The clean path (validated 2026-07-07, NO shim) is the **native seam patch**
`patches/private-deltas/trtllm-0.21.0-deepep-v2-seam.patch`: one
file (`deep_ep_utils.py`) on the public wheel, constructing `ElasticBuffer`
directly (MPI->torch-PG bootstrap, MetaInitMode deferral, layout-free V2
dispatch/combine, V1 fallback + `TRTLLM_DEEP_EP_FORCE_V1` escape). Apply
into site-packages with `git apply -p1`; serve WITHOUT any `PYTHONPATH`
shim. An upstream PR to NVIDIA/TensorRT-LLM carrying the same seam is in preparation.

**Measured (2026-07-07, seam, 3-cell):** c4/c8/c32 = **67.6 / 142.4 / 563.3**
agg tok/s, **0 errors**; log shows `[deep-ep-v2-seam]` PG+ctor x16 and zero
shim lines. Matches the shim arm within noise.

The legacy **api-shim** lane (`deep_ep.Buffer -> CompatBuffer ->
ElasticBuffer` via sitecustomize, validated 2026-06-22 and re-validated
2026-07-07) remains packaged for reference and for frozen-binary scenarios.
Complete package (Dockerfile FROM `nvcr.io/nvidia/cuda:12.9.0-devel-
ubuntu24.04`, shim source, 9-wall bring-up script, serve A/B, manifests,
measured results): `trtllm/`.

Key pins: `tensorrt_llm==0.21.0` (pypi.nvidia.com cp310/cp312 wheels) +
`tensorrt==10.11.0`; torch 2.7.1/cu126 (wheel dep); deep_ep `_C` REBUILT
against torch 2.7.1 + nccl 2.30.4 + `nvidia-nvshmem-cu12==3.6.5`; MNNVL x86
short-circuit patch; serve via `mpirun -np 16` + `trtllm-serve` with
`enable_attention_dp: true, moe_backend: CUTLASS` YAML; env
`TRTLLM_CAN_USE_DEEP_EP=1 DEEP_EP_USE_V2_SHIM=1 PYTHONPATH=<shim>`.

**Measured (2026-07-07, re-reproduced on the bench pods, full 3-cell sweep):**
c4/c8/c32 = **66.5 / 147.5 / 586.7** agg tok/s, **0 errors** (232 requests),
TTFT p50 160/163/203 ms. Log proof: `AlltoallMethodType.DeepEP`,
`trtllm-comm-bridge` x16, `shim5c combine` x16, `Selected provider is efa,
fabric is efa-direct` x16. The c4 cell reproduces the 2026-06-22 baseline
(64.2) at 1.04x. Honest verdict unchanged: on EFA's CPU-proxy path TRT-LLM's
DeepEP arm loses to its own dense arm (106.4 at c4 — WHY-TRT-SLOW.md); the
gate is "serves + AIPerf clean + backend-asserted", which it does.
Two bring-up routes, both committed: from-scratch image
(`trtllm/Dockerfile` + `scripts/00*`) or on a LIVE cu13 serving pod
without touching its stack (`scripts/09-bringup-venv-on-live-cu13-pod.sh` +
`scripts/10-serve-venv-deepep-arm.sh` — the exact as-run 2026-07-07 chain).
The walls are pinned in the scripts: cuda-python MUST be 12.9.0 (trtllm
resolver pulls 13.x -> `from cuda import cuda` ImportError); deep_ep's `_C`
build needs `EP_NCCL_ROOT_DIR` at nccl 2.30.4 (a stray venv nccl 2.26 loses
`nccl_device/core.h` + `ncclTeamWorld`); pip's nvcc wheel is ptxas-only (use
the official redist archive); AIPerf offline needs `TOKENIZER=<snapshot dir>`.
Per-wall troubleshooting table: `trtllm/INSTRUCTIONS.md`.

## 7. Expected-numbers summary

| Gate | Metric | Expected |
|---|---|---|
| micro | D+C p50, EP16 2-node | <= 740us contract (measured 443.6us) |
| vLLM | AIPerf c32 agg (EPNC=1) | ~145 tok/s, 0 err |
| SGLang | AIPerf c32 agg | ~253-255 tok/s, 0 err |
| TRT-LLM | AIPerf c4/c32, DeepEP arm | ~66-68 / ~563-587 tok/s, 0 err, `AlltoallMethodType.DeepEP` in log (native seam and shim equivalent within noise) |

EFA proof on clusters that expose hw_counters:
`scripts/verify_efa_traffic.sh` (TX delta >= 1 GB). On pods without
hw_counters sysfs (efa-direct fabric), substitute: `Selected provider is efa,
fabric is efa-direct` on all ranks + cross-node-only topology.

## 8. Gotchas index (each cost hours; all root-caused live)

- torch/vllm/sglang wheels silently replace `nvidia-nccl-cu13`/`nvshmem-cu13`
  -> `_C.so` undefined symbols. Always re-pin with `--force-reinstall --no-deps`.
- `mp.spawn` test_ep hits "too many XML nodes (max 256)" on p5en — use torchrun.
- "Gin barrier timeout tag:N" during serve = usually ONE dead rank elsewhere
  (import error, JIT failure), not a transport bug. Find the first dead worker.
- gdrdrv kernel 2.4 + libgdrapi 2.5 skew: transport bench passes, serving
  combine faults 719 unless the 9c44 v1-fallback + forced-pcie plugin is used.
- Zombie rdzv/dist ports survive crashes — fresh port per attempt, reap by
  explicit PID (`nvidia-smi --query-compute-apps=pid`), never `pkill -f`.
- Ubuntu's stdlib `sitecustomize` shadows venv sitecustomize — deliver venv
  hooks via a `.pth` import line instead.

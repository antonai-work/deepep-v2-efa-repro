# RESULTS — every measured number, one page

All numbers from the committed AIPerf exports and serve logs under
`results/` (this repo), measured on **2x AWS p5en.48xlarge** (8x H200 +
16x EFA each; gdrdrv kernel 2.4), model **Qwen/Qwen3-30B-A3B-FP8**, EP16
across both nodes. Sweep shape everywhere: AIPerf 0.10.0, ISL 120 / OSL
128, `ignore_eos`, concurrency 4/8/32, streaming completions. Transport:
`deep_ep.ElasticBuffer` over NCCL-GIN v2 CPU-proxy on EFA.

## Gate 1 — micro D+C (DeepEP's own test_ep, cross-node EP16)

| Metric | Value |
|---|---|
| dispatch p50 | 185.4 us |
| combine p50 | 258.2 us |
| **dispatch+combine p50** | **443.6 us** (contract: <= 740 us -> 1.67x headroom) |
| ranks | 16/16 rc=0 |

Config: num_tokens=128, hidden=7168, topk=8, experts=256, num_sms=4,
num_qps=6. Source: `results/vllm-deepep-dropin-20260706/vllm/
t2_ep16_2node_test_ep_microbench.json`.

## Gates 2-4 — inference serving + AIPerf (all cells 0 errors, 232/232 requests per arm)

Aggregate output tok/s (higher is better):

| Arm | c4 | c8 | c32 |
|---|---|---|---|
| vLLM 0.24.0 (EPNC=1, pure public) | 19.5 | 38.8 | 144.6 |
| vLLM 0.24.0 (**EPNC=8** multicomm) | **39.4** | **76.5** | **273.0** |
| SGLang 0.5.11 + seam | 34.5 | 67.2 | 255.2 |
| TRT-LLM 0.21.0 api-shim | 66.5 | 147.5 | 586.7 |
| TRT-LLM 0.21.0 **native seam** | 67.6 | 142.4 | 563.3 |

Per-user tok/s and latency (p50):

| Arm | per-user c4/c8/c32 | TTFT ms c4/c8/c32 | ITL ms c4/c8/c32 |
|---|---|---|---|
| vLLM (EPNC=1) | 4.9 / 4.9 / 4.7 | 475 / 510 / 899 | 203 / 203 / 206 |
| vLLM (EPNC=8) | 10.0 / 9.7 / 8.9 | 196 / 237 / 280 | 100 / 103 / 106 |
| SGLang | 8.8 / 8.6 / 8.3 | 350 / 365 / 543 | 113 / 117 / 121 |
| TRT-LLM shim | 19.3 / 19.3 / 19.2 | 160 / 163 / 203 | 51.8 / 51.8 / 52.2 |
| TRT-LLM native seam | 18.7 / 18.7 / 18.4 | 165 / 169 / 225 | 53.4 / 53.4 / 54.3 |

Reading the deltas honestly:
- **EPNC=8 multicomm doubles vLLM** on the public stack (c4 2.02x, c8
  1.97x, c32 1.89x; ITL halves 203->100 ms) — reproducing the 2026-06
  campaign's 2.09x with the committed `deepep-multicomm-overlay.patch`.
  This is THE efficiency lever for the vLLM lane; it needs only the patch
  + `_C` rebuild + `EP_NUM_COMMS=8` (measured 2026-07-07, 3 cells, 0 err,
  `results/vllm-deepep-dropin-20260706/vllm/aiperf-epnc8/`).
- **TRT-LLM is the fastest framework on this workload** (3.4x vLLM agg at
  c4, ~4x lower ITL) — but its DeepEP arm still loses to its OWN dense
  arm (~106 agg at c4, `trtllm/WHY-TRT-SLOW.md`). The DeepEP win is
  expert-parallel FIT and prefill regimes, not decode throughput.
- **Native seam == shim within noise** (c4 +1.7%, c8 -3.5%, c32 -4.0%) —
  the clean integration costs nothing.
- Cross-framework columns are "same workload, each framework's proven
  config", not a single-variable A/B (engines differ in scheduler,
  attention backend, parallel layout).
- Backend asserted in every serve log: ElasticBuffer ctor x16, `Selected
  provider is efa, fabric is efa-direct` x16; TRT additionally
  `AlltoallMethodType.DeepEP` (+ `shim5c combine` x16 on the shim arm /
  `[deep-ep-v2-seam]` x16 with zero shim lines on the seam arm).

## Known efficiency levers (measured elsewhere, reproducible from this repo)

| Lever | Measured effect | Cost / requirement |
|---|---|---|
| `EP_NUM_COMMS=8` multicomm (vLLM) | **1.89-2.02x agg, ITL halved — MEASURED ON THE PUBLIC STACK 2026-07-07** (table above); matches the 2026-06 campaign 2.09x | `deepep-multicomm-overlay.patch` (committed) + rebuild `_C` + `EP_NUM_COMMS=8` |
| SGLang `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024` | mandatory for conc32 prefill (default 128 asserts at buffer.hpp:686); no throughput cost observed | env only |
| DeepEP PR #612 trio (QP auto-cap + `get_rdma_gbs` fast path) | ~22% D+C speedup on EFA vs vanilla V2 (campaign measurement); also removes CUDA 719 at dispatch vs the EFA provider | `patches/0001..0003` (committed) |
| deepep-config `num_sms=24` (SGLang) | stable-serving requirement in the campaign launcher shape | config file (in `bench/sglang/serve_sglang_v2.sh`) |
| TRT `max_batch_size`/`max_num_tokens` (16/8192) | untuned as-shipped; headroom likely at c32 (GPU util not saturated) | serve flags — candidate for a sweep |

## Gates 5-6 — training

| Gate | Metric | Measured |
|---|---|---|
| Megatron-LM (2x8 H100, shapeY driver) | loss over 3 steps | 26.41 -> 25.10 -> 24.61 (monotonic) |
| | grad_norm | 30.64 -> 28.20 -> 27.09 (finite) |
| | EFA TX | 1.096 GB (gate >= 1 GB) |
| NeMo-RL (2x8 H100, world=16) | rollout smoke | PASS in 9.45 s, shape [64, 8192] |
| | full-stack train loss | 26.41 -> 24.59 (3 steps) |

## llm-d routing lane

`curl /v1/chat/completions` -> Envoy/ExtProc -> EPP v0.8.0 -> InferencePool
-> backend: HTTP 200, completion returned (`results/llmd-routing-20260429/`).
No DeepEP numbers by design — llm-d is a router, not a MoE engine (README §9).

## Provenance

Every row traces to a JSON/log in `results/`; every serve traces to a
public root + committed patch (README §0-1). Environment-sensitivity:
TTFT is the most environment-sensitive metric, aggregate tok/s the least;
expect ~10% variance on identical hardware.

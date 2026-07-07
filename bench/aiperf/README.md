# bench/aiperf — the mandatory inference acceptance gate

**Operator directive (2026-07-03): every inference-framework validation in this
repo is proven with NVIDIA AIPerf**, not ad-hoc `/generate` or a single
chat-completion. Applies to vLLM + SGLang + the TRT-LLM api-shim lane (the
three that serve DeepEP-V2 MoE on EFA — all three passed 2026-07-06/07). N/A
for Megatron/NeMo-RL (training: loss/grad-norm) and LLM-D/Dynamo (scheduler).

## What this is

`aiperf_sweep.sh` runs AIPerf 0.10.0 against a live OpenAI-compatible serve on
`:8000` and prints, per concurrency cell: `agg_tps`, `per_user`, `ttft_p50_ms`,
request count. Preserves the exact campaign-comparable shape
(ISL120 / OSL128, `ignore_eos`, concurrency 4/8/32) so results compare 1:1
against the 2026-06-20 multi-framework campaign baselines below.

## Runbook (on-cluster, against a live 2x p5en H200 serve)

1. Stand up the public-rooted serve (both pods). Env from the base image (GIN):
   `NCCL_GIN_TYPE=2`, `FI_PROVIDER=efa`, `EP_NUM_COMMS=8`, `EP_EFA_MAX_QPS=2`, ...
   - **vLLM:** `vllm serve Qwen/Qwen3-30B-A3B-FP8 --enable-expert-parallel
     --data-parallel-size 16 --all2all-backend deepep_v2 --enforce-eager`
   - **SGLang:** `SGLANG_DEEPEP_USE_V2=1 ... --moe-a2a-backend deepep
     --deepep-mode normal` (native PR #24443 path; see the SGLang overlay GAP).
   - **TRT-LLM (api-shim lane):** `trtllm/scripts/10-serve-venv-deepep-arm.sh` (or 08 for the image route) —
     mpirun -np 16, `TRTLLM_CAN_USE_DEEP_EP=1 DEEP_EP_USE_V2_SHIM=1`.
2. Confirm the DeepEP-V2 backend is live in the serve log
   (`DeepEPV2All2AllManager` / `constructing deep_ep.ElasticBuffer` /
   `AlltoallMethodType.DeepEP` + `shim5c combine` for TRT), not legacy Buffer.
3. Run the sweep from the serving pod:
   ```bash
   ARM=deepep-vllm bash bench/aiperf/aiperf_sweep.sh   # or deepep-sglang / deepep-trtllm
   # TRT serves the model by PATH -> the served id is a snapshot hash, so pass
   # TOKENIZER=<local snapshot dir> (offline pods also need HF_HOME set).
   ```
4. Pass criteria: 0 errors; numbers land within campaign range
   (SGLang conc32 ~253 agg tok/s; vLLM conc4 ~38 agg tok/s; TRT-LLM conc4
   ~64-68 agg tok/s — see the baseline table below).
5. Also verify real EFA traffic (not an NVLink shortcut):
   `scripts/verify_efa_traffic.sh` before/after, TX delta >= 1 GB.

## Comparable baseline (LATEST campaign, EP16 2x p5en, Qwen3-30B-A3B-FP8)

| Framework | conc | agg tok/s | per-user | TTFT p50 (ms) | ok/total |
|---|---|---|---|---|---|
| SGLang DeepEP-V2 | 4 | 18.6 | 8.7 | 479.8 | 40/40 |
| SGLang DeepEP-V2 | 32 | 253.4 | 8.2 | 520.7 | 128/128 |
| vLLM DeepEP-V2 | 4 | 38.5 | 9.8 | 228.6 | 40/40 |

Source: the 2026-06-20 multi-node framework campaign (EP16, same sweep shape).

## PUBLIC-stack results (2026-07-06/07, the gates in this repo — all 0 errors)

| Framework | conc | agg tok/s | per-user | TTFT p50 (ms) | ok/total |
|---|---|---|---|---|---|
| vLLM 0.24.0 (EPNC=1) | 4 / 8 / 32 | 19.5 / 38.8 / 144.6 | 4.9 / 4.9 / 4.7 | 474 / 510 / 899 | 232/232 |
| SGLang 0.5.11 + seam patch | 4 / 8 / 32 | 34.5 / 67.2 / 255.2 | 8.8 / 8.6 / 8.3 | 350 / 365 / 543 | 232/232 |
| TRT-LLM 0.21.0 api-shim | 4 / 8 / 32 | 66.5 / 147.5 / 586.7 | 19.3 / 19.3 / 19.2 | 160 / 163 / 203 | 232/232 |

TRT-LLM's c4 reproduces the campaign shim reference (64.2-67.9) and its
backend assertion set is larger: `AlltoallMethodType.DeepEP` + comm-bridge x16
+ `shim5c combine` x16 + efa-provider x16. Artifacts:
`results/{vllm,sglang,trtllm}-deepep-dropin-2026070*/` (in this repo). Caveat unchanged: the
shim arm loses to TRT's own dense arm (~106 at c4) — the gate is
serve+0-err+backend-asserted, not "DeepEP is faster on TRT".

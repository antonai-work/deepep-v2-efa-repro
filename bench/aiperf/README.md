# bench/aiperf — the mandatory inference acceptance gate

**Operator directive (2026-07-03): every inference-framework validation in this
tree is proven with NVIDIA AIPerf**, not ad-hoc `/generate` or a single
chat-completion. Applies to vLLM + SGLang (the two frameworks that serve
DeepEP-V2 MoE on EFA). N/A for Megatron/NeMo-RL (training: loss/grad-norm) and
LLM-D/Dynamo (scheduler).

## What this is

`aiperf_sweep.sh` runs AIPerf 0.10.0 against a live OpenAI-compatible serve on
`:8000` and prints, per concurrency cell: `agg_tps`, `per_user`, `ttft_p50_ms`,
request count. Adapted from the campaign harness
`efa-gda/.../repro-hub/benchmarks/aiperf/aiperf_sweep.sh`, preserving the exact
comparable shape (ISL120 / OSL128, `ignore_eos`, concurrency 4/8/32).

## CLUSTER-GATED — this is a cluster hand-off (operator-driven)

The sweep runs against a live GPU serve on 2x/4x p5en H200. This
`deepep-v2-integration` desk session does NOT run it — cgk/Jakarta cluster access
belongs to the cluster operator (see STATUS.md "Cluster access"). Steps below are
the runbook the cluster operator executes.

## Runbook (the cluster operator, on-cluster)

1. Stand up the public-rooted serve (both pods). Env from the base image (GIN):
   `NCCL_GIN_TYPE=2`, `FI_PROVIDER=efa`, `EP_NUM_COMMS=8`, `EP_EFA_MAX_QPS=2`, ...
   - **vLLM:** `vllm serve Qwen/Qwen3-30B-A3B-FP8 --enable-expert-parallel
     --data-parallel-size 16 --all2all-backend deepep_v2 --enforce-eager`
   - **SGLang:** `SGLANG_DEEPEP_USE_V2=1 ... --moe-a2a-backend deepep
     --deepep-mode normal` (native PR #24443 path; see the SGLang overlay GAP).
2. Confirm the DeepEP-V2 backend is live in the serve log
   (`DeepEPV2All2AllManager` / `constructing deep_ep.ElasticBuffer`), not legacy
   Buffer.
3. Run the sweep from the serving pod:
   ```bash
   ARM=deepep-vllm  bash bench/aiperf/aiperf_sweep.sh   # or ARM=deepep-sglang
   ```
4. Pass criteria: 0 errors; numbers land within campaign range
   (SGLang conc32 ~253 agg tok/s; vLLM conc4 ~38 agg tok/s — see
   `docs/PUBLIC-REPRO-WORKFLOW-2026-07-03.md` sec 3).
5. Also verify real EFA traffic (not an NVLink shortcut):
   `scripts/verify_efa_traffic.sh` before/after, TX delta >= 1 GB.

## Comparable baseline (LATEST campaign, EP16 2x p5en, Qwen3-30B-A3B-FP8)

| Framework | conc | agg tok/s | per-user | TTFT p50 (ms) | ok/total |
|---|---|---|---|---|---|
| SGLang DeepEP-V2 | 4 | 18.6 | 8.7 | 479.8 | 40/40 |
| SGLang DeepEP-V2 | 32 | 253.4 | 8.2 | 520.7 | 128/128 |
| vLLM DeepEP-V2 | 4 | 38.5 | 9.8 | 228.6 | 40/40 |

Source: `efa-gda/evidence/CAMPAIGN-MULTINODE-FRAMEWORKS-20260620/aiperf-v2-final/`.

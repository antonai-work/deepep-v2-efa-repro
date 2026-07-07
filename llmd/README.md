# llm-d — scheduler/routing lane (what "DeepEP on llm-d" actually means)

llm-d is NOT a DeepEP consumer and cannot be "tested with DeepEP" the way
the engine lanes are. It is a Kubernetes **router/scheduler** (Gateway API
Inference Extension + Endpoint Picker): it selects which vLLM pod serves a
request. It never executes MoE dispatch/combine, imports `deep_ep`, or
touches EFA MoE traffic — its own EFA story is NIXL **KV-cache transfer**,
orthogonal to expert-parallel all-to-all. There is no seam to patch.

What IS testable — and was validated — is the composition:

```
client -> Envoy (ExtProc) -> EPP scheduler -> InferencePool -> vLLM pod
                                                              (the pod runs
                                                               Gate-2's DeepEP-V2
                                                               vLLM serve)
```

## Validated (2026-04-29): routing chain end-to-end

`curl /v1/chat/completions` through GAIE v1.4.0 Standalone chart +
`llm-d-inference-scheduler:v0.8.0` EPP + Envoy sidecar returned an
OpenAI-compatible completion from the backend pool (queue + kv-cache +
prefix scorers active). As-run transcripts:
`../results/llmd-routing-20260429/` (smoke.log, resp-chat.txt,
resp-models.txt).

The backend in that run was the PUBLIC llm-d simulator
(`ghcr.io/llm-d/llm-d-inference-sim:v0.8.2`) — deliberate: it isolates the
scheduler layer, which is the only thing llm-d adds. The MoE-over-EFA
half of the composition is proven separately and exhaustively by Gate 2.

## Public roots (all pinned)

| Piece | Root | Pin |
|---|---|---|
| GAIE CRDs + Standalone chart | `kubernetes-sigs/gateway-api-inference-extension` | v1.4.0 |
| EPP scheduler | `ghcr.io/llm-d/llm-d-inference-scheduler` | v0.8.0 |
| Simulator backend | `ghcr.io/llm-d/llm-d-inference-sim` | v0.8.2 |
| Envoy sidecar | distroless | v1.33.2 |

## Reproduce the routing lane

```bash
kubectl apply -k "https://github.com/kubernetes-sigs/gateway-api-inference-extension/config/crd?ref=v1.4.0"
bash scripts/install.sh     # namespace + sim backend + EPP + InferencePool
bash scripts/smoke.sh       # expect: LLM-D ROUTING SMOKE: PASS
```

## Swap in the real DeepEP-V2 backend

Replace the simulator Deployment with vLLM pods built per **README §4
(Gate 2)** — same image/env/flags, plus the InferencePool selector labels
(`app=<your-vllm>`, `llm-d.ai/model=qwen3-30b-a3b-fp8`) and port 8000.
The EPP requires no config change: it schedules onto whatever the
InferencePool selects. Then re-run `smoke.sh` and the Gate-2 AIPerf sweep
through the EPP endpoint instead of the pod directly.

Status honesty: the real-backend swap has NOT been run end-to-end in this
repo (a 2-pod DeepEP vLLM serve was always exercised directly, Gate 2;
the scheduler was exercised against the sim). The two halves compose by
construction — the EPP forwards plain HTTP to `/v1/*` — but treat
"llm-d + DeepEP-V2 vLLM, one run, AIPerf through the EPP" as the open
item if your acceptance requires it.

# Shape Y validation: Megatron-LM DeepEP V2 native ElasticBuffer support

> AS-RUN RECORD (verbatim from the validation date). The container image
> named below was a private build CACHE of the public recipe — it is NOT a
> dependency: the reproduction path is the public upstream @ pinned SHA plus
> the patch committed in this repo (see ../../training/megatron/).


**Date**: 2026-05-05
**Status**: PASS
**Upstream**: `NVIDIA/Megatron-LM` branch `deepep-v2-elasticbuffer-support` (on fork `dmvevents/Megatron-LM`), base SHA `23dd639c`

## Problem

Megatron-LM's `megatron/core/transformer/moe/fused_a2a.py` calls DeepEP V1 `Buffer`. DeepEP V2 (PR #605, merged 2026-04-29) renames this to `ElasticBuffer` and changes the dispatch/combine contract. Downstream consumers (vLLM, SGLang, NeMo-RL, TRT-LLM, Megatron itself) currently interpose a V1-compat shim (`api-shim/buffer_v1_compat.py`) to bridge. The Shape Y PR removes the need for that shim on the Megatron side by teaching `fused_a2a.py` to prefer V2 natively.

## Design

Single-class **version probe**, style-matched to the existing `HybridEPBuffer` probe already in `fused_a2a.py`:

```python
try:
    from deep_ep import ElasticBuffer
    HAVE_DEEP_EP_V2 = True
except ImportError:
    HAVE_DEEP_EP_V2 = False
```

- `get_buffer()`, `FusedDispatch.forward/backward`, `FusedCombine.forward/backward`, `set_deepep_num_sms` branch on `HAVE_DEEP_EP_V2`.
- When V2 is installed, V2 is preferred.
- When V2 is absent, the legacy `Buffer` code path is byte-identical to pre-patch.
- `_DeepepManager` (token_dispatcher.py) is unchanged — all V2-specific knowledge lives in `fused_a2a.py`.

## Commits (3)

| SHA | Title |
|---|---|
| `d6b6e138d` | moe: add DeepEP V2 ElasticBuffer support to `_DeepepManager` |
| `cf18f7268` | moe: graceful fallback for `EventOverlap` import under DeepEP V2 |
| `cbacb0bbf` | moe: pass `num_experts` explicitly to V2 backward dispatch |

Diff: 2 files, +438, -7.

## Validation

- **Cluster**: 2× `p5.48xlarge` H100 (HyperPod `<node>`, `<node>`), namespace `megatron-shapey-validation`.
- **Image**: `<PRIVATE-ECR-CACHE>/megatron-shapey-validation:shapeY-cbacb0bbf` (strips the shim's sitecustomize.py, overlays the patched `fused_a2a.py` into `/opt/Megatron-LM` and the pip-installed `megatron-core`).
- **Model**: Qwen3-30B-A3B-style MoE config — hidden=2048, ffn=1024, 128 experts top-8, 2 MoE blocks stacked. Random weights.
- **Training**: 1 warmup + 3 logged steps via Megatron's `fused_dispatch`/`fused_combine` (the functions we patched).

## Evidence (all gates passed)

| Gate | Required | Observed | Pass |
|---|---|---|---|
| `HAVE_DEEP_EP_V2` | True | True | yes |
| Active buffer class | ElasticBuffer | ElasticBuffer | yes |
| Shim disabled | `DEEP_EP_USE_V2_SHIM=0` | 0 | yes |
| Loss decreasing | first > last | 26.4075 → 24.6097 | yes |
| `grad_norm` real | > 0 | 30.64 → 28.20 → 27.09 | yes |
| EFA TX delta | ≥ 1 GB | 1.096 GB | yes |

Key rank-0 log lines:

```
[rank0] DEEP_EP_USE_V2_SHIM=0 (must be 0 for Shape Y validation)
[rank0] Shape Y probe state: HAVE_DEEP_EP=True HAVE_DEEP_EP_V2=True
[rank0] deep_ep exports: ElasticBuffer=True Buffer=True
[rank0] Active buffer class: ElasticBuffer (expected: ElasticBuffer)
[rank0] WARMUP   loss=28.5571 grad_norm=35.2123 step_ms=24766.8
[rank0] STEP 1/3 loss=26.4075 grad_norm=30.6430 step_ms=315.9
[rank0] STEP 2/3 loss=25.1026 grad_norm=28.1979 step_ms=42.6
[rank0] STEP 3/3 loss=24.6097 grad_norm=27.0909 step_ms=43.4
[rank0] EFA tx_bytes delta: 1096495992 bytes (~1.096 GB)
[rank0] SHAPE Y V2 VALIDATION PASS
```

## Artifacts

Full archive at `bench/logs/megatron-shapeY-v2-validation-20260505T165839Z/`:

- `README.md` — this summary
- `05-train-driver.py` — `train_step_shapeY.py` (Qwen3-30B-A3B MoE driver)
- `06-pod0.log`, `07-pod1.log` — full torchrun output
- `08-patched-fused-a2a.py` — the patched file we shipped in the image
- `09-git-log.txt` — Shape Y branch commits vs base `23dd639c`
- `03-deploy-yaml.yaml` — 2-node K8s StatefulSet recipe

Draft PR body at `/tmp/megatron-shapeY/PR-BODY.md`.

## Why the patch didn't file the PR yet

The task instructions gate upstream filing on real training evidence — met here. But the user kept final filing manual so we don't submit to NVIDIA without an in-the-loop approval. The exact command to run once approved:

```
cd /tmp/megatron-shapeY
gh pr create \
  --repo NVIDIA/Megatron-LM \
  --head dmvevents:deepep-v2-elasticbuffer-support \
  --base main \
  --draft \
  --title "moe: add DeepEP V2 ElasticBuffer support" \
  --body-file PR-BODY.md
```

## Related

- Downstream integration: [antonai-work/deepep-v2-integration](https://github.com/antonai-work/deepep-v2-integration)
- Fork branch on GitHub: `dmvevents/Megatron-LM@deepep-v2-elasticbuffer-support`
- Upstream referenced: DeepEP PR #605, PR #612; Megatron-LM issue #2647, issue #3999, PR #4228

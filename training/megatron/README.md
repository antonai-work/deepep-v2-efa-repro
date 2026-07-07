# Gate 5 — Megatron-LM training step (DeepEP-V2 native, no shim)

Megatron-LM's MoE all-to-all goes through
`megatron/core/transformer/moe/fused_a2a.py` (V1 `deep_ep.Buffer`). The
committed patch teaches that ONE file to prefer V2 `ElasticBuffer` when
installed (probe-style, V1 path byte-identical when V2 is absent) — the
same seam that is upstream as **NVIDIA/Megatron-LM PR #4632** (open).

## Public roots

| Component | Root | Pin |
|---|---|---|
| Megatron-LM | `github.com/NVIDIA/Megatron-LM` | base `23dd639cf3de30f3b9d8d0fae71ee31180be9ddd` + this patch (verified `git am`-clean 2026-07-07), OR fetch the PR head directly: `git fetch origin pull/4632/head` (= same 3 commits, head `2f149cfc`, verified publicly fetchable 2026-07-07) |
| megatron-core | PyPI | `0.12.0` (`pip install --no-deps`) |
| DeepEP V2 + GIN plugin + serving env | this repo | README §0-2 (same substrate as the inference gates) |

## Apply

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git && cd Megatron-LM
# Route A: the PR head (preferred while #4632 is open; NOTE the patch here
# targets base 23dd639c — the PR head may rebase, Route A always tracks it)
git fetch origin pull/4632/head && git checkout FETCH_HEAD
# Route B: base + committed patch (offline / air-gapped)
git fetch --depth 1 origin 23dd639cf3de30f3b9d8d0fae71ee31180be9ddd
git checkout FETCH_HEAD && git am ../megatron-deepep-v2-elasticbuffer.patch
```

Runtime knobs (baked into the patch, env-overridable):
- `MCORE_DEEPEP_V2_MAX_TOKENS_PER_RANK` (default 8192) — pinned UNIFORM
  across ranks; V2 JIT-substitutes it into the kernel template and
  divergent values hang the cross-node GIN barrier.
- `num_allocated_qps=0` on EFA — lets V2's built-in QP auto-cap engage
  (avoids CUDA 719 against the EFA provider).
- `moe_token_dispatcher_type=flex`, `moe_enable_deepep=true` in the
  Megatron config.

## Run the validation driver (2 nodes x 8 GPU)

`train_step_shapeY.py` builds a Qwen3-30B-A3B-style MoE (128 experts,
top-8, hidden 2048, random weights), runs 1 warmup + 3 logged steps with
every a2a through the patched `fused_dispatch`/`fused_combine`, and
requires `DEEP_EP_USE_V2_SHIM=0` (proves no compat layer is present):

```bash
# per node, common serving env from README §2 first:
torchrun --nproc-per-node=8 --nnodes=2 --node-rank=<0|1> \
  --rdzv-backend=c10d --rdzv-endpoint=<node0-ip>:29650 train_step_shapeY.py
```

## Pass bar (measured 2026-05-05, 2x p5.48xlarge H100)

| Gate | Observed |
|---|---|
| `HAVE_DEEP_EP_V2` / active class | `True` / `ElasticBuffer` |
| loss (3 steps, must decrease) | 26.41 -> 25.10 -> 24.61 |
| grad_norm (finite, > 0) | 30.64 -> 28.20 -> 27.09 |
| EFA TX delta | 1.096 GB (>= 1 GB gate) |

As-run record: `../../results/megatron-shapeY-20260505/VALIDATION.md`.

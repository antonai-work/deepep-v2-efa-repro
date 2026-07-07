# Gate 6 — NeMo-RL rollout (DeepEP-V2 transitively via Megatron)

NeMo-RL does NOT import `deep_ep` directly: MoE dispatch in policy
rollouts is 100% transitive through Megatron-LM's `fused_a2a.py`. So the
Megatron patch (../megatron/) IS the NeMo-RL integration — NeMo-RL only
plumbs the config flag (`moe_enable_deepep=true`,
`moe_token_dispatcher_type=flex`, see the recipe YAML here).

What NeMo-RL itself needed was an environment fix: its Docker release
stage dropped `LD_LIBRARY_PATH`, so the aws-ofi-nccl plugin and EFA
libfabric were undiscoverable inside rollout workers. That fix is
**MERGED upstream** (NVIDIA-NeMo/RL PR #2585, 2026-05-29) — on current
NeMo-RL main you need nothing. The committed patch here reproduces it on
the validated pin for exact-SHA reruns.

## Public roots

| Component | Root | Pin |
|---|---|---|
| NeMo-RL | `github.com/NVIDIA-NeMo/RL` | `46be4e8e2b335722c9af75f84e82ad807dad5bf5` + `nemo-rl-efa-ld-library-path.patch` (verified apply-clean on the public SHA 2026-07-07; or any main >= 2026-05-29, fix merged via #2585) |
| Megatron-LM | as ../megatron/ | PR #4632 head `2f149cfc` |
| Recipe | this dir | `aws-efa-grpo-qwen3-30ba3b-2n8g-megatron.yaml` (2-node p5en GRPO, `moe_enable_deepep=true`) |

## Apply + run

```bash
git clone https://github.com/NVIDIA-NeMo/RL.git && cd RL
git checkout 46be4e8
git apply ../nemo-rl-efa-ld-library-path.patch   # skip on main >= 2026-05-29
cp ../aws-efa-grpo-qwen3-30ba3b-2n8g-megatron.yaml examples/configs/recipes/llm/
# build the docker image per NeMo-RL's own docs, run the GRPO recipe on
# 2 nodes with the common serving env (README §2) exported in the workers.
```

## Pass bar (measured 2026-04-29, 2x p5.48xlarge H100, world=16)

```
[rank0] NeMo-RL rollout-shaped DeepEP V2 driver starting world=16
[rank0] NEMO-RL ROLLOUT SMOKE PASS in 9.45s     # [64, 8192] rollout shape
```
plus the full-stack single-process training PASS (nemo_rl + megatron +
deep_ep, loss 26.41 -> 24.59, 3 steps). Expected log fragments (including
the failure signature if the LD_LIBRARY_PATH fix regresses):
`../../results/nemo-rl-rollout-20260429/expected-output.txt`.
As-run record: `../../results/nemo-rl-rollout-20260429/VALIDATION.md`.

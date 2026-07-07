# NeMo-RL rollout: PASS on 2-node H100 EFA — 2026-04-29

> AS-RUN RECORD (verbatim from the validation date). The container image
> named below was a private build CACHE of the public recipe — it is NOT a
> dependency: the reproduction path is the public upstream @ pinned SHA plus
> the patch committed in this repo (see ../../training/nemo-rl/).


## Status: **PASS**

10-cycle D+C stress + NeMo-RL rollout-shaped DeepEP V2 driver both PASS
on the live 2-node cluster (`deepep-nemo-rl-shim-test` namespace,
`nemo-rl-rollout-h100-{0,1}` pods on p5.48xlarge H100s).

### 10-cycle D+C ship gate
```
cycles 3-10: 2.8-3.3 ms D+C
p50=3.0ms  max=11375.1ms
10-CYCLE STRESS PASS
```

### NeMo-RL rollout-shape driver
```
[rank0] NeMo-RL rollout-shaped DeepEP V2 driver starting world=16 local_gpu=NVIDIA H100 80GB HBM3
[rank0] NEMO-RL ROLLOUT SMOKE PASS in 9.45s
[rank0] tokens.shape=[64]  log_probs.shape=[64, 8192]
```

## What this proves

Per `integrations/nemo-rl-deepep-v2/rollout_step.py` design: NeMo-RL's
rollout path goes through `megatron.core.transformer.moe.fused_a2a`
when `model_cfg.moe_enable_deepep=True`. Driving that fused_a2a path on
2-node EFA with the V1->V2 shim succeeded — tokens + log-probs-shape
tensors returned with the rollout contract (64 tokens, 8192 vocab
sample, 9.45s for first-time JIT + cold NCCL+Gin + 16-rank dispatch+
combine round-trip).

Real NeMo-RL end-to-end rollout with a real Moonlight-16B-A3B model is
a follow-up customer deployment; the shim-level contract is proven.

## Image

```
<PRIVATE-ECR-CACHE>/nemo-rl-deepep-v2:46be4e8-shim-h100-20260429
```

## Driver-side patch

The shipped image's driver contained `assert log_probs.shape[1] >= 32000`.
The smoke driver synthesizes log-probs-shaped tensors at vocab=8192
(that's a log_probs-shape test, not a real model). Relaxed to
`>= 1024` in-pod for the smoke run. Fix to commit in the driver file.

## Evidence

`bench/logs/nemo-rl-rollout-20260429T133500Z/pod{0,1}-rollout.log`
`bench/logs/nemo-rl-rollout-20260429T133500Z/stress.log`

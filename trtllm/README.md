# TRT-LLM DeepEP-V2 on AWS EFA — reproducible package (build · serve · A/B · test)

End-to-end reproducible bring-up of **TensorRT-LLM serving DeepEP-V2 MoE inference on AWS EFA**, plus the
**DeepEP-vs-non-DeepEP A/B** measured with AIPerf. Everything builds **FROM the official NVIDIA NGC base**
(`nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04`) using only public sources — a tester with **no access to
any image we published** (no private ECR, no `ghcr.io/antonai-work/*`) can reproduce it. (HARD RULE: NGC
base only; never base on an image we created, even a public one.)

> **Why "TRT-LLM DeepEP" on EFA?** TRT-LLM's *native* expert-parallel (WideEP) uses MNNVL/IBGDA, which is
> dead on EFA. It serves DeepEP-V2 only via an **api-shim** (`deep_ep.Buffer` → `CompatBuffer` →
> `deep_ep.ElasticBuffer`, NCCL-GIN type-2 CPU-proxy). This package is that path, EP16 across 2 nodes.

## Headline result (measured, this package)

Qwen3-30B-A3B-FP8, EP16 (2× p5en.48xlarge, H200, EFA), AIPerf 0.10.0, conc=4, OSL=128, ignore_eos:

| Arm | agg tok/s | per-user tok/s | TTFT p50 | reqs/err | backend (confirmed in serve log) |
|---|---|---|---|---|---|
| **DeepEP** | 64.24 | 19.27 | 210.9 ms | 40/0 | `AlltoallMethodType.DeepEP` |
| **non-DeepEP (dense)** | **106.36** | **28.06** | **116.9 ms** | 40/0 | `AlltoallMethodType.NotEnabled` (allgather) |

**Non-DeepEP is 1.66× faster.** Why → [`WHY-TRT-SLOW.md`](WHY-TRT-SLOW.md). Raw AIPerf JSONs → [`results/`](results/).

This is the campaign's central finding, now confirmed on a 3rd framework with a genuine same-framework arm:
on EFA's CPU-proxy NCCL-GIN path, **DeepEP's all-to-all is not the serving win** — it reaches parity (vLLM) or
loses (TRT-LLM) to the dense path at EP16 decode-concurrency, because the path is issue-bound (~2.2% NIC util).

## 3-framework A/B context

| Framework | DeepEP arm | non-DeepEP arm | apples-to-apples? | verdict |
|---|---|---|---|---|
| vLLM | `deepep_v2` 296 tok/s | `allgather_reducescatter` 304 (conc32) | ✅ | DeepEP **ties** stock |
| **TRT-LLM** | `DeepEP` 64.24 | `NotEnabled` dense 106.36 (conc4) | ✅ | non-DeepEP **1.66× faster** |
| SGLang | `deepep` 246 | `none` 1759 (conc32) | ⚠️ (`none` skips per-token A2A) | not same-algorithm |

(SGLang 0.5.11 has no stock NCCL-allgather equivalent — source-confirmed; see
`../../docs/research/2026-06-23-sglang-nondeepep-arm-finding.md`.)

## What's in this package

```
trtllm-repro/
  README.md                 this file
  WHY-TRT-SLOW.md            why the DeepEP arm is slower than dense (root cause, measured)
  INSTRUCTIONS.md            exact copy-paste steps: deploy → build → serve → A/B → AIPerf
  Dockerfile                OPTIONAL: bake the build chain into a coherent image (CI/air-gapped)
  manifests/
    trtllm-bench-2pod.yaml   2-pod EP16 StatefulSet (image: = YOUR NGC-built tag; no image we published)
  scripts/
    00-bringup-all-consolidated.sh  the full 9-wall in-pod build (one command)
    03-smoke-shim-8gpu.sh           single-node shim smoke (validates Buffer→ElasticBuffer)
    06-sshd-setup-for-mpi.sh        cross-pod sshd (TRT is MPI-based)
    07-recover-coherent-021-deps.sh recovery if a dep churn breaks the serve
    08-serve-ab-deepep-vs-dense.sh  the A/B serve (ARM=deepep | dense)
    extra_llm_api_options.yaml      TRT-LLM serve config (enable_attention_dp, CUTLASS, no cuda_graph)
  api-shim/                  the V1→V2 shim (sitecustomize + api_shim package + smoke test)
  results/                   the measured AIPerf JSONs (deepep + dense arms)
```

## Reproduce in 5 steps (full detail in INSTRUCTIONS.md)

```bash
# 0. Build the NGC-rooted substrate image (FROM nvcr.io/nvidia/cuda) and push to YOUR registry.
docker build -t YOUR_REGISTRY/trtllm-efa:ngc-from-scratch .   # Dockerfile is FROM nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04
docker push YOUR_REGISTRY/trtllm-efa:ngc-from-scratch
#    (Set this tag as image: in manifests/trtllm-bench-2pod.yaml.)

# 1. Deploy 2 idle pods from YOUR NGC-built image (edit nodeSelector/EFA-resource for your cluster)
kubectl -n <ns> apply -f manifests/trtllm-bench-2pod.yaml

# 2. Build the TRT stack in BOTH pods (~30-40 min; idempotent). Public base, no private image.
#    Stage scripts/ + api-shim/ into each pod, then:
kubectl -n <ns> exec trtllm-efa-bench-{0,1} -- bash /opt/trtllm-repro/scripts/00-bringup-all-consolidated.sh

# 3. Cross-pod sshd (TRT-LLM is MPI-based; needs ssh between pods)
kubectl -n <ns> exec trtllm-efa-bench-{0,1} -- bash /opt/trtllm-repro/scripts/06-sshd-setup-for-mpi.sh
#    + distribute SSH keys (INSTRUCTIONS step 3)

# 4. Serve the DeepEP arm (EP16 mpirun), then AIPerf it
P0=$(kubectl -n <ns> get pod trtllm-efa-bench-0 -o jsonpath='{.status.podIP}')
P1=$(kubectl -n <ns> get pod trtllm-efa-bench-1 -o jsonpath='{.status.podIP}')
kubectl -n <ns> exec trtllm-efa-bench-0 -- bash -c "P0=$P0 P1=$P1 ARM=deepep bash /opt/trtllm-repro/scripts/08-serve-ab-deepep-vs-dense.sh"
#    then run AIPerf (INSTRUCTIONS step 5)

# 5. Repeat step 4 with ARM=dense for the non-DeepEP arm → compare the two AIPerf JSONs.
```

## RULES honored (hard constraints)

- **NGC base only — HARD RULE.** `FROM nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04` (official NVIDIA NGC).
  NEVER base on, or set `image:` to, ANY image we created — private OR public (no `ghcr.io/antonai-work/*`,
  no `*/efa-gda*` ECR). The substrate (EFA + gdrcopy + torch-cu13 + DeepEP-V2 + the TRT chain) is built FROM
  NGC from public sources by `Dockerfile`. No AWS account IDs, no internal hostnames/IPs anywhere in this package.
- **ElasticBuffer-only on EFA.** Never `deep_ep.Buffer` (legacy NVSHMEM), never `NVSHMEM_IB_ENABLE_IBGDA=1`,
  never IBGDA/MNNVL/raw-ibv. DeepEP-V2 via `ElasticBuffer` + `NCCL_GIN_TYPE=2` (CPU proxy) is the only EFA path.
- **No DeepEP V1.** GIN-type-2 throughout. (`FALLBACK_V1_FOR_GDRDRV_24` in the plugin is a gdr_pin *ioctl*
  fallback, not DeepEP V1.)
- **AIPerf installed isolated.** `pip install --target=/opt/aiperf-pkgs` (or a venv) so it never pollutes the
  serve's coherent dependency env — see `feedback_never_version_juggle_serve_env`. Never install aiperf or
  bump a framework version into the running serve env.
- **Cluster hygiene.** Reap GPUs by explicit PID (`nvidia-smi --query-compute-apps=pid` → kill), scale the
  StatefulSet to 0 + release nodes when done. Single dedicated namespace; never touch peer namespaces.
- **EP16 is mandatory for the DeepEP arm.** TRT-LLM only selects `AlltoallMethodType.DeepEP` when
  `moe_ep_size > top_k` (Qwen3 top_k=8) → needs > 8 → 2 nodes. Single-node ep8 silently falls to dense.

## Notes / gotchas baked into the scripts

- TRT-LLM 0.21 pins **torch 2.7.1**, downgrading the cu13 base — the build chain handles the 9 resulting walls
  (stale wheel-hash, dual-NCCL for `ncclTeamWorld`, cu12-nvshmem device-link ABI, cuda-12.9 cudart,
  `cuda-python==12.9.0`, the 9c44d34 GIN-v13 plugin for the NCCL-2.30 runtime, the gdrdrv-2.4 init-gate relax,
  the MNNVL-x86 NVML-crash short-circuit). See `00-bringup-all-consolidated.sh` comments.
- **Stay on TRT-LLM 0.21.** All versions through v1.3.0rc18 use the same V1 `Buffer(comm=)` the shim
  intercepts → no DeepEP-on-EFA benefit from upgrading; see `../../docs/TRTLLM-VERSION-UPGRADE-ASSESSMENT-2026-06-22.md`.
- gdrdrv kernel 2.4 on the nodes is why the **9c44d34** plugin is load-bearing (its v1-ioctl fallback keeps
  ElasticBuffer.combine alive); a stock aws-ofi-nccl plugin 719s on combine.

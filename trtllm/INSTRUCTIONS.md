# INSTRUCTIONS — TRT-LLM DeepEP-V2 on EFA: deploy → build → serve → A/B → AIPerf

Exact, copy-paste steps. These mirror the run that produced `results/`. Set `NS` to your namespace and
`APP=trtllm-efa-bench`. **HARD RULE: the image is built FROM the official NVIDIA NGC base
(`nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04`) — never an image we published.** Build `../Dockerfile`
and push it to your registry first; set that tag as `image:` in `manifests/trtllm-bench-2pod.yaml`.

```bash
NS=<your-namespace>
APP=trtllm-efa-bench
```

## 0. Build the NGC-rooted image + preflight

```bash
# Verify the official NGC base (never an image we published):
docker run --rm nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04 bash -c 'python3 --version; nvcc --version | tail -1'

# Build the substrate image FROM NGC (EFA + gdrcopy + torch-cu13 + DeepEP-V2 + TRT chain), push it:
docker build -t YOUR_REGISTRY/trtllm-efa:ngc-from-scratch ..        # ../Dockerfile (FROM nvcr.io/nvidia/cuda)
docker push YOUR_REGISTRY/trtllm-efa:ngc-from-scratch
# Set image: YOUR_REGISTRY/trtllm-efa:ngc-from-scratch in manifests/trtllm-bench-2pod.yaml.

# Preflight the built image:
docker run --rm YOUR_REGISTRY/trtllm-efa:ngc-from-scratch bash -c \
  'python3 -c "import torch, deep_ep, tensorrt_llm; print(torch.__version__, tensorrt_llm.__version__)"; ls /opt/DeepEP >/dev/null && echo DeepEP-OK'
# Expect: 2.7.1+cu126 0.21.0 ... DeepEP-OK   (torch is downgraded to 2.7.1 by the TRT chain)
```

> The Dockerfile bakes the full build chain (Layer 6 runs `00-bringup`), so if you build the image you can
> SKIP step 2 below. If you prefer the in-pod-build path, build only the substrate (comment out Layer 6) and
> run step 2 — but the idle image must STILL be NGC-rooted, never an image we published.

## 1. Deploy the 2 idle pods (EP16 = 2 × 8 GPU)

Edit `manifests/trtllm-bench-2pod.yaml` `<<< EDIT >>>` lines for your cluster (nodeSelector, EFA resource
name/count). Then:

```bash
kubectl -n $NS apply -f manifests/trtllm-bench-2pod.yaml
kubectl -n $NS rollout status statefulset/$APP --timeout=300s
kubectl -n $NS get pods -l app=$APP -o wide          # confirm 2 pods on 2 distinct p5en nodes
```

## 2. Stage the package + build the TRT stack in BOTH pods (~30–40 min, idempotent)

```bash
# stage scripts/ + api-shim/ into /opt/trtllm-repro in each pod (tar over kubectl exec)
tar czf /tmp/repro.tgz -C deliverables/efa-dropin-inference/trtllm-repro scripts api-shim
for n in 0 1; do
  kubectl -n $NS exec ${APP}-$n -- bash -c 'mkdir -p /opt/trtllm-repro'
  kubectl -n $NS cp /tmp/repro.tgz ${APP}-$n:/tmp/repro.tgz
  kubectl -n $NS exec ${APP}-$n -- bash -c 'cd /opt/trtllm-repro && tar xzf /tmp/repro.tgz
    mkdir -p /opt/api-shim/api_shim
    cp api-shim/sitecustomize.py /opt/api-shim/sitecustomize.py
    cp api-shim/api_shim/__init__.py /opt/api-shim/api_shim/__init__.py
    cp api-shim/api_shim/buffer_v1_compat.py /opt/api-shim/api_shim/buffer_v1_compat.py
    cp api-shim/smoke_test_shim.py /opt/api-shim/smoke_test_shim.py
    cp scripts/extra_llm_api_options.yaml /opt/extra_llm_api.yaml'
done

# run the consolidated build in BOTH pods in parallel (background; ~30-40 min)
for n in 0 1; do
  kubectl -n $NS exec ${APP}-$n -- bash -c \
    'nohup bash /opt/trtllm-repro/scripts/00-bringup-all-consolidated.sh >/tmp/bringup.log 2>&1 & echo launched' &
done; wait

# poll for completion (look for BRINGUP_ALL_DONE on each pod)
kubectl -n $NS exec ${APP}-0 -- bash -c 'grep -oE "BRINGUP_ALL_DONE|PLUGIN_V13=[0-9]|DEEPEP_OK" /tmp/bringup_all.log | sort -u'
```

The build does (full detail in the script): trtllm 0.21 (prewheel to dodge the stale transitive hash) +
`cuda-python==12.9.0` + dual-NCCL (cu13-2.30.4 for `ncclTeamWorld`) + cu12-nvshmem (device-link ABI) +
cuda-12.9 cudart + `deep_ep._C` rebuild vs torch 2.7.1 + the **9c44d34** GIN-v13 plugin + gdrdrv-2.4 init-gate
relax + the MNNVL-x86 NVML-crash short-circuit.

## 3. Cross-pod SSH (TRT-LLM is MPI-based; mpirun needs ssh between pods)

```bash
for n in 0 1; do kubectl -n $NS exec ${APP}-$n -- bash /opt/trtllm-repro/scripts/06-sshd-setup-for-mpi.sh; done

# generate a keypair on pod-0, push the pubkey to both + the privkey to pod-1, write an ssh config (port 2222)
PUB=$(kubectl -n $NS exec ${APP}-0 -- bash -c 'mkdir -p /root/.ssh && chmod 700 /root/.ssh
  [ -f /root/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -f /root/.ssh/id_ed25519 -N "" -q; cat /root/.ssh/id_ed25519.pub')
PRIV=$(kubectl -n $NS exec ${APP}-0 -- bash -c 'base64 -w0 /root/.ssh/id_ed25519')
for n in 0 1; do
  kubectl -n $NS exec ${APP}-$n -- bash -c "mkdir -p /root/.ssh; echo '$PUB' > /root/.ssh/authorized_keys
    printf 'Host *\n  Port 2222\n  StrictHostKeyChecking no\n  UserKnownHostsFile /dev/null\n' > /root/.ssh/config
    chmod 600 /root/.ssh/authorized_keys /root/.ssh/config"
done
kubectl -n $NS exec ${APP}-1 -- bash -c "echo '$PRIV' | base64 -d > /root/.ssh/id_ed25519; chmod 600 /root/.ssh/id_ed25519"

# verify pod0 -> pod1 (should print trtllm-efa-bench-1)
P1=$(kubectl -n $NS get pod ${APP}-1 -o jsonpath='{.status.podIP}')
kubectl -n $NS exec ${APP}-0 -- ssh -p 2222 -o BatchMode=yes -o StrictHostKeyChecking=no root@$P1 hostname
```

## 4. Serve an arm (EP16 mpirun) — repeat for ARM=deepep and ARM=dense

```bash
P0=$(kubectl -n $NS get pod ${APP}-0 -o jsonpath='{.status.podIP}')
P1=$(kubectl -n $NS get pod ${APP}-1 -o jsonpath='{.status.podIP}')

# --- DeepEP arm ---
kubectl -n $NS exec ${APP}-0 -- bash -c \
  "P0=$P0 P1=$P1 ARM=deepep nohup bash /opt/trtllm-repro/scripts/08-serve-ab-deepep-vs-dense.sh >/tmp/serve.log 2>&1 & echo serving"

# wait for readiness (~8-10 min: model load + warmup). Confirm health + the SELECTED backend:
kubectl -n $NS exec ${APP}-0 -- bash -c 'curl -s -o /dev/null -w "health=%{http_code}\n" http://127.0.0.1:8000/health'
kubectl -n $NS exec ${APP}-0 -- bash -c "grep -oE 'AlltoallMethodType.[A-Za-z]+' /tmp/trt_serve_deepep.log | tail -1"
# Expect: health=200 ... AlltoallMethodType.DeepEP

# coherence smoke
MODEL=$(kubectl -n $NS exec ${APP}-0 -- bash -c "curl -s http://127.0.0.1:8000/v1/models | python3 -c 'import sys,json;print(json.load(sys.stdin)[\"data\"][0][\"id\"])'")
kubectl -n $NS exec ${APP}-0 -- bash -c "curl -s http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{\"model\":\"$MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":24,\"temperature\":0}'"
```

## 5. AIPerf the running arm (isolated install — does NOT touch the serve env)

```bash
# install aiperf once into an isolated --target dir (per the never-pollute-serve-env rule)
kubectl -n $NS exec ${APP}-0 -- bash -c \
  'pip install --no-cache-dir --target=/opt/aiperf-pkgs --ignore-installed blinker "aiperf==0.10.0" >/tmp/aiperf_install.log 2>&1 && echo installed'

# run AIPerf (conc=4, 40 req, ISL120/OSL128, ignore_eos) — same for both arms
MODELDIR=/data/hf-cache/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/*
kubectl -n $NS exec ${APP}-0 -- bash -c "export PYTHONPATH=/opt/aiperf-pkgs
  M=\$(curl -s http://127.0.0.1:8000/v1/models | python3 -c 'import sys,json;print(json.load(sys.stdin)[\"data\"][0][\"id\"])')
  TD=\$(ls -d $MODELDIR | head -1)
  rm -rf /tmp/aiperf-out; mkdir -p /tmp/aiperf-out
  /opt/aiperf-pkgs/bin/aiperf profile --model \$M --tokenizer \$TD --url http://127.0.0.1:8000 \
    --endpoint-type completions --streaming --concurrency 4 --request-count 40 \
    --synthetic-input-tokens-mean 120 --output-tokens-mean 128 --extra-inputs ignore_eos:true \
    --artifact-dir /tmp/aiperf-out"

# read the numbers
kubectl -n $NS exec ${APP}-0 -- bash -c "PYTHONPATH=/opt/aiperf-pkgs python3 -c \"
import json,glob; d=json.load(open(glob.glob('/tmp/aiperf-out/**/profile_export_aiperf.json',recursive=True)[0]))
print('reqs', d['request_count']['avg'], 'agg_tok/s', round(d['output_token_throughput']['avg'],2),
      'per_user', round(d['output_token_throughput_per_user']['avg'],2),
      'ttft_p50_ms', round(d['time_to_first_token']['p50'],1))\""
```

### Then the DENSE arm
Reap the DeepEP serve's GPUs by explicit PID, then repeat steps 4–5 with `ARM=dense`:

```bash
for n in 0 1; do
  kubectl -n $NS exec ${APP}-$n -- bash -c \
    'for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 "$pid" 2>/dev/null; done; sleep 3
     echo gpu_pids_remaining=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)'
done
# step 4 with ARM=dense -> confirm "AlltoallMethodType.NotEnabled" -> step 5 AIPerf -> compare.
```

## 6. Teardown (cluster hygiene — REQUIRED)

```bash
kubectl -n $NS scale statefulset $APP --replicas=0
kubectl -n $NS get pods -l app=$APP        # confirm 0 pods => nodes released
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: cannot import name 'cuda' from 'cuda'` (all ranks) | cuda-python 13.x (needs 12.x for `from cuda import cuda`) | `pip install cuda-python==12.9.0` (the bringup script pins this) |
| serve worker crash: `NVMLError_InvalidArgument` in `supports_mnnvl` | TRT MNNVL NVML probe crashes on non-MNNVL H200 | the `MNNVL_X86_SHORTCIRCUIT` patch (bringup step 6) |
| `8/16 clients` TCPStore timeout | `MASTER_ADDR` was a non-routable hostname | pass `P0=<pod0 routable IP>` (step 4) |
| `GIN/Plugin: Failed to initialize` / `props.ginType==NONE` | base plugin GIN ABI too old, or gdrdrv-2.4 init gate | bringup builds 9c44d34 (v13) + relaxes the gate |
| `bindings.so: undefined symbol c10::SymInt::sym_ne` | a dep churn bumped torch off 0.21's ABI | `scripts/07-recover-coherent-021-deps.sh` (restore the coherent set; reference the untouched sibling pod) |
| ElasticBuffer.combine 719 | gdrdrv kernel 2.4 + a non-9c44d34 plugin | ensure `NCCL_NET_PLUGIN=/opt/aws-ofi-9c44/src/.libs/libnccl-net-ofi.so` |

#!/bin/bash
# Serve companion to 09-bringup-venv-on-live-cu13-pod.sh: TRT-LLM EP16 serve
# from the /tmp/venv-trt bring-up, DeepEP (shim) or dense arm. This is the
# exact shape of the 2026-07-07 AIPerf 3-cell PASS (66.5/147.5/586.7 agg
# tok/s, 0 err) — the venv adaptation of 08-serve-ab-deepep-vs-dense.sh.
#
# Run ON pod-0 with P0/P1 exported (pod IPs are ephemeral):
#   P0=<pod0-ip> P1=<pod1-ip> ARM=deepep nohup bash 10-serve-venv-deepep-arm.sh &
# Prereqs on BOTH pods: 09-bringup done, sshd:2222 up with pubkeys exchanged
# BOTH ways, api-shim at /tmp/api-shim, YAML at /tmp/extra_llm_api.yaml, model
# snapshot under $HF_HOME (default /tmp/hf), the gdrdrv-2.4-safe GIN plugin
# built (PLUGIN env below).
#
# Server is up when the log shows "Uvicorn running". Backend assertions to
# demand in /tmp/trt_serve_deepep.log before trusting any number:
#   CutlassFusedMoE selects alltoall_method_type <AlltoallMethodType.DeepEP: 2>
#   [trtllm-comm-bridge] lazy-initializing ... x16
#   [shim5c combine] ... x16
#   NET/OFI Selected provider is efa, fabric is efa-direct   x16 (NCCL_DEBUG=INFO)
# AIPerf gate (bench/aiperf/aiperf_sweep.sh): pass TOKENIZER=<snapshot dir> —
# the served model id is the snapshot hash, not a HF repo id (WALL 5).
ARM="${ARM:-deepep}"
exec >/tmp/trt_serve_${ARM}.log 2>&1
set -x
: "${P0:?set P0 to rank-0 pod IP}"
: "${P1:?set P1 to worker pod IP}"
VBIN=/tmp/venv-trt/bin
SP=/tmp/venv-trt/lib/python3.10/site-packages
NVDIR=/tmp/nvshmem-cu12/nvidia/nvshmem
N2304=$(dirname $(find /tmp/nccl-2304 -path "*nccl/lib" -type d | head -1))/lib
PLUGIN=${PLUGIN:?abs path to gdrdrv-safe libnccl-net-ofi.so (9c44d34 + 2 patches on 2.4 kernels)}
[ -f "$PLUGIN" ] || { echo "FATAL no plugin"; exit 3; }
HF=${HF_HOME:-/tmp/hf}
MODEL=$(ls -d $HF/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/*/ | head -1)
[ -n "$MODEL" ] || { echo "FATAL no model under $HF"; exit 5; }
CUSPARSELT=$SP/nvidia/cusparselt/lib
LD="$N2304:$NVDIR/lib:/tmp/cuda12/lib64:$CUSPARSELT:/usr/local/lib:/opt/amazon/efa/lib"
cat > /tmp/hostfile <<HF2
$P0 slots=8
$P1 slots=8
HF2
if [ "$ARM" = "dense" ]; then
  ARM_ENV="-x TRTLLM_MOE_DISABLE_ALLTOALLV=1 -x TRTLLM_CAN_USE_DEEP_EP=0"
else
  ARM_ENV="-x TRTLLM_MOE_DISABLE_ALLTOALLV=0 -x TRTLLM_CAN_USE_DEEP_EP=1"
fi
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export PATH=/opt/amazon/openmpi/bin:$PATH
mpirun --allow-run-as-root -np 16 --hostfile /tmp/hostfile \
  --prefix /opt/amazon/openmpi \
  --mca plm_rsh_agent "ssh -p 2222 -o BatchMode=yes -o StrictHostKeyChecking=no" \
  --mca btl_tcp_if_include eth0 --mca oob_tcp_if_include eth0 \
  -x PATH=$VBIN:/opt/amazon/openmpi/bin:/usr/local/nvidia/bin:/usr/bin:/bin \
  -x MASTER_ADDR=$P0 -x MASTER_PORT=29555 -x TRTLLM_SHIM_PG_MASTER_PORT=29555 \
  -x DEEP_EP_USE_V2_SHIM=1 -x PYTHONPATH=/tmp/api-shim -x DEEP_EP_BACKEND=nccl \
  $ARM_ENV \
  -x NCCL_NET_PLUGIN=$PLUGIN \
  -x NCCL_GIN_TYPE=2 -x OFI_NCCL_GIN_GDAKI=0 -x NCCL_GIN_ENABLE=1 -x NCCL_NVLS_ENABLE=0 -x NCCL_CUMEM_ENABLE=1 \
  -x OFI_NCCL_PROTOCOL=RDMA -x OFI_NCCL_GIN_MAX_REQUESTS=512 -x EP_EFA_MAX_QPS=2 -x EP_EFA_RDMA_GBS=25.0 \
  -x FI_PROVIDER=efa -x FI_EFA_USE_DEVICE_RDMA=1 -x FI_EFA_FORK_SAFE=1 -x FI_EFA_ENABLE_SHM_TRANSFER=0 \
  -x GLOO_SOCKET_IFNAME=eth0 -x NCCL_SOCKET_IFNAME=eth0 \
  -x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=INIT,NET \
  -x HF_HOME=$HF -x HF_HUB_OFFLINE=1 -x TRANSFORMERS_OFFLINE=1 \
  -x LD_LIBRARY_PATH=$LD \
  $VBIN/trtllm-llmapi-launch $VBIN/trtllm-serve serve $MODEL \
    --backend pytorch --tp_size 16 --ep_size 16 --host 0.0.0.0 --port 8000 \
    --max_batch_size 16 --max_num_tokens 8192 --kv_cache_free_gpu_memory_fraction 0.7 \
    --trust_remote_code --extra_llm_api_options /tmp/extra_llm_api.yaml &
echo "MPIRUN_PID=$! ARM=$ARM"
for i in $(seq 1 120); do
  sleep 5
  grep -qiE "Uvicorn running|Executor worker returned error|Engine loading failed" /tmp/trt_serve_${ARM}.log 2>/dev/null && break
done
echo "WATCH_DONE ARM=$ARM"
grep -iE "AlltoallMethodType|Uvicorn running|returned error" /tmp/trt_serve_${ARM}.log | head -5

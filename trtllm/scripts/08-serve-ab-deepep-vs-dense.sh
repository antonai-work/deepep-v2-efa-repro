#!/bin/bash
# A/B-parameterized TRT-LLM EP16 serve. ARM=deepep (TRTLLM_CAN_USE_DEEP_EP=1) or ARM=dense (TRTLLM_MOE_DISABLE_ALLTOALLV=1).
ARM="${ARM:-deepep}"
exec >/tmp/trt_serve_${ARM}.log 2>&1
set -x
# Pod IPs are passed in via the environment (they are ephemeral — change on every pod recreate).
# On the cluster, derive them and run this script ON pod-0 with P0/P1 exported:
#   P0=$(kubectl -n NS get pod trtllm-efa-bench-0 -o jsonpath='{.status.podIP}')   # rank-0 (this pod)
#   P1=$(kubectl -n NS get pod trtllm-efa-bench-1 -o jsonpath='{.status.podIP}')   # worker pod
# See INSTRUCTIONS.md step 4.
: "${P0:?set P0 to rank-0 pod IP (kubectl get pod ...-0 -o jsonpath={.status.podIP})}"
: "${P1:?set P1 to worker pod IP (kubectl get pod ...-1 -o jsonpath={.status.podIP})}"
NVDIR=/opt/nvshmem-cu12/nvidia/nvshmem
N2304=$(dirname $(find /opt/nccl-2304 -path '*nccl/lib' -type d|head -1))/lib
PLUGIN=/opt/aws-ofi-9c44/src/.libs/libnccl-net-ofi.so
MODEL=/data/hf-cache/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782
LD="$N2304:$NVDIR/lib:/usr/local/cuda-12.9/targets/x86_64-linux/lib:/usr/local/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusparselt/lib:/opt/amazon/efa/lib"
cat > /tmp/hostfile <<HF
$P0 slots=8
$P1 slots=8
HF
# arm-specific MoE selection
if [ "$ARM" = "dense" ]; then
  ARM_ENV="-x TRTLLM_MOE_DISABLE_ALLTOALLV=1 -x TRTLLM_CAN_USE_DEEP_EP=0"
else
  ARM_ENV="-x TRTLLM_MOE_DISABLE_ALLTOALLV=0 -x TRTLLM_CAN_USE_DEEP_EP=1"
fi
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
mpirun --allow-run-as-root -np 16 --hostfile /tmp/hostfile \
  --mca plm_rsh_agent "ssh -p 2222 -o BatchMode=yes -o StrictHostKeyChecking=no" \
  --mca btl_tcp_if_include eth0 --mca oob_tcp_if_include eth0 \
  -x MASTER_ADDR=$P0 -x MASTER_PORT=29555 -x TRTLLM_SHIM_PG_MASTER_PORT=29555 \
  -x DEEP_EP_USE_V2_SHIM=1 -x PYTHONPATH=/opt/api-shim -x DEEP_EP_BACKEND=nccl \
  $ARM_ENV \
  -x NCCL_NET_PLUGIN=$PLUGIN \
  -x NCCL_GIN_TYPE=2 -x OFI_NCCL_GIN_GDAKI=0 -x NCCL_GIN_ENABLE=1 -x NCCL_NVLS_ENABLE=0 -x NCCL_CUMEM_ENABLE=1 \
  -x OFI_NCCL_PROTOCOL=RDMA -x OFI_NCCL_GIN_MAX_REQUESTS=512 -x EP_EFA_MAX_QPS=2 -x EP_EFA_RDMA_GBS=25.0 \
  -x FI_PROVIDER=efa -x FI_EFA_USE_DEVICE_RDMA=1 -x FI_EFA_FORK_SAFE=1 -x FI_EFA_ENABLE_SHM_TRANSFER=0 \
  -x HF_HOME=/data/hf-cache -x LD_LIBRARY_PATH=$LD \
  trtllm-llmapi-launch trtllm-serve serve $MODEL \
    --backend pytorch --tp_size 16 --ep_size 16 --host 0.0.0.0 --port 8000 \
    --max_batch_size 16 --max_num_tokens 8192 --kv_cache_free_gpu_memory_fraction 0.7 \
    --trust_remote_code --extra_llm_api_options /opt/extra_llm_api.yaml &
echo "MPIRUN_PID=$! ARM=$ARM"
for i in $(seq 1 90); do sleep 5; grep -qiE "Uvicorn running|Executor worker returned error|AlltoallMethodType" /tmp/trt_serve_${ARM}.log 2>/dev/null && break; done
echo "WATCH_DONE ARM=$ARM"

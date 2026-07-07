#!/bin/bash
# Gate-3 serve: the FAITHFUL campaign launcher shape (efa-gda repro-hub
# assets/uccl/serve_sglang.sh lineage, 2026-07-02 gdrdrv-2.4 LIVE-VERIFIED).
# This is the exact shape of the
# 2026-07-07 AIPerf PASS (c4/c8/c32 = 34.5/67.3/255.2, 0 errors).
# LAUNCHER SHAPE MATTERS: a slimmed variant faults combine CUDA 719 even with
# the correct 9c44 plugin. Do not drop flags without re-gating.
# Usage: MASTER=<node0-ip> NCCL_NET_PLUGIN=<abs path> bash serve_sglang_v2.sh <0|1>
NR=${1:?node_rank 0|1}
export PYTHONUNBUFFERED=1
MASTER=${MASTER:?node0 ip}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-30B-A3B-FP8}

export NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:?abs path to gdrdrv-2.4-safe libnccl-net-ofi.so}
PLUGIN_LIBDIR="$(dirname "$NCCL_NET_PLUGIN")"
[ -f "$NCCL_NET_PLUGIN" ] || { echo "FATAL no plugin"; exit 3; }

# GIN v2 CPU-proxy fences (NEVER GDAKI/IBGDA) — REPRODUCE.md common env
export OFI_NCCL_GIN_GDAKI=0 NCCL_GIN_TYPE=2 NCCL_GIN_ENABLE=1 NCCL_CUMEM_ENABLE=1 NCCL_NVLS_ENABLE=0
export FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 OFI_NCCL_PROTOCOL=RDMA EP_EFA_MAX_QPS=2
export EP_NUM_COMMS=${EP_NUM_COMMS:-8}   # >1 requires the multicomm overlay patch
export GLOO_SOCKET_IFNAME=eth0 NCCL_SOCKET_IFNAME=eth0
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} SGLANG_ENABLE_JIT_DEEPGEMM=1 SG_DEEPGEMM_JIT=1
NVSHMEM_LIB="$(python3 -c 'import nvidia.nvshmem,os;print(os.path.join(os.path.dirname(nvidia.nvshmem.__file__),"lib"))' 2>/dev/null || true)"
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:${PLUGIN_LIBDIR}:${CUDA_HOME}/lib64:/usr/local/lib:${NVSHMEM_LIB:-}:${LD_LIBRARY_PATH:-}
export SGLANG_DEEPEP_USE_V2=1
# MANDATORY under AIPerf prefill load: default V2 buffer = 128 tokens/rank;
# conc32 prefill bursts trip buffer.hpp:686 on every rank. sglang hard-caps
# this knob at 1024 (assert <= 1024).
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${SGL_MAXTOK:-1024}
unset NVSHMEM_IB_ENABLE_IBGDA || true

DEEPEP_CFG=${DEEPEP_CFG:-/tmp/deepep_config.json}
cat > "$DEEPEP_CFG" <<'EOF'
{
  "normal_dispatch": {"num_sms": 24, "num_max_nvl_chunked_send_tokens": 16, "num_max_nvl_chunked_recv_tokens": 512, "num_max_rdma_chunked_send_tokens": 16, "num_max_rdma_chunked_recv_tokens": 512},
  "normal_combine":  {"num_sms": 24, "num_max_nvl_chunked_send_tokens": 16, "num_max_nvl_chunked_recv_tokens": 512, "num_max_rdma_chunked_send_tokens": 16, "num_max_rdma_chunked_recv_tokens": 512}
}
EOF
mkdir -p /tmp/run
rm -f /tmp/run/sgl-node$NR.log
nohup python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" --trust-remote-code \
  --tp-size 16 --ep-size 16 --dp-size 16 \
  --nnodes 2 --node-rank $NR --dist-init-addr $MASTER:${DIST_PORT:-29800} \
  --mem-fraction-static 0.85 --moe-dense-tp-size 1 \
  --chunked-prefill-size 32768 --cuda-graph-bs 256 --page-size 256 \
  --attention-backend fa3 \
  --ep-num-redundant-experts 0 --ep-dispatch-algorithm dynamic \
  --enable-dp-attention --enable-dp-lm-head \
  --moe-a2a-backend deepep --deepep-mode normal --deepep-config "$DEEPEP_CFG" \
  --host 0.0.0.0 --port 30000 \
  > /tmp/run/sgl-node$NR.log 2>&1 &
echo "SGL_LAUNCHED node=$NR pid=$! plugin=$NCCL_NET_PLUGIN"

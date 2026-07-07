#!/bin/bash
# vLLM EP16 (TP1xDP16) DeepEP-V2 over EFA — leader/worker A/B launcher.
# Consolidated into deepep-v2-integration from efa-gda
#   deliverables/efa-dropin-inference/launch/serve_ep16_AB.sh, SANITIZED +
#   re-pointed to the CANONICAL NGC-built plugin path (public-root, CLAUDE.md
#   rule 7): env-specific baked values (private MASTER IP, /opt/aws-ofi-nccl-
#   gdaki-g3n2 build path, /work/hf) are now REQUIRED/overridable env vars.
#
# Hard fences (VLLM-V2 continuation brief): GIN v2 CPU-proxy + ElasticBuffer
# ONLY. NEVER GDAKI (OFI_NCCL_GIN_GDAKI stays 0), NEVER IBGDA. Wrap every serve
# with bench/vllm/assert_backend.py so a silent legacy-Buffer/NVSHMEM fallback
# fails loud.
#
# Required env:
#   ROLE      leader | worker
#   MASTER    node-0 EFA IP (data-parallel address). NO default (was a baked
#             10.240.* literal — do not hardcode; pass your cluster's node-0 EFA IP).
# Optional env:
#   EPNC      EP_NUM_COMMS (default 8; the multi-comm 2.09x win. Set 1 for the A/B baseline arm).
#   BACKEND   deepep_v2 (default) | allgather_reducescatter (non-DeepEP control arm).
#   NCCL_NET_PLUGIN  default /opt/aws-ofi-nccl/lib/libnccl-net-ofi.so (canonical NGC-built path).
#   HF_HOME   default /work/hf.  MODEL default Qwen/Qwen3-30B-A3B-FP8.
set -u
: "${ROLE:?set ROLE=leader|worker}"
: "${MASTER:?set MASTER=<node-0 EFA IP> (no baked default)}"
export HF_HOME="${HF_HOME:-/work/hf}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"

# Canonical NGC-built plugin (public-root). Override only if your image installs elsewhere.
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-/opt/aws-ofi-nccl/lib/libnccl-net-ofi.so}"
[ -f "$NCCL_NET_PLUGIN" ] || { echo "FATAL: NCCL_NET_PLUGIN not found: $NCCL_NET_PLUGIN"; exit 3; }

# --- GIN v2 CPU-proxy transport (the ONLY EFA-viable DeepEP path) ---
export NCCL_GIN_TYPE=2 NCCL_GIN_ENABLE=1 NCCL_CUMEM_ENABLE=1 NCCL_CUMEM_HOST_ENABLE=1
export OFI_NCCL_GIN_GDAKI=0        # proxy GIN, NOT GDAKI (type-4 reject). NEVER flip to 1 in this lane.
export NCCL_NVLS_ENABLE=0 NCCL_IGNORE_DISABLED_P2P=1
export FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 FI_EFA_ENABLE_SHM_TRANSFER=0
export OFI_NCCL_PROTOCOL=RDMA OFI_NCCL_GIN_MAX_REQUESTS=512
export EP_EFA_MAX_QPS=2 EP_EFA_RDMA_GBS=25.0
export VLLM_USE_FLASHINFER_MOE_FP8=0
export VLLM_USE_FLASHINFER_SAMPLER=0   # flashinfer not installed; default-on import crashes sampler pre-fallback
export EP_NUM_COMMS=${EPNC:-8}
export VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE=1

BACKEND=${BACKEND:-deepep_v2}          # A/B: deepep_v2 | allgather_reducescatter
# The non-DeepEP allgather arm triggers a ~6h DeepGEMM autotune (no cache); force
# triton MoE so it serves in ~100s. The deepep_v2 arms never trigger DeepGEMM.
if [ "$BACKEND" = "allgather_reducescatter" ]; then export VLLM_USE_DEEP_GEMM=${USE_DG:-0}; else export VLLM_USE_DEEP_GEMM=${USE_DG:-1}; fi

export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,ENV
COMMON="--model $MODEL --enable-expert-parallel --tensor-parallel-size 1 \
  --data-parallel-size 16 --data-parallel-size-local 8 --all2all-backend $BACKEND \
  --trust-remote-code --enforce-eager --data-parallel-address $MASTER --data-parallel-rpc-port 29500"
mkdir -p /work/run
if [ "$ROLE" = "leader" ]; then
  nohup vllm serve $COMMON --port 8000 --data-parallel-start-rank 0 > /work/run/vllm-leader.log 2>&1 &
  echo "LEADER_PID $! BACKEND=$BACKEND EP_NUM_COMMS=$EP_NUM_COMMS NCCL_NET_PLUGIN=$NCCL_NET_PLUGIN"
else
  nohup vllm serve $COMMON --headless --data-parallel-start-rank 8 > /work/run/vllm-worker.log 2>&1 &
  echo "WORKER_PID $! BACKEND=$BACKEND EP_NUM_COMMS=$EP_NUM_COMMS NCCL_NET_PLUGIN=$NCCL_NET_PLUGIN"
fi

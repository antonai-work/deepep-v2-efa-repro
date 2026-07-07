#!/bin/bash
exec >/tmp/smoke10.log 2>&1
set -x
rm -rf /tmp/smklog10; mkdir -p /tmp/smklog10
NVDIR="/opt/nvshmem-cu12/nvidia/nvshmem"
N2304="/opt/nccl-2304/nvidia/nccl/lib"
export LD_LIBRARY_PATH="${N2304}:${NVDIR}/lib:/usr/local/cuda-12.9/targets/x86_64-linux/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/opt/api-shim:${PYTHONPATH:-}"
export DEEP_EP_USE_V2_SHIM=1
export NCCL_NET_PLUGIN=/opt/aws-ofi-9c44/src/.libs/libnccl-net-ofi.so
export NCCL_GIN_TYPE=2 OFI_NCCL_GIN_GDAKI=0 NCCL_GIN_ENABLE=1 NCCL_NVLS_ENABLE=0 NCCL_CUMEM_ENABLE=1
export OFI_NCCL_PROTOCOL=RDMA EP_EFA_MAX_QPS=2
export FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 FI_EFA_FORK_SAFE=1
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export NCCL_DEBUG=WARN
cd /opt/api-shim
timeout 180 torchrun --nproc-per-node=8 --master-addr=127.0.0.1 --master-port=29509 \
  --redirects 3 --log-dir /tmp/smklog10 /opt/api-shim/smoke_test_shim.py
echo "SMOKE10_RC=$?"
F=$(find /tmp/smklog10 -path "*/0/stdout.log"|head -1); FE=$(find /tmp/smklog10 -path "*/0/stderr.log"|head -1)
echo "=== rank0 stdout (full result) ==="; cat "$F" 2>/dev/null | grep -ivE "Channel|NCCL INFO.*Conn|P2P|via SHM|GPU Direct" | tail -45
echo "=== rank0 stderr ==="; tail -15 "$FE" 2>/dev/null
echo "SMOKE10_DONE"

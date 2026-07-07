#!/bin/bash
# Recover a TRT-LLM 0.21 pod whose dep set was corrupted by a version-juggle / aiperf install.
# Restores the coherent working set (reference: the untouched sibling pod). 2026-06-22.
# Symptoms fixed: bindings.so 'undefined symbol c10::SymInt::sym_ne' (torch mismatch),
#   deep_ep._C 'undefined ncclTeamWorld' (cu12 nccl re-bundled), transformers Conv1D /
#   huggingface_hub is_offline_mode (transformers 5.x churn), inductor 'triton_key' (triton 3.5).
set -x
pip install --no-cache-dir --break-system-packages --force-reinstall --no-deps --extra-index-url https://pypi.nvidia.com "torch==2.7.1"
pip install --no-cache-dir --break-system-packages "nvidia-cusparselt-cu12"
SO=$(find /usr/local/lib/python3.12/dist-packages/nvidia -name "libcusparseLt.so*" 2>/dev/null|head -1); echo "$(dirname $SO)" > /etc/ld.so.conf.d/99-cusparselt.conf
pip install --no-cache-dir --break-system-packages --no-deps --force-reinstall "nvidia-nccl-cu13>=2.30.4"
NCCL13=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib; ln -sf "$(basename $(ls $NCCL13/libnccl.so.2*|head -1))" "$NCCL13/libnccl.so"; ldconfig
pip install --no-cache-dir --break-system-packages --force-reinstall --no-deps "transformers==4.51.3" "huggingface-hub==0.36.2" "tokenizers==0.21.4" "triton==3.3.1"
# verify (LD path = the cu13-nccl + nvshmem-cu12 + cuda12.9 + cusparselt)
NVDIR=/opt/nvshmem-cu12/nvidia/nvshmem
export LD_LIBRARY_PATH="$NCCL13:$NVDIR/lib:/usr/local/cuda-12.9/targets/x86_64-linux/lib:/usr/local/lib:$(dirname $SO)"
python3 -c "import tensorrt_llm, deep_ep, triton, transformers; print('ALL_IMPORT_OK', tensorrt_llm.__version__)"

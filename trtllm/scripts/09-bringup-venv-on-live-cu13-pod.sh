#!/bin/bash
# ALTERNATIVE bring-up: TRT-LLM 0.21 + api-shim on a LIVE cu13 serving pod,
# fully isolated in a venv — zero mutation of the pod's cu13 serving stack
# (torch 2.11+cu130, /opt/DeepEP, 9c44 plugin all untouched).
#
# This is the exact chain that produced the 2026-07-07 AIPerf 3-cell PASS
# (66.5/147.5/586.7 agg tok/s, 0 err) on 2x p5en H200 bench pods. Use it when
# the pods already serve vLLM/SGLang gates and you want the TRT lane WITHOUT
# rebuilding the image (the canonical from-scratch path stays ../Dockerfile +
# 00pre-substrate-from-ngc.sh + 00-bringup-all-consolidated.sh).
#
# Everything installs from PUBLIC roots: pypi.nvidia.com (tensorrt_llm),
# PyPI (torch 2.7.1 pulled as trtllm dep, nvidia-* wheels), and the official
# NVIDIA redist server (real nvcc — see WALL 2). Run as root inside the pod,
# detached: nohup bash 09-bringup-venv-on-live-cu13-pod.sh >/tmp/t5.log 2>&1 &
# Sentinel: /tmp/t5_done. ~35-50 min (deep_ep _C nvcc build dominates).
#
# The five WALLS this script encodes (each cost a debug loop, 2026-07-07):
#  1. cuda-python: trtllm's resolver pulls 13.x which drops the
#     `from cuda import cuda` layout -> ImportError at import. Pin 12.9.0
#     WITH cuda-bindings==12.9.0, AFTER the trtllm install.
#  2. pip nvidia-cuda-nvcc-cu12 ships ONLY ptxas — it is not a compiler.
#     Fetch the real cuda_nvcc 12.9.86 archive from the official redist.
#  3. deep_ep's find_nccl_root() finds the venv nccl first, and trtllm dep-pins
#     nvidia-nccl-cu12==2.26.2 — but the V2 backend needs 2.30.x headers
#     (nccl_device/core.h) AND torch must resolve ncclTeamWorld at import.
#     Fix = EP_NCCL_ROOT_DIR override + force-reinstall venv nccl-cu13 2.30.4.
#  4. torch's CUDAContextLight.h includes cusparse.h/cublas_v2.h/cusolverDn.h;
#     the synthesized CUDA_HOME needs those headers + UNVERSIONED .so linker
#     names (pip wheels ship only libfoo.so.12).
#  5. (serve-time, see 10-serve...) AIPerf needs TOKENIZER=<snapshot dir> when
#     the model is served by path — the served id is a hash, not a HF repo.
set -x
C="--no-cache-dir"
NV="--extra-index-url https://pypi.nvidia.com"
rm -f /tmp/t5_done

echo "=== STEP 0: isolated venv (python3.10, NO system site) ==="
/usr/bin/python3 -m pip install --user -q virtualenv    # pods lack ensurepip
rm -rf /tmp/venv-trt
/usr/bin/python3 -m virtualenv -q /tmp/venv-trt
V=/tmp/venv-trt/bin/pip
P=/tmp/venv-trt/bin/python
SP=/tmp/venv-trt/lib/python3.10/site-packages

echo "=== STEP 1: tensorrt 10.11 prewheels + tensorrt_llm 0.21.0 + mpi4py ==="
timeout 1200 $V install $C $NV --prefer-binary "tensorrt-cu12-bindings==10.11.0.33" "tensorrt-cu12-libs==10.11.0.33" "tensorrt-cu12==10.11.0.33" "tensorrt==10.11.0"
timeout 2400 $V install $C $NV --prefer-binary "tensorrt_llm==0.21.0"
export PATH=/opt/amazon/openmpi/bin:$PATH
timeout 600 $V install $C mpi4py || timeout 600 env MPICC=/opt/amazon/openmpi/bin/mpicc $V install $C mpi4py
# WALL 1: pin cuda-python AFTER trtllm (its resolver pulled 13.x)
$V install $C --force-reinstall "cuda-python==12.9.0" "cuda-bindings==12.9.0"
echo "TRT_INSTALLED=$($P -c "import tensorrt_llm;print(tensorrt_llm.__version__)" 2>&1 | tail -1)"

echo "=== STEP 2: dual-NCCL 2.30.4 + cu12 nvshmem + pip CUDA-12 toolchain ==="
timeout 600 $V install $C --no-deps --target=/tmp/nccl-2304 "nvidia-nccl-cu13==2.30.4"
N2304=$(dirname $(find /tmp/nccl-2304 -path "*nccl/lib" -type d | head -1))/lib
ln -sf "$(basename $(ls $N2304/libnccl.so.2* | head -1))" "$N2304/libnccl.so"
# WALL 3b: the VENV nccl must also be 2.30.4 (torch RPATH resolves ncclTeamWorld there)
$V install $C --no-deps --force-reinstall "nvidia-nccl-cu13==2.30.4"
VNCCL=$SP/nvidia/nccl/lib
[ -e $VNCCL/libnccl.so ] || ln -sf "$(basename $(ls $VNCCL/libnccl.so.2* | head -1))" "$VNCCL/libnccl.so"
timeout 600 $V install $C --no-deps --target=/tmp/nvshmem-cu12 "nvidia-nvshmem-cu12==3.6.5"
NVDIR=/tmp/nvshmem-cu12/nvidia/nvshmem
ln -sf "$(basename $(ls $NVDIR/lib/libnvshmem_host.so.* | head -1))" "$NVDIR/lib/libnvshmem_host.so"
timeout 900 $V install $C "nvidia-cuda-nvcc-cu12==12.9.86" "nvidia-cuda-runtime-cu12==12.9.79" "nvidia-cuda-cccl-cu12" "nvidia-cuda-nvrtc-cu12==12.9.86"
# WALL 4 headers: torch includes pull the math-lib headers
timeout 600 $V install $C --no-deps "nvidia-cusparse-cu12" "nvidia-cublas-cu12" "nvidia-cusolver-cu12" "nvidia-curand-cu12"

echo "=== STEP 3: synthesize CUDA_HOME=/tmp/cuda12 ==="
rm -rf /tmp/cuda12; mkdir -p /tmp/cuda12/bin /tmp/cuda12/include /tmp/cuda12/lib64/stubs
# WALL 2: real nvcc from the official NVIDIA redist (pip wheel = ptxas only)
cd /tmp
curl -fsSLO https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.9.86-archive.tar.xz
rm -rf /tmp/nvcc-redist && mkdir -p /tmp/nvcc-redist
tar -xJf cuda_nvcc-linux-x86_64-12.9.86-archive.tar.xz -C /tmp/nvcc-redist --strip-components=1
cp -af /tmp/nvcc-redist/bin/. /tmp/cuda12/bin/
cp -af /tmp/nvcc-redist/nvvm /tmp/cuda12/
cp -rn /tmp/nvcc-redist/include/. /tmp/cuda12/include/ || true
# headers: runtime + cccl + nvrtc + the math libs (WALL 4)
for d in $SP/nvidia/cuda_runtime/include $SP/nvidia/cuda_cccl/include $SP/nvidia/cuda_nvrtc/include \
         $SP/nvidia/cusparse/include $SP/nvidia/cublas/include $SP/nvidia/cusolver/include $SP/nvidia/curand/include; do
  [ -d "$d" ] && cp -rsn $d/* /tmp/cuda12/include/ 2>/dev/null
done
# libs + UNVERSIONED linker names (WALL 4)
for d in $SP/nvidia/cuda_runtime/lib $SP/nvidia/cuda_nvrtc/lib $SP/nvidia/cusparse/lib \
         $SP/nvidia/cublas/lib $SP/nvidia/cusolver/lib $SP/nvidia/curand/lib; do
  [ -d "$d" ] && ln -sf $d/* /tmp/cuda12/lib64/ 2>/dev/null
done
cd /tmp/cuda12/lib64
for so in libcudart libnvrtc libcusparse libcublas libcublasLt libcusolver libcurand; do
  tgt=$(ls ${so}.so.12* 2>/dev/null | grep -v ".alt" | head -1)
  [ -n "$tgt" ] && [ ! -e ${so}.so ] && ln -sf "$tgt" "${so}.so"
done
ln -sf /usr/local/cuda-13/lib64/stubs/libcuda.so /tmp/cuda12/lib64/stubs/libcuda.so
/tmp/cuda12/bin/nvcc --version | tail -2

echo "=== STEP 4: rebuild deep_ep _C in a COPY vs torch 2.7.1 + nccl 2.30.4 + cu12 nvshmem ==="
TORCH_INC=$($P -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),\"include\"))")
SYS_FMT=/usr/local/lib/python3.10/dist-packages/torch/include/fmt   # fmt headers from the pod's torch 2.11
[ -d "$TORCH_INC/fmt" ] || cp -r $SYS_FMT "$TORCH_INC/fmt"
rm -rf /tmp/DeepEP-cu12
cp -r /opt/DeepEP /tmp/DeepEP-cu12                                  # never touch the serving /opt/DeepEP
rm -rf /tmp/DeepEP-cu12/build /tmp/DeepEP-cu12/deep_ep/_C*.so /tmp/DeepEP-cu12/*.egg-info
cd /tmp/DeepEP-cu12
export CUDA_HOME=/tmp/cuda12 NVSHMEM_DIR=$NVDIR EP_NVSHMEM_ROOT_DIR=$NVDIR
export EP_NCCL_ROOT_DIR=$(dirname $N2304)                           # WALL 3a: beat the venv 2.26 in find_nccl_root
export PATH=/tmp/cuda12/bin:$PATH
export LIBRARY_PATH="$NVDIR/lib:$N2304:/tmp/cuda12/lib64:/tmp/cuda12/lib64/stubs"
export LD_LIBRARY_PATH="$N2304:$NVDIR/lib:/tmp/cuda12/lib64"
export TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=32 NVCC_THREADS=2
timeout 2700 $V install --no-build-isolation $C --no-deps -e . 2>&1 | tail -4
LD_LIBRARY_PATH="$N2304:$NVDIR/lib:/tmp/cuda12/lib64" $P -c "import deep_ep; from deep_ep.buffers.elastic import ElasticBuffer; print(\"DEEPEP_OK\", deep_ep.__file__)"
echo "DEEPEP_IMPORT_RC=$?"

echo "=== STEP 5: MNNVL x86 short-circuit patch (venv tensorrt_llm) ==="
$P - <<PY
f="/tmp/venv-trt/lib/python3.10/site-packages/tensorrt_llm/_mnnvl_utils.py"; s=open(f).read()
old="""        arch = platform.machine().lower()
        is_on_aarch64 = "aarch64" in arch
        support_nvlink_and_all_up = MnnvlMemory.support_nvlink(True)
        return is_on_aarch64 and support_nvlink_and_all_up"""
new="""        arch = platform.machine().lower()
        is_on_aarch64 = "aarch64" in arch
        # MNNVL_X86_SHORTCIRCUIT
        if not is_on_aarch64:
            return False
        try:
            support_nvlink_and_all_up = MnnvlMemory.support_nvlink(True)
        except Exception:
            return False
        return is_on_aarch64 and support_nvlink_and_all_up"""
if "MNNVL_X86_SHORTCIRCUIT" not in s and old in s:
    open(f,"w").write(s.replace(old,new)); print("MNNVL_PATCHED")
else:
    print("MNNVL_SKIP", "MNNVL_X86_SHORTCIRCUIT" in s)
PY

echo "=== STEP 6: sshd 2222 for cross-pod mpirun ==="
mkdir -p /run/sshd /root/.ssh && chmod 700 /root/.ssh
grep -qE "^Port 2222" /etc/ssh/sshd_config || sed -i "s/#\?Port .*/Port 2222/" /etc/ssh/sshd_config
sed -i "s/#\?PermitRootLogin.*/PermitRootLogin yes/" /etc/ssh/sshd_config
pgrep -f "sshd -D" >/dev/null || setsid /usr/sbin/sshd -D -p 2222 </dev/null >/tmp/sshd.run.log 2>&1 &
[ -f /root/.ssh/id_ed25519 ] || ssh-keygen -q -t ed25519 -N "" -f /root/.ssh/id_ed25519
grep -qf /root/.ssh/id_ed25519.pub /root/.ssh/authorized_keys 2>/dev/null || cat /root/.ssh/id_ed25519.pub >> /root/.ssh/authorized_keys
# NOTE: exchange pubkeys BETWEEN the two pods before serving (each pod's
# id_ed25519.pub into the other's authorized_keys) — mpirun sshes both ways.

echo "BRINGUP_DONE trt=$($P -c "import tensorrt_llm;print(tensorrt_llm.__version__)" 2>&1 | tail -1)"
echo "rc=0" > /tmp/t5_done

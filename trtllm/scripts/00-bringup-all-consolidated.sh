#!/bin/bash
# Consolidated TRT-LLM 0.21 DeepEP-V2 bring-up on fresh GHCR base. Idempotent-ish. ~30-40 min.
exec >/tmp/bringup_all.log 2>&1
set -x
NV="--extra-index-url https://pypi.nvidia.com"
C="--no-cache-dir --break-system-packages"
echo "=== STEP 0: capture fmt headers (torch 2.11, before downgrade) + python symlink ==="
cp -r /usr/local/lib/python3.12/dist-packages/torch/include/fmt /opt/fmt_headers_from_torch2_11 2>/dev/null
ls /opt/fmt_headers_from_torch2_11/base.h && echo FMT_OK
[ -z "$(which python 2>/dev/null)" ] && ln -sf "$(which python3)" /usr/local/bin/python
echo "=== STEP 1: trtllm 0.21 (prewheel tensorrt-cu12 by name to dodge hash) ==="
pip install $C $NV --prefer-binary "tensorrt-cu12-bindings==10.11.0.33" "tensorrt-cu12-libs==10.11.0.33" "tensorrt-cu12==10.11.0.33" "tensorrt==10.11.0"
pip install $C $NV --prefer-binary "tensorrt_llm==0.21.0" "mpi4py"
# trtllm's resolver pulls cuda-python 13.x, which drops the `from cuda import cuda`
# layout trtllm 0.21 imports (live-hit 2026-07-07). Pin 12.9 WITH its cuda-bindings dep.
pip install $C --force-reinstall "cuda-python==12.9.0" "cuda-bindings==12.9.0"
echo "TRT_INSTALLED=$(python3 -c 'import tensorrt_llm;print(tensorrt_llm.__version__)' 2>&1|tail -1)"
echo "=== STEP 2: dual-NCCL cu13 2.30.4 (isolated) + cu12 nvshmem ==="
pip install $C --no-deps --target=/opt/nccl-2304 "nvidia-nccl-cu13==2.30.4"
N2304=$(dirname $(find /opt/nccl-2304 -path '*nccl/lib' -type d|head -1))/lib
ln -sf "$(basename $(ls $N2304/libnccl.so.2*|head -1))" "$N2304/libnccl.so"
pip install $C --no-deps --target=/opt/nvshmem-cu12 "nvidia-nvshmem-cu12==3.6.5"
NVDIR=/opt/nvshmem-cu12/nvidia/nvshmem
ln -sf "$(basename $(ls $NVDIR/lib/libnvshmem_host.so.*|head -1))" "$NVDIR/lib/libnvshmem_host.so"
# also dist-packages cu13 nccl (for torch RPATH ncclTeamWorld)
pip install $C --no-deps --force-reinstall "nvidia-nccl-cu13>=2.30.4"
NCCL13=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib
ln -sf "$(basename $(ls $NCCL13/libnccl.so.2*|head -1))" "$NCCL13/libnccl.so"
echo "=== STEP 3: cudart 12.9 (cudaMemcpyBatchAsync) ==="
pip uninstall -y $C nvidia-cuda-runtime-cu12 || true
echo "/usr/local/cuda-12.9/targets/x86_64-linux/lib" > /etc/ld.so.conf.d/00-system-cuda.conf
echo "=== STEP 4: fmt into torch 2.7.1 include + rebuild deep_ep _C vs torch 2.7.1 + nccl 2.30.4 + cu12 nvshmem ==="
TORCH_INC=$(python3 -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"include"))')
[ -d "$TORCH_INC/fmt" ] || cp -r /opt/fmt_headers_from_torch2_11 "$TORCH_INC/fmt"
cd /opt/DeepEP
export NVSHMEM_DIR="$NVDIR" EP_NVSHMEM_ROOT_DIR="$NVDIR"
export LIBRARY_PATH="$NVDIR/lib:$N2304:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$N2304:$NVDIR/lib:/usr/local/cuda-12.9/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS="$(nproc)"
ldconfig
pip install --no-build-isolation $C --force-reinstall --no-deps -e . 2>&1 | tail -8
echo "DEEPEP_BUILD_RC=${PIPESTATUS[0]}"
LD_LIBRARY_PATH="$N2304:$NVDIR/lib:/usr/local/cuda-12.9/targets/x86_64-linux/lib:$LD_LIBRARY_PATH" python3 -c "import deep_ep;print('DEEPEP_OK')" 2>&1|tail -1
echo "=== STEP 5: build 9c44d34 GIN plugin (v13) + gdrdrv-2.4 gate relax ==="
# PUBLIC root (rule 7): official aws/aws-ofi-nccl @ 9c44d34 + the two committed
# patch scripts (gdr-pin v1-fallback + forced-pcie) from this repo's
# scripts/patches/private-deltas/ (ship as $PATCH_DIR into the pod).
PATCH_DIR="${PATCH_DIR:-/opt/deepep-patches}"
rm -rf /opt/aws-ofi-9c44; git clone https://github.com/aws/aws-ofi-nccl /opt/aws-ofi-9c44 2>&1|tail -1
cd /opt/aws-ofi-9c44 && git checkout 9c44d34476f90ddbf4a12d0ac4fc412d46bd8ab4 2>&1|tail -1
python3 "$PATCH_DIR"/private-deltas/patch-gdr-pin-v1-fallback.py /opt/aws-ofi-9c44/src/nccl_ofi_gdrcopy.cpp
python3 "$PATCH_DIR"/private-deltas/apply_forced_pcie_copy_bypass.py /opt/aws-ofi-9c44/src/nccl_ofi_gdrcopy.cpp
# gate relax patch
python3 - <<'PY'
f='/opt/aws-ofi-9c44/src/rdma/gin/nccl_ofi_gin_api.cpp'; s=open(f).read()
old='''\t\tif (!gdr.forced_pcie_copy()) {
\t\t\tNCCL_OFI_WARN("GIN requires GDRCopy 2.5+ for forced PCIe copy support");
\t\t\treturn ncclInternalError;
\t\t}'''
new='''\t\tif (!gdr.forced_pcie_copy()) {
\t\t\t/* GDRDRV_24_GATE_RELAX */
\t\t\tNCCL_OFI_WARN("GIN: forced PCIe copy unavailable (gdrdrv<2.5); continuing with default GDR pin");
\t\t}'''
if 'GDRDRV_24_GATE_RELAX' not in s and old in s: open(f,'w').write(s.replace(old,new)); print('GATE_PATCHED')
else: print('GATE_SKIP', 'GDRDRV_24_GATE_RELAX' in s)
PY
./autogen.sh 2>&1|tail -1
./configure --prefix=/opt/aws-ofi-9c44-install --with-libfabric=/opt/amazon/efa --with-cuda=/usr/local/cuda-12.9 --enable-platform-aws 2>&1|tail -2
make -j$(nproc) -C src 2>&1 | tail -4
PLUGIN=/opt/aws-ofi-9c44/src/.libs/libnccl-net-ofi.so
echo "PLUGIN_V13=$(nm -D $PLUGIN 2>/dev/null|grep -c ncclGinPlugin_v13)"
echo "=== STEP 6: MNNVL x86 short-circuit patch ==="
python3 - <<'PY'
f='/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_mnnvl_utils.py'; s=open(f).read()
old='''        arch = platform.machine().lower()
        is_on_aarch64 = "aarch64" in arch
        support_nvlink_and_all_up = MnnvlMemory.support_nvlink(True)
        return is_on_aarch64 and support_nvlink_and_all_up'''
new='''        arch = platform.machine().lower()
        is_on_aarch64 = "aarch64" in arch
        # MNNVL_X86_SHORTCIRCUIT
        if not is_on_aarch64:
            return False
        try:
            support_nvlink_and_all_up = MnnvlMemory.support_nvlink(True)
        except Exception:
            return False
        return is_on_aarch64 and support_nvlink_and_all_up'''
if 'MNNVL_X86_SHORTCIRCUIT' not in s and old in s: open(f,'w').write(s.replace(old,new)); print('MNNVL_PATCHED')
else: print('MNNVL_SKIP', 'MNNVL_X86_SHORTCIRCUIT' in s)
PY
echo "BRINGUP_ALL_DONE rc_summary trt=$(python3 -c 'import tensorrt_llm;print(tensorrt_llm.__version__)' 2>&1|tail -1)"

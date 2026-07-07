#!/bin/bash
# Build the EFA+DeepEP substrate on a BARE NGC cuda:13.0.0-devel-ubuntu24.04 pod, from PUBLIC source only.
# Mirrors deliverables/.../repro-hub/Dockerfile.cuda13 Layers 1-5. ~25-35 min.
exec >/tmp/substrate.log 2>&1
set -x
export DEBIAN_FRONTEND=noninteractive
echo "=== L1: apt deps (ubuntu24.04 public repos) ==="
apt-get update -q
apt-get install -y -q --no-install-recommends \
  build-essential autoconf automake libtool pkg-config git curl wget ca-certificates \
  libhwloc-dev libudev-dev libnuma-dev openmpi-bin libopenmpi-dev libopenmpi3 \
  python3 python3-pip python3-dev python3-venv openssh-server openssh-client \
  pciutils environment-modules tcl udev dmidecode ethtool iproute2 kmod libfmt-dev 2>&1 | tail -2
python3 -m pip install --no-cache-dir --break-system-packages --upgrade "pip>=24.0" 2>&1 | tail -1
ln -sf "$(which python3)" /usr/local/bin/python
echo "L1_DONE py=$(python3 --version)"

echo "=== L2: AWS EFA installer 1.48 (public) ==="
curl -fsSL https://efa-installer.amazonaws.com/aws-efa-installer-1.48.0.tar.gz | tar -xzf - -C /tmp
cd /tmp/aws-efa-installer && ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify --disable-ngc --disable-build-ngc 2>&1 | tail -3
echo "L2_DONE efa=$(ls /opt/amazon/efa/lib/libfabric.so* 2>/dev/null|head -1)"
export PATH=/opt/amazon/efa/bin:/opt/amazon/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:${LD_LIBRARY_PATH:-}

echo "=== L3: gdrcopy 2.5.2 (public github) ==="
git clone --depth 1 --branch v2.5.2 https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy 2>&1 | tail -1
cd /tmp/gdrcopy && make prefix=/usr/local lib lib_install 2>&1 | tail -2 && ldconfig
echo "L3_DONE gdrapi=$(ls /usr/local/lib/libgdrapi.so* 2>/dev/null|head -1)"

echo "=== L4: torch 2.11+cu130 + nccl-cu13 2.30.4 (public pytorch index) ==="
pip install --no-cache-dir --break-system-packages torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130 2>&1 | tail -2
pip install --no-cache-dir --break-system-packages --no-deps "nvidia-nccl-cu13==2.30.4" 2>&1 | tail -1
python3 -c "import torch; print('L4_DONE torch', torch.__version__)"

echo "=== L5: DeepEP-V2 @ public deepseek-ai/DeepEP b306af06 + committed patches ==="
# PATCH_DIR = this repo's scripts/patches (ship it into the pod alongside this script)
PATCH_DIR="${PATCH_DIR:-/opt/deepep-patches}"
git clone https://github.com/deepseek-ai/DeepEP.git /opt/DeepEP 2>&1 | tail -1
cd /opt/DeepEP && git checkout b306af06afd4dbfa1a58aba7a7b430b81a2fe270
git -c user.email=build@local -c user.name=build am "$PATCH_DIR"/0001-*.patch "$PATCH_DIR"/0002-*.patch "$PATCH_DIR"/0003-*.patch
git apply "$PATCH_DIR"/private-deltas/deepep-multicomm-overlay.patch "$PATCH_DIR"/private-deltas/deepep-multicomm-newfile.patch
cd - >/dev/null
test -f /opt/DeepEP/csrc/elastic/buffer.hpp && echo "L5_DONE deepep=$(cd /opt/DeepEP && git rev-parse --short HEAD)+patches"
echo "SUBSTRATE_ALL_DONE"

# =============================================================================
# TRT-LLM 0.21 DeepEP-V2-on-EFA — coherent image, NGC-FROM-SCRATCH.
#
# HARD RULE (CLAUDE.md, set 2026-06-23): start from an official NVIDIA NGC base.
# NEVER base on any image WE created — private OR public (no ghcr.io/antonai-work/*,
# no */efa-gda* ECR). A tester with only public NGC + public GitHub reproduces this.
#
# This builds the SAME substrate the convenience base used to provide, but rooted at
# NGC and from public sources (mirrors deliverables/.../repro-hub/Dockerfile.cuda13):
#   NGC cuda:12.9.0-devel-ubuntu24.04  (cuda-12.9 nvcc matches torch cu12x; ubuntu24.04 => python3.12.
#     12.9 base is REQUIRED: torch 2.7.1 is cu126 so the deep_ep _C nvcc build needs a cu12 toolkit, and
#     deep_ep links cudaMemcpyBatchAsync@libcudart.so.12 which only the 12.9 cudart exports. The cuda-13
#     base fails both — validated from-scratch 2026-06-23, see ../docs/EXP-E2E-NGC-FROM-SCRATCH.)
#     -> Layer 1  system build deps
#     -> Layer 2  AWS EFA installer 1.48 (public)            libfabric 2.4 + rdma-core 61
#     -> Layer 3  gdrcopy 2.5.2 (public github)              libgdrapi for CPU-proxy GIN
#     -> Layer 4  torch 2.11+cu130 + nccl-cu13 2.30.4 (public)   the cu13 baseline
#     -> Layer 5  DeepEP-V2 source @ /opt/DeepEP (public fork)   PR#612 + multi-comm
#     -> Layer 6  the TRT-LLM 0.21 + api-shim build chain (scripts/00-bringup...)
#     -> Layer 7  sshd for cross-pod mpirun (TRT is MPI-based)
#
# OPTIONAL: the default repro path is in-pod build on the NGC-substrate image (pods idle
# on sleep infinity; scripts/00 runs via kubectl exec). This Dockerfile bakes the chain
# for CI/air-gapped reruns. The serve + A/B still run via scripts (need the live 2-pod MPI).
#
# Verify the NGC base first:
#   docker run --rm nvcr.io/nvidia/cuda:12.9.0-devel-ubuntu24.04 bash -c 'python3 --version; nvcc --version | tail -1'
#
# RULES: ElasticBuffer-only on EFA (no deep_ep.Buffer / NVSHMEM / IBGDA); NCCL_GIN_TYPE=2
#   CPU-proxy; never NVSHMEM_IB_ENABLE_IBGDA=1; no DeepEP V1.
# =============================================================================
ARG CUDA_VER=12.9.0
FROM nvcr.io/nvidia/cuda:${CUDA_VER}-devel-ubuntu24.04 AS trtllm-efa

LABEL org.opencontainers.image.description="TRT-LLM 0.21 DeepEP-V2 MoE inference on AWS EFA — NGC-from-scratch (api-shim -> ElasticBuffer, CPU-proxy NCCL-GIN)"
LABEL org.opencontainers.image.licenses="Apache-2.0"
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# -----------------------------------------------------------------------------
# Layer 1 — system build deps (ubuntu24.04 public repos; python3.12 is default)
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential autoconf automake libtool pkg-config git curl wget ca-certificates \
      libhwloc-dev libudev-dev libnuma-dev openmpi-bin libopenmpi-dev libopenmpi3 \
      python3 python3-pip python3-dev python3-venv \
      openssh-server openssh-client \
      pciutils environment-modules tcl udev dmidecode ethtool iproute2 kmod libfmt-dev \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir --break-system-packages --upgrade "pip>=24.0" \
    && ln -sf "$(which python3)" /usr/local/bin/python

# -----------------------------------------------------------------------------
# Layer 2 — AWS EFA installer 1.48 (public: https://efa-installer.amazonaws.com/)
# -----------------------------------------------------------------------------
ARG EFA_INSTALLER_VER=1.48.0
RUN curl -fsSL https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VER}.tar.gz \
      | tar -xzf - -C /tmp \
    && cd /tmp/aws-efa-installer \
    && ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify --disable-ngc --disable-build-ngc \
    && echo "${EFA_INSTALLER_VER}" > /opt/efa-installer.version \
    && rm -rf /tmp/aws-efa-installer
ENV PATH=/opt/amazon/efa/bin:/opt/amazon/openmpi/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:${LD_LIBRARY_PATH:-}

# -----------------------------------------------------------------------------
# Layer 3 — gdrcopy 2.5.2 userspace (public: https://github.com/NVIDIA/gdrcopy)
# -----------------------------------------------------------------------------
ARG GDRCOPY_VER=v2.5.2
RUN git clone --depth 1 --branch ${GDRCOPY_VER} https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy \
    && cd /tmp/gdrcopy && make prefix=/usr/local lib lib_install && ldconfig \
    && rm -rf /tmp/gdrcopy

# -----------------------------------------------------------------------------
# Layer 4 — torch cu13 baseline (public PyTorch index) + nccl-cu13 + deep_ep deps.
# This is the cu13 line the in-pod TRT bringup downgrades-from; baking it gives the
# scripts a coherent /usr/local/lib/python3.12/dist-packages to start from.
# -----------------------------------------------------------------------------
ARG TORCH_SPEC="torch==2.11.0"
ARG TORCH_INDEX="https://download.pytorch.org/whl/cu130"
ARG NVIDIA_NCCL_CU13="2.30.4"
RUN pip install --no-cache-dir --break-system-packages ${TORCH_SPEC} --index-url ${TORCH_INDEX} \
    && pip install --no-cache-dir --break-system-packages --no-deps "nvidia-nccl-cu13==${NVIDIA_NCCL_CU13}" \
    && python3 -c "import torch; print('torch', torch.__version__); assert torch.__version__.startswith('2.11')"

# -----------------------------------------------------------------------------
# Layer 5 — DeepEP-V2 source @ /opt/DeepEP (public fork; PR#612 base + multi-comm).
# Source only — the _C.so is an nvcc JIT build done in-pod (needs a CUDA context).
# -----------------------------------------------------------------------------
# PUBLIC root (rule 7): official deepseek-ai/DeepEP @ b306af06 + the committed
# patch set (PR#612 trio + multi-comm overlay) from this repo's scripts/patches/.
ARG DEEPEP_REPO=https://github.com/deepseek-ai/DeepEP.git
ARG DEEPEP_SHA=b306af06afd4dbfa1a58aba7a7b430b81a2fe270
COPY scripts/patches/ /opt/deepep-patches/
RUN git clone ${DEEPEP_REPO} /opt/DeepEP \
    && cd /opt/DeepEP && git checkout ${DEEPEP_SHA} \
    && git -c user.email=build@local -c user.name=build am /opt/deepep-patches/0001-*.patch /opt/deepep-patches/0002-*.patch /opt/deepep-patches/0003-*.patch \
    && git apply /opt/deepep-patches/private-deltas/deepep-multicomm-overlay.patch \
    && git apply /opt/deepep-patches/private-deltas/deepep-multicomm-newfile.patch \
    && git rev-parse HEAD > /opt/deepep.base.sha \
    && test -f /opt/DeepEP/csrc/elastic/buffer.hpp \
    && grep -q EP_NUM_COMMS deep_ep/buffers/elastic.py \
    && echo "DeepEP-V2 public b306af06 + PR612 trio + multicomm staged @ /opt/DeepEP"

# -----------------------------------------------------------------------------
# Layer 6 — api-shim + the TRT-LLM 0.21 build chain (the 9-wall recipe).
# scripts/00-bringup-all-consolidated.sh installs trtllm 0.21 (+cuda-python 12.9) +
# dual-NCCL + cu12-nvshmem + cuda-12.9 cudart + deep_ep _C rebuild + 9c44d34 GIN-v13
# plugin + gdrdrv-2.4 gate relax + MNNVL-x86 short-circuit.
# -----------------------------------------------------------------------------
COPY api-shim/sitecustomize.py             /opt/api-shim/sitecustomize.py
COPY api-shim/api_shim/__init__.py         /opt/api-shim/api_shim/__init__.py
COPY api-shim/api_shim/buffer_v1_compat.py /opt/api-shim/api_shim/buffer_v1_compat.py
COPY api-shim/smoke_test_shim.py           /opt/api-shim/smoke_test_shim.py
COPY scripts/extra_llm_api_options.yaml    /opt/extra_llm_api.yaml
COPY scripts/00-bringup-all-consolidated.sh /opt/build/00-bringup.sh
RUN bash /opt/build/00-bringup.sh \
    && python3 -c "import tensorrt_llm; print('trtllm', tensorrt_llm.__version__)"

# -----------------------------------------------------------------------------
# Layer 7 — sshd for cross-pod mpirun (TRT-LLM is MPI-based, unlike SGLang/vLLM torchrun)
# -----------------------------------------------------------------------------
RUN ssh-keygen -A && mkdir -p /run/sshd \
    && sed -i 's/#\?Port .*/Port 2222/; s/#\?PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config

# Pods idle on sleep infinity, driven by kubectl exec + the serve/A-B scripts.
CMD ["/bin/bash", "-c", "if [ ! -c /dev/gdrdrv ]; then mknod /dev/gdrdrv c 237 0 || true; chmod 666 /dev/gdrdrv || true; fi; mkdir -p /data/hf-cache; setsid /usr/sbin/sshd -D -p 2222 </dev/null >/tmp/sshd.log 2>&1 & echo 'trtllm-efa NGC-from-scratch image ready'; sleep infinity"]

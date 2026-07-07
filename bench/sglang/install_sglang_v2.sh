#!/bin/bash
# Gate-3 install: PUBLIC sglang==0.5.11 (PyPI) + the V2 seam patch (PR #24443
# lineage) + invariant-1 nccl/nvshmem re-pin. Promoted from the 2026-07-07
# PASS install.
# Run inside the serving image (system python3 example; adjust for venv).
# REPO = checkout of this repo (for patches/private-deltas/).
set -euxo pipefail
REPO=${REPO:?path to deepep-v2-integration checkout}

python3 -m pip install --no-cache-dir "sglang[all]==0.5.11" "kernels==0.12.0"

# Session invariant 1: wheel deps silently replace nccl/nvshmem -> deep_ep _C
# undefined symbols. Always re-pin after ANY pip install.
python3 -m pip install --no-cache-dir --force-reinstall --no-deps \
  "nvidia-nccl-cu13==2.30.4" "nvidia-nvshmem-cu13==3.6.5"

# V2 seam patch onto the fresh PyPI wheel (verified git-apply-clean 2026-07-07)
SP=$(python3 -c "import sglang, os; print(os.path.dirname(os.path.dirname(sglang.__file__)))")
cd "$SP"
git apply --check "$REPO/patches/private-deltas/sglang-0.5.11-deepep-v2-seam.patch"
git apply "$REPO/patches/private-deltas/sglang-0.5.11-deepep-v2-seam.patch"
cp "$REPO/patches/private-deltas/sglang-v2_compat_buffer.py" \
   sglang/srt/layers/moe/token_dispatcher/v2_compat_buffer.py

# guard: torch stayed cu130, deep_ep still imports
python3 - <<'PY'
import torch
assert torch.__version__.endswith("+cu130"), f"torch replaced: {torch.__version__}"
import sglang
import sys; sys.path.insert(0, "/opt/DeepEP")
import deep_ep
print("INSTALL_OK sglang", sglang.__version__, "torch", torch.__version__)
PY

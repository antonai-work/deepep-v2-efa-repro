#!/usr/bin/env python3
"""Patch aws-ofi-nccl-gdaki/src/nccl_ofi_gdrcopy.cpp to fall back to v1
when v2 returns EINVAL (cluster has gdrdrv 2.4, libgdrapi 2.5).

Idempotent (looks for marker comment).
"""
import sys, shutil

path = sys.argv[1] if len(sys.argv) > 1 else "/opt/aws-ofi-nccl-gdaki/src/nccl_ofi_gdrcopy.cpp"
src = open(path).read()

if "FALLBACK_V1_FOR_GDRDRV_24" in src:
    print(f"already patched: {path}")
    sys.exit(0)

# Find the v2 block. Match the closing `}` of `if (forced_pcie_copy())` v2 path.
old_block = """\t\tret = pimpl->gdr_pin_buffer_v2_fn(pimpl->gdr, regbgn, handle->gdr_reglen,
\t\t\t\t\t\t  flags, &mh);
\t\tif (ret != 0) {
\t\t\tflags = 0;
\t\t\tret = pimpl->gdr_pin_buffer_v2_fn(pimpl->gdr, regbgn,
\t\t\t\t\t\t\t  handle->gdr_reglen, flags, &mh);
\t\t}
\t} else {"""

new_block = """\t\tret = pimpl->gdr_pin_buffer_v2_fn(pimpl->gdr, regbgn, handle->gdr_reglen,
\t\t\t\t\t\t  flags, &mh);
\t\tif (ret != 0) {
\t\t\tflags = 0;
\t\t\tret = pimpl->gdr_pin_buffer_v2_fn(pimpl->gdr, regbgn,
\t\t\t\t\t\t\t  handle->gdr_reglen, flags, &mh);
\t\t}
\t\t/* FALLBACK_V1_FOR_GDRDRV_24: cluster has gdrdrv 2.4 + libgdrapi 2.5 — v2 ioctl
\t\t * returns EINVAL because kernel doesn't know v2 cmd. Fall back to v1 which is
\t\t * identical at the gdr_mh_t output level (only difference is the flags input).
\t\t */
\t\tif (ret == 22 || ret == 25) {
\t\t\tret = pimpl->gdr_pin_buffer_fn(pimpl->gdr, regbgn,
\t\t\t\t\t\t       handle->gdr_reglen, 0, 0, &mh);
\t\t}
\t} else {"""

if old_block not in src:
    print("FAIL: old_block not found verbatim — anchor drift")
    print("--- search hint ---")
    idx = src.find("gdr_pin_buffer_v2_fn(pimpl->gdr, regbgn")
    if idx >= 0:
        print(repr(src[idx:idx+500]))
    sys.exit(2)

src = src.replace(old_block, new_block)
shutil.copy2(path, path + ".PRE-V1-FALLBACK")
open(path, "w").write(src)
print(f"patched: {path}")
print(f"backup:  {path}.PRE-V1-FALLBACK")

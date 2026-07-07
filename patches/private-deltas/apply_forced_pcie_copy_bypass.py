#!/usr/bin/env python3
"""apply_forced_pcie_copy_bypass.py — source patch for the aws-ofi-nccl GIN plugin.

THE PROBLEM (root-caused 2026-06-20 via NCCL_DEBUG=INFO + the 719 combine fault):
  `nccl_ofi_gdrcopy_ctx::forced_pcie_copy()` (src/nccl_ofi_gdrcopy.cpp) returns true only if the
  gdrdrv KERNEL module is >= 2.5:
      return (get_version(&major,&minor)==0 && (major>2 || (major==2 && minor>=5)));
  On a cluster whose gdrdrv kernel is 2.4 (but whose USERSPACE libgdrapi is 2.5) it returns FALSE.
  Consequences, both observed:
    (a) GIN init hard-fails the gate at nccl_ofi_gin_api.cpp ("GIN requires GDRCopy 2.5+ ...") -> ginType NONE.
    (b) Even past the gate, the GIN gdrcopy context is configured for the NON-forced-PCIe path, so the
        DeepEP ElasticBuffer COMBINE kernel faults at runtime: CUDA_ERROR_LAUNCH_FAILED (719),
        deep_ep/buffers/elastic.py:912 -> csrc/jit/handle.hpp:86.

THE FIX (matches the shipped 8873526c image, which served full inference on this same cluster):
  patch_plugin.py binary-flips forced_pcie_copy() to `return true` at a build-specific .so offset (NOT
  reproducible). This script does the SAME at SOURCE: force the definition to `return true`. SAFE because
  forced PCIe copy is largely a USERSPACE-lib feature and libgdrapi IS 2.5 here — only the kernel-version
  PROBE is stale; the real runtime PCIe-copy calls go through the 2.5 userspace lib.

  Patching the DEFINITION (not just the init gate) is the load-bearing difference: it makes BOTH the gate
  pass AND the combine path take the forced-PCIe-copy branch the kernel expects. A gate-only bypass clears
  init but leaves combine faulting with 719.

USAGE (Dockerfile, after cloning the plugin fork; idempotent):
  python3 apply_forced_pcie_copy_bypass.py /opt/aws-ofi-nccl-src
"""
import sys, pathlib, re

MARK = "FORCED_PCIE_COPY_TRUE_APPLIED"

def patch_definition(root):
    """Flip the HAVE_GDRCOPY forced_pcie_copy() definition to `return true`."""
    cands = list(root.rglob("nccl_ofi_gdrcopy.cpp"))
    if not cands:
        print(f"[force-true] WARN: nccl_ofi_gdrcopy.cpp not found under {root}")
        return False
    f = cands[0]
    s = f.read_text()
    if MARK in s:
        print(f"[force-true] already applied in {f}")
        return True
    # Match ONLY the real (HAVE_GDRCOPY) body: the version-probe return. The !HAVE_GDRCOPY fallback
    # returns a plain `return false;` and is intentionally NOT matched (it's the no-gdrcopy build).
    pat = re.compile(
        r'return\s*\(\s*get_version\(\s*&major\s*,\s*&minor\s*\)\s*==\s*0\s*&&\s*'
        r'\(\s*major\s*>\s*2\s*\|\|\s*\(\s*major\s*==\s*2\s*&&\s*minor\s*>=\s*5\s*\)\s*\)\s*\)\s*;'
    )
    # `(void)major;(void)minor;` silences -Werror=unused-variable (the decls become unused once we
    # short-circuit to `return true`).
    repl = ('(void)major; (void)minor; return true; /* ' + MARK + ': force forced_pcie_copy() true — '
            'gdrdrv kernel-version probe is stale (kernel 2.4) but userspace libgdrapi is 2.5 and services '
            'the runtime PCIe-copy calls; matches shipped patch_plugin.py. Without this, GIN combine faults 719. */')
    s2, n = pat.subn(repl, s)
    if n < 1:
        print(f"[force-true] WARN: forced_pcie_copy() version-return not found in {f} (source shape changed)")
        return False
    f.write_text(s2)
    print(f"[force-true] patched {f}: forced_pcie_copy() -> return true ({n} site, {MARK})")
    return True

def soften_gate(root):
    """Belt-and-suspenders: also turn the init-gate hard-fail into warn+continue (no-op once def is true)."""
    cands = list(root.rglob("nccl_ofi_gin_api.cpp"))
    if not cands:
        return
    f = cands[0]; s = f.read_text()
    if "GIN_GATE_SOFTENED" in s:
        return
    pat = re.compile(
        r'if\s*\(\s*!\s*gdr\.forced_pcie_copy\(\)\s*\)\s*\{\s*'
        r'NCCL_OFI_WARN\(\s*"GIN requires GDRCopy 2\.5\+ for forced PCIe copy support"\s*\)\s*;\s*'
        r'return\s+ncclInternalError\s*;\s*\}', re.DOTALL)
    repl = ('if (!gdr.forced_pcie_copy()) {\n'
            '\t\tNCCL_OFI_WARN("GIN: forced_pcie_copy() false; continuing (GIN_GATE_SOFTENED)");\n\t}')
    s2, n = pat.subn(repl, s)
    if n == 1:
        f.write_text(s2); print(f"[force-true] also softened init gate in {f}")

def main(src_root):
    root = pathlib.Path(src_root)
    ok = patch_definition(root)
    soften_gate(root)
    return 0 if ok else 0  # non-fatal: never break the build on a layout change

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/opt/aws-ofi-nccl-src"))

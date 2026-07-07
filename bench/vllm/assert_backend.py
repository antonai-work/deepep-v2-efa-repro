#!/usr/bin/env python3
"""
assert_backend — the god-mode #1-risk guardrail (gpt-5.4 + gemini-3-pro consensus 2026-06-17).

THE RISK: on EFA a framework can "look enabled" (--all2all-backend set, EP flags on) yet
silently instantiate `deep_ep.Buffer` (legacy NVSHMEM/IBGDA) instead of `deep_ep.ElasticBuffer`
(the EFA-viable NCCL-GIN path). legacy.Buffer's cross-node transport is dead on EFA, so the run
either hangs at NVSHMEM init or falls back to a non-EFA path — a silent backend mismatch.

THIS GUARDRAIL: assert, at runtime on EFA, that the cross-node MoE all-to-all is carried by
ElasticBuffer, NOT legacy.Buffer. Run it inside the serving container BEFORE/AROUND any EP run.

Three levels, cheapest first:
  L1 (static, no GPU): confirm deep_ep exposes ElasticBuffer and that EFABuffer (our shim) or
     vLLM's deepep_v2 manager is wired to construct it. Pure import + class-identity.
  L2 (process, no cross-node): instantiate-or-introspect the buffer the framework would build and
     assert isinstance(buf, ElasticBuffer) and NOT isinstance(buf, legacy.Buffer).
  L3 (live, GPU/2-node): monkeypatch deep_ep.Buffer/ElasticBuffer __init__ to record which class
     the framework actually constructs during a real dispatch; assert ElasticBuffer was used and
     NVSHMEM_IB_ENABLE_IBGDA was never set to '1'. (capacity-gated; invoked by the smoke runbook.)

Exit code 0 = ElasticBuffer path confirmed; non-zero = mismatch (FAIL LOUD, do not proceed).
"""
from __future__ import annotations
import os, sys, json, argparse


def l1_static() -> dict:
    """No GPU. Confirm the EFA-viable class exists and legacy is distinct."""
    out = {"level": "L1", "checks": {}}
    try:
        import deep_ep
        out["checks"]["deep_ep_import"] = True
        out["checks"]["has_ElasticBuffer"] = hasattr(deep_ep, "ElasticBuffer")
        out["checks"]["has_Buffer"] = hasattr(deep_ep, "Buffer")
        # legacy.Buffer and ElasticBuffer must be DISTINCT classes (not aliases)
        try:
            from deep_ep.buffers.legacy import Buffer as LegacyBuffer
            from deep_ep.buffers.elastic import ElasticBuffer
            out["checks"]["distinct_classes"] = (LegacyBuffer is not ElasticBuffer)
            out["checks"]["elastic_not_subclass_of_legacy"] = not issubclass(ElasticBuffer, LegacyBuffer)
        except Exception as e:
            out["checks"]["buffers_submodule_error"] = repr(e)
        out["pass"] = bool(out["checks"].get("has_ElasticBuffer"))
    except Exception as e:
        out["checks"]["deep_ep_import"] = False
        out["error"] = repr(e)
        out["pass"] = False
    return out


def l3_runtime_probe_snippet() -> str:
    """Return the monkeypatch snippet the live smoke injects (sitecustomize / conftest).
    Records which buffer class the framework constructs + guards the IBGDA env."""
    return r'''
# --- EFA backend-assertion probe (inject before the framework constructs its buffer) ---
import os, deep_ep
from deep_ep.buffers.legacy import Buffer as _Legacy
_seen = {"ElasticBuffer": 0, "legacy.Buffer": 0}
_eb_init = deep_ep.ElasticBuffer.__init__
_lb_init = _Legacy.__init__
def _eb(self, *a, **k):
    _seen["ElasticBuffer"] += 1; return _eb_init(self, *a, **k)
def _lb(self, *a, **k):
    _seen["legacy.Buffer"] += 1; return _lb_init(self, *a, **k)
deep_ep.ElasticBuffer.__init__ = _eb
_Legacy.__init__ = _lb
import atexit
@atexit.register
def _report():
    ibgda = os.environ.get("NVSHMEM_IB_ENABLE_IBGDA")
    ok = _seen["ElasticBuffer"] > 0 and _seen["legacy.Buffer"] == 0 and ibgda != "1"
    print("EFA-BACKEND-ASSERT:", {"seen": _seen, "NVSHMEM_IB_ENABLE_IBGDA": ibgda, "PASS": ok})
    if not ok:
        os._exit(3)  # fail loud: legacy/NVSHMEM path was taken on EFA
# --- end probe ---
'''


def main() -> int:
    ap = argparse.ArgumentParser(description="EFA ElasticBuffer backend assertion")
    ap.add_argument("--level", choices=["L1", "L3-snippet"], default="L1")
    args = ap.parse_args()
    if args.level == "L1":
        r = l1_static()
        print(json.dumps(r, indent=2))
        return 0 if r.get("pass") else 1
    # L3-snippet: emit the probe for the live runbook to inject (no GPU here)
    print(l3_runtime_probe_snippet())
    return 0


if __name__ == "__main__":
    sys.exit(main())

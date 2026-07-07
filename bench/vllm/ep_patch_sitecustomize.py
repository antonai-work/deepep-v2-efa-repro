# Gate-2 serve patch: ElasticBuffer timeout injection + legacy-Buffer trap +
# flashinfer kill, delivered as a sitecustomize/.pth hook. Committed verbatim
# (env-gate names included) — the exact hook active during the 2026-07-07 vLLM AIPerf PASS.
#
# Install into the SERVING python env (venv example):
#   cp bench/vllm/ep_patch_sitecustomize.py <venv>/lib/python3.*/site-packages/t3_ep_patch.py
#   echo "import t3_ep_patch" > <venv>/lib/python3.*/site-packages/zzz_t3_ep_patch.pth
# then export T3_EP_PATCH=1 (plus optional EP_GPU_TIMEOUT_SECS/EP_CPU_TIMEOUT_SECS,
# default 600). The .pth line is REQUIRED on Ubuntu: the stdlib sitecustomize
# shadows a venv sitecustomize.py, so a plain sitecustomize copy never loads.
#
# LAZY by necessity: hooks __import__ and patches when deep_ep first loads —
# a .pth executes before dist-packages is on sys.path, so eager import fails.
# What it fixes (both root-caused live, docs/loop-findings/):
# (1) vLLM 0.24 passes NO timeouts to ElasticBuffer; the 100s gpu default loses
#     to cross-rank first-dispatch JIT skew -> "Gin barrier timeout tag:8".
# (2) flashinfer is the vLLM serve killer on minimal pods (JIT needs ninja;
#     cubin/python version clashes) — force every has_*() gate False so vLLM
#     takes the campaign-proven triton MoE path.
# (3) constructing legacy deep_ep.Buffer on EFA is always a bug -> exit 3.
import builtins as _b
import os as _os

if _os.environ.get("T3_EP_PATCH") == "1":
    _orig_import = _b.__import__
    _state = {"done": False, "fi_done": False}

    def _kill_flashinfer(mod):
        n = 0
        for name in dir(mod):
            if name.startswith("has_") and callable(getattr(mod, name)):
                setattr(mod, name, (lambda *a, **k: False))
                n += 1
        print(f"[t3-patch] vllm.utils.flashinfer: forced {n} has_* gates -> False", flush=True)

    def _apply(mod):
        try:
            eb = getattr(mod, "ElasticBuffer", None)
            if eb is None:
                return False
            _oinit = eb.__init__

            def _pinit(self, *a, **k):
                k.setdefault("num_gpu_timeout_secs", int(_os.environ.get("EP_GPU_TIMEOUT_SECS", "600")))
                k.setdefault("num_cpu_timeout_secs", int(_os.environ.get("EP_CPU_TIMEOUT_SECS", "600")))
                print("[t3-patch] constructing deep_ep.ElasticBuffer "
                      f"(gpu_timeout={k['num_gpu_timeout_secs']}s cpu_timeout={k['num_cpu_timeout_secs']}s)",
                      flush=True)
                return _oinit(self, *a, **k)

            eb.__init__ = _pinit
            try:
                from deep_ep.buffers.legacy import Buffer as _Legacy

                def _linit(self, *a, **k):
                    print("[t3-patch] FATAL: legacy deep_ep.Buffer constructed on EFA", flush=True)
                    _os._exit(3)

                _Legacy.__init__ = _linit
            except Exception:
                pass
            print("[t3-patch] installed (ElasticBuffer timeout inject + legacy trap)", flush=True)
            return True
        except Exception as e:
            print("[t3-patch] apply failed:", repr(e), flush=True)
            return True  # don't retry forever

    def _hook(name, globals=None, locals=None, fromlist=(), level=0):
        mod = _orig_import(name, globals, locals, fromlist, level)
        import sys as _sys
        if not _state["done"] and (name == "deep_ep" or name.startswith("deep_ep.")):
            real = _sys.modules.get("deep_ep")
            if real is not None and _apply(real):
                _state["done"] = True
        if not _state["fi_done"] and "vllm.utils.flashinfer" in _sys.modules:
            _kill_flashinfer(_sys.modules["vllm.utils.flashinfer"])
            _state["fi_done"] = True
        if _state["done"] and _state["fi_done"]:
            _b.__import__ = _orig_import
        return mod

    _b.__import__ = _hook

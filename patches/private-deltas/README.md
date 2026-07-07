# private-deltas — every fork-only capability, as patches on PUBLIC roots

Chain-of-custody rule 7: nothing here requires any repo/image we published.
Each patch applies onto a named PUBLIC upstream. Verified desk-side 2026-07-07
(`git apply --check` on fresh clones/wheels; see loop-findings it3/it4).

| Patch | Applies onto (PUBLIC root) | What it adds | Upstream path |
|---|---|---|---|
| `../0001..0003-*.patch` (PR #612 trio) | `deepseek-ai/DeepEP @ b306af06` | EFA auto-QP cap 2, get_rdma_gbs EFA fast path, kScaleoutUpdateInterval 3->16 | DeepEP PR #612 (OPEN) |
| `deepep-multicomm-overlay.patch` + `deepep-multicomm-newfile.patch` | DeepEP b306af06 + trio (apply trio FIRST) | EP_NUM_COMMS>1 multi-comm (G2c bridge, K independent comms over one buffer; the vLLM 2.09x EPNC=8 win). Inert when EP_NUM_COMMS unset/1. | PR candidate (not yet filed) |
| `sglang-0.5.11-deepep-v2-seam.patch` + `sglang-v2_compat_buffer.py` (copy in) | PyPI wheel `sglang==0.5.11` (`git apply --check` clean on unpacked wheel) | `SGLANG_DEEPEP_USE_V2=1` -> constructs `deep_ep.ElasticBuffer` (normal mode only) | SGLang PR #24443 (OPEN) |
| `patch-gdr-pin-v1-fallback.py` | `aws/aws-ofi-nccl @ 9c44d34` (public commit) | gdr_pin_buffer_v2 EINVAL/ENOTTY -> v1 ioctl fallback (gdrdrv kernel-2.4 + libgdrapi-2.5 skew) | PR-A candidate |
| `apply_forced_pcie_copy_bypass.py` | same | forced_pcie_copy() kernel-version probe is stale on 2.4/2.5 skew; force true (userspace lib IS 2.5) | PR-A candidate |

Apply order (DeepEP): `git checkout b306af06 && git am 0001 0002 0003 && git apply
deepep-multicomm-overlay.patch deepep-multicomm-newfile.patch`.

`deepep-base-commits.txt` records the pod HEAD these were extracted from
(2026-07-07, deepep-efa-bench-1) — proving the serving runs' provenance:
public b306af06 + exactly the trio + exactly this overlay.

Serving-time env knobs proven this loop (no code change needed):
- vLLM: flashinfer must be uninstalled/gated off (triton MoE path);
  ElasticBuffer needs `num_gpu/cpu_timeout_secs>=600` at serve (inject via
  `bench/vllm/` sitecustomize hook or pass-through once vLLM exposes it).
- SGLang: `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024` (default 128
  trips `buffer.hpp:686` under AIPerf-conc32 prefill; sglang caps knob at 1024).

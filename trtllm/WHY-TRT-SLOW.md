# Why is the TRT-LLM DeepEP arm "slow"? (and what "slow" actually means)

This answers the question precisely, grounded in the measured A/B (`results/`), not intuition.

## First: TRT-LLM is the FASTEST of the three frameworks in absolute terms

At conc=4, OSL=128, Qwen3-30B-A3B-FP8, EP16 over EFA — aggregate output throughput:

| Framework | agg tok/s (DeepEP arm) |
|---|---|
| SGLang | 18.6 |
| vLLM | 38.5 |
| **TRT-LLM** | **64.24** |

So "TRT is slow" is **not** true across frameworks — it's the opposite. What's slow is the **DeepEP arm
relative to TRT's own non-DeepEP (dense) arm**:

| TRT-LLM arm | agg tok/s | per-user tok/s | TTFT p50 | ITL |
|---|---|---|---|---|
| **DeepEP** (`AlltoallMethodType.DeepEP`) | 64.24 | 19.27 | 210.9 ms | ~52 ms |
| **Dense** (`NotEnabled` → allgather) | **106.36** | **28.06** | **116.9 ms** | lower |

**Non-DeepEP is 1.66× faster.** That's the real "slowness": turning DeepEP ON makes TRT-LLM slower than
leaving it off, on EFA. Same finding holds for vLLM (DeepEP *ties* stock) — see the 3-framework table in
`README.md`.

## Root cause: DeepEP on EFA is a CPU-proxy path, and it's issue-bound

DeepEP-V2 on EFA runs over **NCCL-GIN type-2 (CPU proxy)**. There is no GPU-initiated RDMA on EFA (no IBGDA,
no GPUDirect Async, no MNNVL). Every per-token MoE dispatch and combine is:

```
GPU kernel writes to a D2H ring  →  CPU proxy thread polls the ring  →  CPU posts fi_writedata to the EFA NIC
   →  SRD over the wire  →  peer CPU proxy  →  peer GPU
```

The campaign measured this path directly:

- **Issue-bound, not wire-bound.** EFA NIC utilization sits at ~2.2% during dispatch — the proxy thread
  cannot *issue* operations fast enough to saturate the wire. The bottleneck is the per-op CPU cost, not
  bandwidth. (Per-thread ~4 GB/s vs the same EFA silicon doing ~15 GB/s under a raw-ibverbs issuer like UCCL.)
- At **decode** (one token per step) with **small payloads** and **low concurrency**, the MoE all-to-all
  fires constantly with tiny messages. Each fires the full CPU-proxy round-trip → ITL ~52 ms → ~19 tok/s/user.
- DeepEP's structural pitch is **O(topk) communication-volume reduction** vs a dense allgather. That saving is
  real in *bytes*, but at decode the bytes are tiny, so the volume saving is negligible while the per-op CPU
  tax is paid in full on every step.

## Why the dense (NotEnabled) arm wins

The `NotEnabled` arm doesn't do a per-token expert-parallel all-to-all at all. It moves activations with a
**bulk NCCL allgather/reducescatter** — one large, well-optimized collective per layer, GPU-driven through
the NCCL fast path, no per-token CPU-proxy scatter. Larger messages amortize far better on EFA SRD, so it
delivers lower ITL and ~1.66× the throughput at this regime.

## When would DeepEP actually win?

DeepEP's edge needs BOTH of:

1. **A payload regime where O(topk) volume reduction matters** — prefill (long prompts, big token batches),
   high expert counts, high concurrency. The decode-conc4 cell measured here is the worst case for DeepEP.
2. **A transport that isn't CPU-proxy-issue-bound** — GPU-initiated RDMA (IBGDA on InfiniBand, or GDAKI). On
   InfiniBand DeepEP posts WQEs from the GPU and saturates the NIC; on EFA the CPU proxy is the ceiling.

EFA exposes neither IBGDA nor (working) GDAKI for this path today, so on EFA DeepEP reaches **parity-or-worse**
with the dense path at the serving layer. This is the campaign's central, now-3-framework-confirmed finding.

## One-line summary

TRT-LLM isn't slow — it's the fastest framework here. **DeepEP-on-EFA is the slow part**, because EFA forces
DeepEP through a CPU proxy that is issue-bound (~2.2% NIC util), and at decode the per-op CPU tax outweighs
DeepEP's communication-volume saving — so a bulk NCCL allgather (DeepEP OFF) is 1.66× faster.

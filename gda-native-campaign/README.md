# Native GPU-initiated DeepEP-V2 on EFA (type-5 EFA-GDA) — latency campaign evidence

Campaign window: 2026-07-09 -> 2026-07-17 (banked 07-14, lever-exhaust re-bank 07-17).
Hardware: 2x p5en.48xlarge (H200, 16 EFA NICs/node), 16 EP ranks, N=128 decode shape.

## What this is

The DeepEP-V2 sections of the guidance site measure the CPU-PROXY GIN backend
(NCCL_GIN_TYPE=2): the NIC work is posted by a host proxy thread. This campaign
built and measured the NATIVE path: GPU kernels post WQEs directly to the EFA
NIC (kernel-posted-WQE, "type-5 EFA-GDA" backend in a patched aws-ofi-nccl +
NCCL device stack), with CQ-poll completion (no hardware completion counters
required — those remain unshipped for EFA).

## Headline (banked 2026-07-14, unchanged by the 07-17 re-bank)

| axis | native GPU-initiated | CPU proxy | delta |
|---|---:|---:|---|
| dispatch p50 | 338.5 us | 391.6 us | native -53 us (~14%) |
| combine p50 | 480.7 us | 543.4 us | native -63 us (~12%) |
| cached dispatch p50 | ~401 us | ~274 us | proxy wins (attributed: 90% receiver-forward poll) |

Correctness: dispatch+combine torch.equal == NCCL reference, >=2 reps, with
receiver-side landing proof (sender CQ status=0 was never accepted as landing).

## Files

- `results/campaign-summary.json` — every number above plus the full
  optimization-ladder and re-bank adjudications, machine-readable.
- Provenance: each value is copied verbatim from run custody packets
  (CUSTODY.md + RESULTS-*.md + parsed JSONs) in the campaign workspace;
  packets include per-run rank rc, backend-selection prints, preflight
  records, source hashes, and post-run hygiene. This directory is the
  sanitized public projection of that chain.

## Honest verdict

Native GPU-initiated beats the CPU proxy on dispatch and combine at decode
shape — the first such result on EFA we are aware of — but it does NOT close
the gap to the DeepEP-V2 IB/IBGDA reference (~2.5-2.9x remains). A 73%
reduction in doorbell+fence events moved combine latency ~0 us, and cached
dispatch is ~90% receiver-forward poll: the residual is EFA-SRD wire and
readiness physics, not issue-side code. The proxy remains the supported,
reproducible path for production use today; the native path is a measured
research result with its remaining blockers documented (put_many span posting
parked on a codegen-fragile hang; deferred-doorbell aggregation parked
correctness-red; hardware completion counters still unshipped).

Sanitized — no internal IPs, registry IDs, or hostnames.

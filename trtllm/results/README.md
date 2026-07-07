# Results — TRT-LLM DeepEP vs non-DeepEP A/B (AIPerf)

Raw AIPerf 0.10.0 exports from the run that produced the headline. Same model, pods, EP size, and AIPerf
config for both arms — only the MoE all-to-all backend differs.

## Config (both arms identical except the arm flag)
- Model: Qwen3-30B-A3B-FP8 (128 experts, top-8)
- Topology: EP16 = 2 × p5en.48xlarge (H200, 16 EFA NICs/node), gdrdrv kernel 2.4
- Serve: TRT-LLM 0.21 + api-shim → ElasticBuffer, `NCCL_GIN_TYPE=2` (CPU-proxy), 9c44d34 plugin
- AIPerf: `--concurrency 4 --request-count 40 --synthetic-input-tokens-mean 120 --output-tokens-mean 128 --extra-inputs ignore_eos:true`

## Files
| File | Arm | env flag | selected backend (serve log) |
|---|---|---|---|
| `trtllm-deepep-arm-conc4.json` | DeepEP | `TRTLLM_CAN_USE_DEEP_EP=1`, `TRTLLM_MOE_DISABLE_ALLTOALLV=0` | `AlltoallMethodType.DeepEP` |
| `trtllm-dense-arm-conc4.json` | non-DeepEP (dense) | `TRTLLM_MOE_DISABLE_ALLTOALLV=1`, `TRTLLM_CAN_USE_DEEP_EP=0` | `AlltoallMethodType.NotEnabled` (allgather) |

## Numbers (read from the JSONs)
| Metric | DeepEP | Dense (non-DeepEP) |
|---|---|---|
| request_count | 40 | 40 |
| error_request_count | 0 | 0 |
| output_token_throughput (agg, tok/s) | 64.24 | **106.36** |
| output_token_throughput_per_user (tok/s) | 19.27 | **28.06** |
| time_to_first_token p50 (ms) | 210.9 | **116.9** |

**Non-DeepEP is 1.66× faster** (106.36 / 64.24). Root cause in `../WHY-TRT-SLOW.md`.

## How to re-read a JSON
```bash
PYTHONPATH=/opt/aiperf-pkgs python3 -c "
import json; d=json.load(open('trtllm-deepep-arm-conc4.json'))
for k in ['request_count','output_token_throughput','output_token_throughput_per_user','time_to_first_token']:
    print(k, d[k])"
```

#!/bin/bash
# AIPerf validation sweep — the MANDATORY inference acceptance gate for this tree
# (operator directive 2026-07-03). Runs NVIDIA AIPerf against a LIVE serve on
# :8000 and extracts the standard metrics. Framework-agnostic: point it at any
# OpenAI-compatible endpoint (vLLM or SGLang DeepEP-V2 serve).
#
# Adapted from the campaign harness
#   efa-gda/.../repro-hub/benchmarks/aiperf/aiperf_sweep.sh
# preserving the exact comparable shape: aiperf==0.10.0, ISL120/OSL128,
# ignore_eos, concurrency 4/8/32 (matches CAMPAIGN-MULTINODE-FRAMEWORKS-20260620
# so results compare 1:1 to the SGLang-vs-UCCL + vLLM baselines).
#
# CLUSTER-GATED: this runs against a live GPU serve on 2x/4x p5en. This desk tree
# does NOT execute it — it is a `rivian` hand-off (cgk/Jakarta cluster is theirs).
# See bench/aiperf/README.md.
#
# Env:
#   ARM        label for the output dir (e.g. deepep-vllm, deepep-sglang, uccl)
#   URL        serve base URL (default http://127.0.0.1:8000)
#   MODEL      served model id; auto-detected from /v1/models if unset
#   TOKENIZER  tokenizer path or HF id; defaults to MODEL
set -uo pipefail
ARM="${ARM:-deepep-v2}"
URL="${URL:-http://127.0.0.1:8000}"
export PYTHONPATH="${PYTHONPATH:-/opt/aiperf-pkgs}"

# Isolated install so aiperf's deps never perturb the serving venv.
[ -x /opt/aiperf-pkgs/bin/aiperf ] || \
  pip install --no-cache-dir --target=/opt/aiperf-pkgs --ignore-installed blinker \
    "aiperf==0.10.0" >/tmp/aiperf_install.log 2>&1

MODEL="${MODEL:-$(curl -s "${URL}/v1/models" | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])')}"
TOKENIZER="${TOKENIZER:-$MODEL}"
echo "AIPerf sweep: arm=${ARM} model=${MODEL} url=${URL}"

run_cell() { # $1 label $2 conc $3 isl $4 osl $5 reqcount
  local lbl="$1" c="$2" isl="$3" osl="$4" rc="$5"
  local D="/tmp/aiperf-${ARM}-${lbl}"
  rm -rf "$D"; mkdir -p "$D"
  /opt/aiperf-pkgs/bin/aiperf profile --model "$MODEL" --tokenizer "$TOKENIZER" --url "$URL" \
    --endpoint-type completions --streaming --concurrency "$c" --request-count "$rc" \
    --synthetic-input-tokens-mean "$isl" --output-tokens-mean "$osl" --extra-inputs ignore_eos:true \
    --artifact-dir "$D" >"/tmp/aiperf_${ARM}_${lbl}.log" 2>&1
  local ec=$?
  PYTHONPATH=/opt/aiperf-pkgs python3 -c "
import json,glob
f=glob.glob('$D/**/profile_export_aiperf.json',recursive=True)
if not f: print('CELL $lbl arm=$ARM NO_JSON exit=$ec'); raise SystemExit
d=json.load(open(f[0]))
print('CELL $lbl arm=$ARM conc=$c isl=$isl osl=$osl reqs',d['request_count']['avg'],
      'agg_tps',round(d['output_token_throughput']['avg'],2),
      'per_user',round(d['output_token_throughput_per_user']['avg'],2),
      'ttft_p50_ms',round(d['time_to_first_token']['p50'],1))
"
}

# The campaign-comparable 3-cell shape.
run_cell c4_isl120   4  120  128 40
run_cell c8_isl120   8  120  128 64
run_cell c32_isl120  32 120  128 128
echo "SWEEP_DONE arm=${ARM}"

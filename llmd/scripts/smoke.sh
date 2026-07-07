#!/usr/bin/env bash
# smoke.sh - Prove llm-d scheduler routes a /v1/chat/completions request
# to the vllm-sim backend and returns a token.
#
# Usage:
#   scripts/smoke.sh [output_dir]
#
# If output_dir is omitted, logs are archived to
# bench/logs/llmd-shim-smoke-<UTC_TIMESTAMP>/ at the repo root.

set -euo pipefail

NS="deepep-llmd-shim-test"
RELEASE="deepep-v2"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${1:-${REPO_ROOT}/bench/logs/llmd-shim-smoke-${TS}}"

mkdir -p "${OUT_DIR}"

# Service + port the EPP chart stood up. The "http" port (8081) is where
# Envoy's vLLM-compat listener lives - it runs the ExtProc filter against
# the EPP on 9002 and then proxies to whichever vllm-sim pod the EPP picks.
EPP_SVC="${RELEASE}-epp"
EPP_PORT=8081

step() { printf "\n=== %s ===\n" "$*" | tee -a "${OUT_DIR}/smoke.log"; }
run() { echo "$ $*" | tee -a "${OUT_DIR}/smoke.log"; "$@" 2>&1 | tee -a "${OUT_DIR}/smoke.log"; }

step "Cluster state snapshot"
run kubectl -n "${NS}" get pods -o wide
run kubectl -n "${NS}" get svc
run kubectl -n "${NS}" get inferencepool

step "Port-forward EPP service"
# We spawn port-forward in background and kill on exit. Choosing a high
# ephemeral port on localhost to avoid colliding with anything the user
# already has bound.
LOCAL_PORT="${LOCAL_PORT:-18081}"
kubectl -n "${NS}" port-forward "svc/${EPP_SVC}" "${LOCAL_PORT}:${EPP_PORT}" \
  > "${OUT_DIR}/port-forward.log" 2>&1 &
PF_PID=$!
trap 'kill $PF_PID 2>/dev/null || true' EXIT
# Wait for the listener to come up
for i in $(seq 1 30); do
  if curl -sf "http://localhost:${LOCAL_PORT}/v1/models" >/dev/null 2>&1; then
    echo "port-forward ready after ${i}s" | tee -a "${OUT_DIR}/smoke.log"
    break
  fi
  sleep 1
done

step "GET /v1/models via EPP/Envoy"
# This exercises the routing path: Envoy -> ExtProc -> EPP -> InferencePool
# endpoint selection -> Envoy upstream to vllm-sim:8000/v1/models.
# If this returns a model list, the plumbing is wired.
run curl -sS -i -o "${OUT_DIR}/resp-models.txt" \
  "http://localhost:${LOCAL_PORT}/v1/models"
cat "${OUT_DIR}/resp-models.txt" | head -40 | tee -a "${OUT_DIR}/smoke.log"

step "POST /v1/chat/completions via EPP/Envoy"
run curl -sS -i -o "${OUT_DIR}/resp-chat.txt" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-30B-A3B-FP8",
    "messages": [{"role": "user", "content": "Say hi in one word."}],
    "max_tokens": 16,
    "temperature": 0
  }' \
  "http://localhost:${LOCAL_PORT}/v1/chat/completions"
cat "${OUT_DIR}/resp-chat.txt" | head -60 | tee -a "${OUT_DIR}/smoke.log"

step "Capture pod logs"
for pod in $(kubectl -n "${NS}" get pods -o name); do
  name=$(basename "${pod}")
  kubectl -n "${NS}" logs "${pod}" --all-containers --tail=500 \
    > "${OUT_DIR}/logs-${name}.txt" 2>&1 || true
done

step "Summary"
if grep -q '"choices"' "${OUT_DIR}/resp-chat.txt"; then
  echo "LLM-D ROUTING SMOKE: PASS -- real completion token returned" \
    | tee -a "${OUT_DIR}/smoke.log"
  exit 0
else
  echo "LLM-D ROUTING SMOKE: FAIL -- see ${OUT_DIR}/resp-chat.txt" \
    | tee -a "${OUT_DIR}/smoke.log"
  exit 1
fi

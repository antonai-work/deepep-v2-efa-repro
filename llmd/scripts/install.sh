#!/usr/bin/env bash
# install.sh - Deploy llm-d scheduler + vllm-sim backend in deepep-llmd-shim-test.
#
# Idempotent. Safe to run multiple times. Prints the EPP Service address
# you can curl against at the end.
#
# Prereqs (not installed by this script; install them out-of-band):
#   - kubectl v1.28+
#   - helm v3.10+
#   - GAIE v1.4.0 CRDs applied:
#       kubectl apply -k "https://github.com/kubernetes-sigs/gateway-api-inference-extension/config/crd?ref=v1.4.0"

set -euo pipefail

NS="deepep-llmd-shim-test"
RELEASE="deepep-v2"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
DEPLOY_DIR="${SCRIPT_DIR}/../deploy"
CHART="oci://registry.k8s.io/gateway-api-inference-extension/charts/standalone"
CHART_VERSION="v1.4.0"

step() { printf "\n=== %s ===\n" "$*"; }

step "Namespace"
kubectl apply -f "${DEPLOY_DIR}/00-namespace.yaml"

step "GAIE CRDs (no-op if already installed)"
kubectl apply -k "https://github.com/kubernetes-sigs/gateway-api-inference-extension/config/crd?ref=v1.4.0" | tail -5

step "Backend: vllm-sim deployment + service"
kubectl apply -f "${DEPLOY_DIR}/10-vllm-sim-deployment.yaml"

# 20-inferencepool.yaml is reference-only; the GAIE standalone Helm
# chart creates its own InferencePool driven by the values file in
# 30-scheduler-values.yaml (inferencePool.modelServers.matchLabels).
# See the comment block at the top of 20-inferencepool.yaml.

step "EPP + Envoy (Helm)"
if helm -n "${NS}" status "${RELEASE}" >/dev/null 2>&1; then
  helm -n "${NS}" upgrade "${RELEASE}" \
    "${CHART}" --version "${CHART_VERSION}" \
    -f "${DEPLOY_DIR}/30-scheduler-values.yaml"
else
  helm -n "${NS}" install "${RELEASE}" \
    "${CHART}" --version "${CHART_VERSION}" \
    -f "${DEPLOY_DIR}/30-scheduler-values.yaml"
fi

step "Wait for vllm-sim Ready"
kubectl -n "${NS}" wait --for=condition=Available --timeout=180s deployment/vllm-sim

step "Wait for EPP Ready"
# Helm release name is just "${RELEASE}"; the chart names the deployment
# "${RELEASE}-epp" (epp = endpoint picker).
kubectl -n "${NS}" wait --for=condition=Available --timeout=180s deployment/"${RELEASE}-epp"

step "Summary"
kubectl -n "${NS}" get pods -o wide
echo
echo "EPP Service (curl this):"
kubectl -n "${NS}" get svc "${RELEASE}-epp" -o wide
echo
echo "Next:"
echo "  scripts/smoke.sh"

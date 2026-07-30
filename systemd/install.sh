#!/usr/bin/env bash
# Install + enable the llama-swap architecture (needs sudo). Enables start-on-boot (WSL systemd).
#
# ARCHITECTURE CHANGE (was: qwen-text + qwen-vision + qwen-router, all resident):
#   now: qwen-swap (llama-swap :9000, owns BOTH llama-servers, one at a time)
#      + qwen-router (:8000, tool-budget guard -> :9000)
#   vision is RETIRED; qwen-text/qwen-vision are RETIRED as units because llama-swap now spawns
#   the llama-server processes itself. Leaving qwen-text enabled would make it grab port 8080 and
#   23 GB of GPU0 at boot, and llama-swap's qwen-27b would then fail to bind -- the same
#   stale-process conflict as before, but permanent and boot-triggered.
set -euo pipefail
cd "$(dirname "$0")"

echo "== retiring the old per-server units (llama-swap owns those processes now) =="
sudo systemctl disable --now qwen-text.service   2>/dev/null || true
sudo systemctl disable --now qwen-vision.service 2>/dev/null || true
sudo systemctl reset-failed qwen-text.service qwen-vision.service 2>/dev/null || true

echo "== installing units =="
for u in qwen-swap.service qwen-router.service; do
  sudo cp "$u" /etc/systemd/system/"$u"
done
sudo systemctl daemon-reload
sudo systemctl enable --now qwen-swap.service qwen-router.service
# router may have been running the OLD proxy (pointing at :8080/:8081) -- force it onto the new one
sudo systemctl restart qwen-router.service

echo
echo "installed + enabled. status:"
systemctl --no-pager status qwen-swap qwen-router | grep -E "Loaded|Active" || true
echo
echo "retired (should be 'disabled'):"
for u in qwen-text qwen-vision; do printf "  %-14s %s\n" "$u" "$(systemctl is-enabled $u 2>&1)"; done

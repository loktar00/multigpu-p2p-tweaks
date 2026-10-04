#!/bin/sh
# Install gpu-temp-guard. Does not start it: dry-run first, then enable.
set -e
[ "$(id -u)" = 0 ] || { echo "run as root"; exit 1; }
cd "$(dirname "$0")"
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found"; exit 1; }
command -v gputemps >/dev/null || echo "warning: gputemps not on PATH; set GUARD_GPUTEMPS in /etc/default/gpu-temp-guard"
install -m 0755 gpu-temp-guard.py /usr/local/sbin/gpu-temp-guard
install -m 0644 gpu-temp-guard.service /etc/systemd/system/gpu-temp-guard.service
install -m 0644 gpu-temp-guard.logrotate /etc/logrotate.d/gpu-temp-guard
[ -e /etc/default/gpu-temp-guard ] || install -m 0644 gpu-temp-guard.conf /etc/default/gpu-temp-guard
systemctl daemon-reload
/usr/local/sbin/gpu-temp-guard --selftest | tail -1
echo "next: edit /etc/default/gpu-temp-guard"
echo "      gpu-temp-guard --dry-run --verbose --polls 3"
echo "      systemctl enable --now gpu-temp-guard"

#!/bin/bash
# Sync code to the Pi, refresh the venv, and restart the service.
set -euo pipefail
HOST=${HOST:-pi@raspberrypi.local}
cd "$(dirname "$0")"
rsync -a --delete --exclude .venv --exclude __pycache__ --exclude '*.jpg' scanner deploy requirements.txt "$HOST":scanner/
ssh "$HOST" bash -s <<'REMOTE'
set -e
cd ~/scanner
[ -d .venv ] || python3 -m venv --system-site-packages .venv
.venv/bin/pip install -q -r requirements.txt
mkdir -p ~/.config/scanner
[ -f ~/.config/scanner/scanner.env ] || cp deploy/scanner.env.example ~/.config/scanner/scanner.env
sudo cp deploy/scanner.service /etc/systemd/system/scanner.service
sudo systemctl daemon-reload
if grep -q '^ARENA_TOKEN=.\+' ~/.config/scanner/scanner.env; then
  sudo systemctl enable --now scanner && sudo systemctl restart scanner && echo "service restarted"
else
  echo "config not filled in yet: ~/.config/scanner/scanner.env (service not started)"
fi
REMOTE

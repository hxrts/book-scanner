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
# word list and corpus for context-aware OCR correction
[ -f /usr/share/dict/words ] || sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq wamerican >/dev/null
mkdir -p data
[ -f data/big.txt ] || curl -sfL -o data/big.txt https://norvig.com/big.txt
mkdir -p ~/.config/scanner
[ -f ~/.config/scanner/scanner.env ] || cp deploy/scanner.env.example ~/.config/scanner/scanner.env
sudo cp deploy/scanner.service /etc/systemd/system/scanner.service
sudo install -m 755 deploy/scanner-cli /usr/local/bin/scanner
sudo install -m 644 deploy/99-scanner-led.rules /etc/udev/rules.d/ && sudo udevadm control --reload && sudo udevadm trigger --action=add --subsystem-match=leds
sudo systemctl daemon-reload
ENV=~/.config/scanner/scanner.env
if grep -q '^ARENA_TOKEN=.\+' $ENV && grep -q '^GOOGLE_SHEET_ID=.\+' $ENV; then
  sudo systemctl enable --now scanner && sudo systemctl restart scanner && echo "service restarted"
else
  sudo systemctl disable --now scanner 2>/dev/null || true
  echo "ARENA_TOKEN and GOOGLE_SHEET_ID needed in $ENV (service not started)"
fi
REMOTE

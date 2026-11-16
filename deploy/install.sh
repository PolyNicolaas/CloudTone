#!/usr/bin/env bash
# Install CloudTone on Raspberry Pi OS (Bookworm or later, 64-bit Lite recommended).
# Run from the repo root:  ./deploy/install.sh
set -euo pipefail

sudo apt-get update
# OpenCV and Picamera2 come from apt: prebuilt for the Pi and far faster to install than pip
sudo apt-get install -y python3-venv python3-opencv python3-picamera2 python3-numpy \
    libportaudio2 python3-yaml

# --system-site-packages lets the venv see apt's opencv/picamera2/numpy
python3 -m venv --system-site-packages .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[sonos,local]"

[ -f config.yaml ] || cp config.example.yaml config.yaml
echo
echo "Edit config.yaml (Sonos room, px_per_mm), then try:"
echo "  .venv/bin/cloudtone sonos-list"
echo "  .venv/bin/cloudtone -c config.yaml run --source sim     # no camera needed"
echo "  .venv/bin/cloudtone -c config.yaml run                  # the real chamber"
echo
echo "Start at boot:"
echo "  sudo cp deploy/cloudtone.service /etc/systemd/system/ && sudo systemctl enable --now cloudtone"

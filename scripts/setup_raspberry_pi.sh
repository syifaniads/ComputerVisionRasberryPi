#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y \
  python3-venv \
  python3-pip \
  python3-opencv \
  python3-picamera2

# --system-site-packages keeps access to apt-managed PiCamera2/OpenCV bindings.
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip

cat <<'EOF'
Base Raspberry Pi camera environment prepared.

Next:
  1. Install a tflite-runtime build compatible with your Raspberry Pi OS,
     Python version, and CPU architecture.
  2. Transfer the .tflite model and class_names.txt.
  3. Run runtime/raspberry_pi_camera.py.

The script intentionally does not hard-code an old ARM wheel URL because
TensorFlow Lite runtime availability changes across OS/Python versions.
EOF

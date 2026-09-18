#!/usr/bin/env bash
# One-time setup for medical-appointment on Linux (x64): Python venv + packages,
# llama.cpp (llama-server) and the Gemma 4 E4B model. Everything it downloads
# is gitignored. Safe to re-run: existing downloads are skipped.
#
# From the repo root:
#   ./scripts/setup-medical.sh cuda     # NVIDIA GPU (CUDA 12.8 build)
#   ./scripts/setup-medical.sh vulkan   # other GPUs
#   ./scripts/setup-medical.sh cpu      # no GPU
#
# Needs: python3.12 with venv, curl, tar. No system ffmpeg: MP3s are decoded
# by the PyAV wheels that faster-whisper installs.
# Training audio: copy medical-appointment/data/audio from the organisers'
# repo (https://github.com/amboltio/Nordic-AI-Cup-2026) — used by
# local_evaluator.py and the server's warm-up.
set -euo pipefail

BACKEND="${1:-cuda}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$ROOT/medical-appointment"
MODELS="$APP/models"
mkdir -p "$MODELS"

RELEASE=b11029   # pinned: the version everything was tested with
BASE="https://github.com/ggml-org/llama.cpp/releases/download/$RELEASE"
GGUF=gemma-4-E4B-it-Q4_K_M.gguf
GGUF_URL="https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/$GGUF"

# --- Python ---------------------------------------------------------------
if [ ! -d "$APP/.venv" ]; then
  echo "Creating venv (Python 3.12)..."
  python3.12 -m venv "$APP/.venv"
fi
PY="$APP/.venv/bin/python"
if [ "$BACKEND" = cuda ]; then
  "$PY" -m pip uninstall -y onnxruntime >/dev/null 2>&1 || true
  REQ=requirements-gpu.txt
else
  REQ=requirements.txt
fi
echo "Installing $REQ ..."
"$PY" -m pip install -q -r "$APP/$REQ"

# --- llama.cpp ------------------------------------------------------------
case "$BACKEND" in
  cuda)   DIR="$MODELS/llama-cpp-cuda"
          TARS="llama-$RELEASE-bin-ubuntu-cuda-12.8-x64.tar.gz cudart-llama-$RELEASE-bin-ubuntu-cuda-12.8-x64.tar.gz" ;;
  vulkan) DIR="$MODELS/llama-cpp-vulkan"; TARS="llama-$RELEASE-bin-ubuntu-vulkan-x64.tar.gz" ;;
  cpu)    DIR="$MODELS/llama-cpp";        TARS="llama-$RELEASE-bin-ubuntu-x64.tar.gz" ;;
  *) echo "backend must be cuda, vulkan or cpu"; exit 1 ;;
esac
if [ -z "$(find "$DIR" -name llama-server -type f 2>/dev/null | head -1)" ]; then
  mkdir -p "$DIR"
  set -- $TARS
  echo "Downloading $1 ..."
  curl -fL "$BASE/$1" | tar -xz -C "$DIR"
  BIN_DIR="$(dirname "$(find "$DIR" -name llama-server -type f | head -1)")"
  chmod +x "$BIN_DIR/llama-server"
  if [ $# -gt 1 ]; then
    # CUDA runtime libs must sit next to the binary, whatever the archive layout
    echo "Downloading $2 ..."
    TMP="$(mktemp -d)"
    curl -fL "$BASE/$2" | tar -xz -C "$TMP"
    find "$TMP" -name '*.so*' -exec cp -P {} "$BIN_DIR/" \;
    rm -rf "$TMP"
  fi
else
  echo "llama.cpp already in $DIR"
fi

# --- Model ----------------------------------------------------------------
if [ ! -f "$MODELS/$GGUF" ]; then
  echo "Downloading $GGUF (~5 GB) ..."
  curl -fL -o "$MODELS/$GGUF" "$GGUF_URL"
else
  echo "$GGUF already present"
fi

echo
echo "Done. To run:"
echo "  cd medical-appointment"
[ "$BACKEND" = cuda ] && echo "  export ASR_PROVIDER=cuda     # Parakeet on the GPU too"
echo "  .venv/bin/python api.py"
echo "  # second terminal: .venv/bin/python local_evaluator.py"

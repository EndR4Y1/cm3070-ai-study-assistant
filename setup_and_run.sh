#!/bin/bash
# ============================================================
#  AI Study Assistant V4 — One-click setup & launch
#  Run this once:  bash setup_and_run.sh
# ============================================================

set -e  # stop on first error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "====================================================="
echo "  AI Study Assistant V4 — Setup & Launch"
echo "====================================================="
echo ""

# ----- 1. Check Python -----
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install it from https://python.org"
    exit 1
fi
PY=$(python3 --version)
echo "✓ Python found: $PY"

# ----- 2. Check / install ffmpeg -----
if ! command -v ffmpeg &>/dev/null; then
    echo ""
    echo "→ ffmpeg not found. Installing via Homebrew..."
    if ! command -v brew &>/dev/null; then
        echo "ERROR: Homebrew not found. Install from https://brew.sh, then re-run this script."
        exit 1
    fi
    brew install ffmpeg
fi
echo "✓ ffmpeg found"

# ----- 3. Create fresh venv (ignores any conda base) -----
echo ""
echo "→ Creating virtual environment..."
python3 -m venv venv --clear
echo "✓ venv created"

# ----- 4. Use venv pip directly (avoids conda conflicts) -----
VENV_PIP="$SCRIPT_DIR/venv/bin/pip"
VENV_PY="$SCRIPT_DIR/venv/bin/python"

echo ""
echo "→ Upgrading pip..."
"$VENV_PIP" install --upgrade pip --quiet

echo ""
echo "→ Installing dependencies (this may take 3-5 minutes on first run)..."
"$VENV_PIP" install \
    "openai-whisper" \
    "transformers>=4.36" \
    "torch" \
    "scikit-learn" \
    "jiwer" \
    "rouge-score" \
    "numpy" \
    "pandas" \
    "gradio>=4.0,<6.0" \
    --quiet

echo "✓ All packages installed"

# ----- 5. Launch -----
echo ""
echo "====================================================="
echo "  Launching AI Study Assistant at http://127.0.0.1:7860"
echo "  Press Ctrl+C to stop"
echo "====================================================="
echo ""

"$VENV_PY" app.py

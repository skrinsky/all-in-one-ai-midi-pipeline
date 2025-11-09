#!/usr/bin/env bash
set -euo pipefail

echo "==> Creating uv environment with Python 3.10..."
uv venv --python 3.10 .venv

echo "==> Activating environment..."
source .venv/bin/activate

echo "==> Installing requirements..."
uv pip install -r requirements.txt

echo ""
echo "==> Setup complete!"
echo ""
echo "To activate this environment in the future, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To run the pipeline:"
echo "  python pipeline.py run-batch \"data/raw/*.wav\""

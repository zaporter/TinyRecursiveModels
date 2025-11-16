#!/usr/bin/env bash
uv venv
source .venv/bin/activate
uv pip install --upgrade pip wheel setuptools
uv pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129
uv pip install -r requirements.txt # install requirements
uv pip install --no-cache-dir --no-build-isolation adam-atan2

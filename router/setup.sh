#!/usr/bin/env bash
# One-time: create an isolated venv for the router (avoids the broken conda env's pydantic/markupsafe).
set -euo pipefail
cd "$(dirname "$0")"
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python starlette 'uvicorn[standard]' httpx
echo "router venv ready: $(pwd)/.venv"

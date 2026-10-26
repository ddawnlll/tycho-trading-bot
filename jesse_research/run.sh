#!/usr/bin/env bash
# Run a research script inside the pinned Jesse image, isolated from host secrets:
# only this jesse_research/ directory is mounted (code read-only), no host env vars or home dir,
# non-root user, 1.5 GB memory cap. NET=none (default) blocks network; NET=bridge only for fetch_data.py.
# Usage: ./run.sh research_scripts/<script>.py [args]     e.g. NET=bridge ./run.sh research_scripts/fetch_data.py /data
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data results
exec docker run --rm --network "${NET:-none}" --memory 1500m --memory-swap 1500m \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -w /tmp \
  -e GIT_HEAD="$(git rev-parse HEAD)$(git diff --quiet HEAD -- . || echo +dirty)" \
  -v "$PWD/project:/work:ro" -v "$PWD:/repo:ro" -v "$PWD/trials.jsonl:/ledger/trials.jsonl" \
  -v "$PWD/data:/data" -v "$PWD/results:/results" \
  salehmir/jesse:3.5.3 python -I "/work/$1" "${@:2}"

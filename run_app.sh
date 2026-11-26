#!/usr/bin/env bash
# Jalankan aplikasi Streamlit (lokal). Untuk akses dari komputer lain pakai SSH tunnel.
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8501}"
HOST="${HOST:-127.0.0.1}"
exec /opt/data/research/sandbox/.venv/bin/streamlit run app.py \
  --server.address "$HOST" --server.port "$PORT"

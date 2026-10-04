#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo 'Chưa có môi trường .venv. Xem hướng dẫn cài đặt trong README.md.' >&2
  exit 1
fi
exec .venv/bin/python app.py "$@"

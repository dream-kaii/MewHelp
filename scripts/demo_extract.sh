#!/usr/bin/env bash
# 验收 3:售后描述 -> 结构化 JSON。payload 走 UTF-8 fixture 文件。
set -euo pipefail
BASE="${1:-http://127.0.0.1:8000}"
FIX="$(cd "$(dirname "$0")" && pwd)/fixtures"

echo "== 验收3:售后描述 -> 结构化 JSON =="
curl -s -X POST "$BASE/api/extract" \
  -H 'Content-Type: application/json' \
  --data-binary "@$FIX/extract1.json"
echo

#!/usr/bin/env bash
# 验收:工具调用(物流 / FAQ / 漏召回)
# 中文 payload 走 UTF-8 fixture 文件 + --data-binary,避开 Windows 控制台 GBK 编码坑。
set -uo pipefail
BASE="${1:-http://127.0.0.1:8000}"
FIX="$(cd "$(dirname "$0")" && pwd)/fixtures"
for f in tool_logistics tool_faq tool_miss; do
  echo "== $f =="
  curl -sN -X POST "$BASE/api/chat" -H 'Content-Type: application/json' --data-binary "@$FIX/$f.json"
  echo
done

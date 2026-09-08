#!/usr/bin/env bash
# 验收 1&2:SSE 流式 + 同 session 第二轮接上下文。
# 固定 session_id 两连发,避免依赖响应解析;中文 payload 走 UTF-8 fixture 文件 + --data-binary,
# 避免 Windows 控制台 GBK 编码破坏 JSON。
set -uo pipefail
BASE="${1:-http://127.0.0.1:8000}"
FIX="$(cd "$(dirname "$0")" && pwd)/fixtures"

echo "== 验收1:流式回复(首轮为新会话,应含 event: session,逐 delta 输出,结尾 event: done)=="
curl -sN -X POST "$BASE/api/chat" \
  -H 'Content-Type: application/json' \
  --data-binary "@$FIX/chat1.json"
echo
echo
echo "== 验收2:同一 session(demo-acceptance)第二轮,应能接住第一轮上下文、复述订单号 =="
curl -sN -X POST "$BASE/api/chat" \
  -H 'Content-Type: application/json' \
  --data-binary "@$FIX/chat2.json"
echo
exit 0

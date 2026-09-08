#!/usr/bin/env bash
# 验收 1&2:SSE 流式 + 同 session 第二轮接上下文
set -euo pipefail
BASE="${1:-http://127.0.0.1:8000}"

echo "== 验收1:流式回复(应逐 token 出现)=="
curl -sN -X POST "$BASE/api/chat" \
  -H 'Content-Type: application/json' \
  -d '{"message":"你好,我的订单一直没发货,能帮我看看吗?订单号20260901001"}' | tee /tmp/chat1.sse

SID=$(grep -o '"session_id":"[0-9a-f]*"' /tmp/chat1.sse | head -1 | sed 's/.*:"//;s/"//')
if [ -z "$SID" ]; then
  echo
  echo "!! 未在首轮响应中解析到 session_id(可能首轮用了已存在会话或格式变化)" >&2
fi
echo
echo "== 验收2:同一 session 第二轮(应接住第一轮上下文,session_id=$SID)=="
curl -sN -X POST "$BASE/api/chat" \
  -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"$SID\",\"message\":\"那能顺便告诉我大概多久能到吗?\"}"
echo

#!/usr/bin/env bash
# 验收 1&2:SSE 流式 + 同一会话第二轮接上下文。
# 中文 payload 走 UTF-8 fixture 文件 + --data-binary,避免 Windows 控制台 GBK 编码破坏 JSON。
# 第二轮复用服务端在 session 帧回传的 conversation_id(见 README「SSE 契约」)。
set -uo pipefail
BASE="${1:-http://127.0.0.1:8000}"
FIX="$(cd "$(dirname "$0")" && pwd)/fixtures"

echo "== 验收1:流式回复(首轮为新会话,应含 event: session,逐 delta 输出,结尾 event: done)=="
R1=$(curl -sN -X POST "$BASE/api/chat" \
  -H 'Content-Type: application/json' \
  --data-binary "@$FIX/chat1.json")
printf '%s\n' "$R1"
CID=$(printf '%s\n' "$R1" | sed -n 's/.*"conversation_id": *\([0-9][0-9]*\).*/\1/p' | head -n 1)
echo
if [ -z "$CID" ]; then
  echo "!! 未从 session 帧解析到 conversation_id,跳过验收2"
  exit 1
fi
echo
echo "== 验收2:复用同一会话(conversation_id=$CID)第二轮,应能接住第一轮上下文、复述订单号 =="
TMP="$(mktemp)"
sed "s/\"conversation_id\": *0/\"conversation_id\": $CID/" "$FIX/chat2.json" > "$TMP"
curl -sN -X POST "$BASE/api/chat" \
  -H 'Content-Type: application/json' \
  --data-binary "@$TMP"
rm -f "$TMP"
echo
exit 0

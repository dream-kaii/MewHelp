#!/usr/bin/env bash
# ch03 验收:聊天页 → query_faq → 向量召回
#   bash scripts/demo_rag.sh [BASE]
# 复用 ch02 的 UTF-8 fixture 模式:中文 payload 走文件 + --data-binary,
# 避开 Windows 控制台 GBK 编码坑。
#   rag_hit.json  : 知识库内问题(运费说明),应看到 query_faq 命中并据召回作答
#   rag_miss.json : 知识库外问题(营业时间),预期模型礼貌兜底、**不调用 query_faq**,
#                   自然不会给出「未找到」(README ch03 §验收 ②)
set -uo pipefail
BASE="${1:-http://127.0.0.1:8000}"
FIX="$(cd "$(dirname "$0")" && pwd)/fixtures"
for f in rag_hit rag_miss; do
  echo "== $f =="
  curl -sN -X POST "$BASE/api/chat" -H 'Content-Type: application/json' --data-binary "@$FIX/$f.json"
  echo
done

#!/usr/bin/env bash
# 验收 3:售后描述 -> 结构化 JSON
set -euo pipefail
BASE="${1:-http://127.0.0.1:8000}"

echo "== 验收3:售后描述 -> 结构化 JSON =="
curl -s -X POST "$BASE/api/extract" \
  -H 'Content-Type: application/json' \
  -d '{"text":"你好,我在你家买的猫粮订单20260901001拆开漏气了,想退货退款,能上门取件吗"}'
echo

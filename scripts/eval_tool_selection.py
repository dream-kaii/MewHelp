# scripts/eval_tool_selection.py
"""跑真模型验证工具选型。用法:python -m scripts.eval_tool_selection"""
import json
import sys
from pathlib import Path

from app.llm import get_chat_model
from app.tools.business import query_logistics, query_order, query_product
from app.tools.kb import make_kb_tools
from app.tools.ops import make_ops_tools
from app.tools.registry import ToolRegistry
from app.db.base import get_sessionmaker

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "eval_data" / "tool_selection_samples.json"


async def main() -> int:
    sf = get_sessionmaker()
    registry = ToolRegistry([
        query_order, query_product, query_logistics,
        *make_kb_tools(sf), *make_ops_tools(sf, conversation_id=0),
    ])
    model = get_chat_model().bind_tools(registry.bindable())
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    hits = 0
    non_gap_hits = 0
    non_gap_total = 0
    for i, s in enumerate(samples, 1):
        msg = await model.ainvoke([{"role": "user", "content": s["question"]}])
        calls = getattr(msg, "tool_calls", None) or []
        picked = calls[0]["name"] if calls else "none"
        ok = picked == s["expected_tool"]
        hits += ok
        tag = "PASS" if ok else ("GAP" if s.get("known_gap") else "FAIL")
        print(f"[{tag}] #{i} {s['question']} -> 期望 {s['expected_tool']}, 实际 {picked}")
        if s.get("known_gap"):
            print(f"      已知缺口: {s['known_gap']}")
        else:
            # known_gap 样例的命中/漏召回都不参与退出码判定
            non_gap_total += 1
            non_gap_hits += ok
    rate = hits / len(samples)
    print(f"\n选型准确率: {hits}/{len(samples)} = {rate:.0%}")
    return 0 if non_gap_hits == non_gap_total else 1


if __name__ == "__main__":
    import anyio

    sys.exit(anyio.run(main))

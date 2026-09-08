"""跑真实模型验证售后抽取,直至 100% 通过。用法:python -m scripts.eval_extract"""
import json
import sys
from pathlib import Path

from app.config import get_settings
from app.extract import ExtractService
from app.llm import get_chat_model

SAMPLES = Path(__file__).resolve().parents[1] / "eval_data" / "after_sales_samples.json"
KEYWORDS_FIELD = "desired_solution_keywords"


def _norm(v):
    return (v or "").strip()


def check_one(actual: dict, expected: dict) -> list[str]:
    """返回未命中的字段名列表(空 = 全命中)。"""
    misses = []
    for field in ("order_no", "request_type"):
        exp = _norm(expected.get(field)).lower()
        got = _norm(actual.get(field)).lower()
        if exp == "null":
            if got:
                misses.append(field)
        elif got != exp:
            misses.append(field)
    kw = expected.get(KEYWORDS_FIELD) or []
    sol = _norm(actual.get("desired_solution"))
    if kw:
        if not any(k in sol for k in kw):
            misses.append("desired_solution")
    elif sol:
        # 期望该字段留空,模型却填了内容
        misses.append("desired_solution")
    return misses


async def main() -> int:
    settings = get_settings()
    if not settings.llm_api_key and settings.llm_provider not in ("ollama",):
        print("请先在 .env 填好 LLM_API_KEY 再跑 eval", file=sys.stderr)
        return 2

    service = ExtractService(get_chat_model())
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    passed = 0
    for i, s in enumerate(samples, 1):
        try:
            result = await service.extract(s["text"])
            data = result.model_dump()
        except Exception as exc:  # noqa: BLE001 —— 单条失败不应中断整轮,便于看清全量并迭代 prompt
            print(f"[FAIL] #{i} 原文:{s['text'][:30]}")
            print(f"    error: {type(exc).__name__}: {exc}")
            continue
        misses = check_one(data, s["expected"])
        ok = not misses
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] #{i} 原文:{s['text'][:30]}")
        print(f"    got  : order_no={data['order_no']} request_type={data['request_type']} desired_solution={data['desired_solution']}")
        print(f"    exp  : order_no={s['expected'].get('order_no')} request_type={s['expected'].get('request_type')} kw={s['expected'].get(KEYWORDS_FIELD)}")
        if misses:
            print(f"    miss : {misses}")
    rate = passed / len(samples)
    print(f"\n通过率: {passed}/{len(samples)} = {rate:.0%}")
    return 0 if rate == 1.0 else 1


if __name__ == "__main__":
    import anyio

    sys.exit(anyio.run(main))

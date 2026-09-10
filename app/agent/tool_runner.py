"""单轮工具往返:第一阶段流式收集 tool_calls,第二阶段执行并回灌。"""
from typing import Callable

from app.tools.registry import ToolExecutionResult, ToolRegistry


def _text_of(chunk) -> str:
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


async def stream_first_round(model_with_tools, messages, on_delta: Callable[[str], None]) -> tuple[list[dict], str]:
    """跑绑定工具的第一段:逐 chunk 回调文本,累加得到 tool_calls(合并自 tool_call_chunks)。

    模型返回的工具调用 JSON 无法解析时,LangChain 把它们放进 invalid_tool_calls 而 tool_calls 为空;
    若不处理,本轮会静默无输出。这里主动抛错,由调用方(ChatService)转成 error 帧。
    """
    acc = None
    parts: list[str] = []
    async for chunk in model_with_tools.astream(messages):
        acc = chunk if acc is None else acc + chunk
        text = _text_of(chunk)
        if text:
            parts.append(text)
            on_delta(text)
    calls = getattr(acc, "tool_calls", None) or []
    invalid = getattr(acc, "invalid_tool_calls", None) or []
    if not calls and invalid:
        detail = "; ".join(f"{c.get('name') or '?'}: {c.get('args')}" for c in invalid)
        raise ValueError(f"模型返回的工具调用 JSON 无法解析:{detail}")
    normalized = [{"id": c.get("id") or "", "name": c.get("name") or "", "args": c.get("args") or {}} for c in calls]
    return normalized, "".join(parts)


async def execute_tool_calls(
    registry: ToolRegistry,
    tool_calls: list[dict],
    on_status: Callable[[str, str, str, dict, str], None],
    *,
    timeout: float,
    retries: int,
) -> list[ToolExecutionResult]:
    """并行执行该轮全部工具调用;每次调用前后各回调一次状态。"""
    import asyncio

    async def run_one(call: dict) -> ToolExecutionResult:
        call_id, name, args = call["id"], call["name"], call["args"]
        on_status("running", call_id, name, args, "")
        result = await registry.execute(name, args, timeout=timeout, retries=retries, call_id=call_id)
        summary = result.content if result.ok else (result.error or "")
        on_status("ok" if result.ok else "error", call_id, name, args, summary)
        return result

    if not tool_calls:
        return []
    return list(await asyncio.gather(*(run_one(c) for c in tool_calls)))

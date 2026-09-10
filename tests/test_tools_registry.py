import asyncio

import pytest
from langchain_core.tools import tool

from app.tools.registry import ToolExecutionResult, ToolRegistry

pytestmark = pytest.mark.anyio


@tool
def echo(text: str) -> str:
    """回显。"""
    return "echo:" + text


@tool
def boom() -> str:
    """总是失败。"""
    raise RuntimeError("kaboom")


async def test_execute_ok_and_schema_validation_error():
    reg = ToolRegistry([echo])
    r = await reg.execute("echo", {"text": "hi"}, timeout=2, retries=0)
    assert r.ok and r.content == "echo:hi" and r.attempts == 1
    bad = await reg.execute("echo", {"wrong": 1}, timeout=2, retries=0)
    assert not bad.ok and bad.attempts == 1  # 参数校验失败不重试


async def test_unknown_tool_is_error_without_retry():
    reg = ToolRegistry([echo])
    r = await reg.execute("nope", {}, timeout=2, retries=2)
    assert not r.ok and "未知工具" in (r.error or "") and r.attempts == 1


async def test_retry_then_fail_captures_error():
    reg = ToolRegistry([boom])
    r = await reg.execute("boom", {}, timeout=2, retries=2)
    assert not r.ok and r.attempts == 3 and "kaboom" in (r.error or "")


def _make_counting_slow_tool():
    """本地定义一个真正异步的慢工具。

    说明:原 brief 设想用 `reg.by_name("echo").ainvoke = slow` 替换方法,但
    StructuredTool 是 pydantic v2 模型(extra != "allow"),实例上无法赋值类方法,
    会抛 `ValueError: "StructuredTool" object has no field "ainvoke"`。
    故改用真正的慢工具,仍以计数断言重试次数。
    """
    calls = {"n": 0}

    @tool
    async def slow(text: str) -> str:
        """总是很慢(测试用)。"""
        calls["n"] += 1
        await asyncio.sleep(0.5)
        return "late:" + text

    return slow, calls


async def test_timeout_is_enforced_and_retried():
    slow, calls = _make_counting_slow_tool()
    reg = ToolRegistry([slow])
    r = await reg.execute("slow", {"text": "x"}, timeout=0.05, retries=1)
    assert not r.ok and calls["n"] == 2 and "超时" in (r.error or "")


def test_bindable_and_names():
    reg = ToolRegistry([echo, boom])
    assert set(reg.names()) == {"echo", "boom"}
    assert reg.bindable() == [echo, boom]
    assert isinstance(ToolExecutionResult("c", "echo", {}, True, "x", None, 1), ToolExecutionResult)

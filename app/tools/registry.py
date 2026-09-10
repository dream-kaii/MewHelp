"""工具注册表:登记、按名派发、参数校验、超时与重试。"""
import asyncio
import logging
from dataclasses import dataclass, field

from langchain_core.tools import BaseTool
from pydantic import ValidationError

logger = logging.getLogger("mewhelp.tools")


@dataclass
class ToolExecutionResult:
    call_id: str
    name: str
    args: dict
    ok: bool
    content: str
    error: str | None = None
    attempts: int = 1


class ToolRegistry:
    def __init__(self, tools: list[BaseTool]):
        self._tools = {t.name: t for t in tools}
        self._order = list(tools)

    def names(self) -> list[str]:
        return [t.name for t in self._order]

    def bindable(self) -> list[BaseTool]:
        return list(self._order)

    def by_name(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    async def execute(
        self, name: str, args: dict, *, timeout: float, retries: int, call_id: str = ""
    ) -> ToolExecutionResult:
        tool = self.by_name(name)
        if tool is None:
            return ToolExecutionResult(call_id, name, args, False, "", f"未知工具:{name}", 1)

        attempt = 0
        last_err = ""
        while attempt <= retries:
            attempt += 1
            try:
                out = await asyncio.wait_for(tool.ainvoke(args), timeout=timeout)
                text = out if isinstance(out, str) else str(out)
                return ToolExecutionResult(call_id, name, args, True, text, None, attempt)
            except ValidationError as exc:
                # 参数不合法,重试无意义
                return ToolExecutionResult(call_id, name, args, False, "", f"参数校验失败:{exc}", attempt)
            except asyncio.TimeoutError:
                last_err = f"工具执行超时(>{timeout}s)"
                logger.warning("tool %s timeout attempt=%s", name, attempt)
            except Exception as exc:  # noqa: BLE001 —— 工具失败不应打断整轮
                last_err = f"{type(exc).__name__}: {exc}"
                logger.warning("tool %s failed attempt=%s: %s", name, attempt, last_err)
            if attempt <= retries:
                await asyncio.sleep(0.2 * attempt)
        return ToolExecutionResult(call_id, name, args, False, "", last_err, attempt)

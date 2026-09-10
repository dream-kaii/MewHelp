"""历史裁剪 + token 预算。纯函数,不依赖 langchain / 网络,历史一律为 list[dict]。

`to_langchain_messages` 是消息行 → LangChain 对象的唯一转换处(惰性导入 langchain_core)。
"""

import math

_MESSAGE_OVERHEAD = 4  # 每条消息的固定估算开销(角色/格式)


def estimate_tokens(text: str, chars_per_token: float = 2.0) -> int:
    """启发式估算 token 数。默认约 2 个字符 = 1 token(中文为主),向上取整。"""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / chars_per_token))


def total_tokens(messages: list[dict]) -> int:
    return sum(_tokens_of_message(m) for m in messages)


def _tokens_of_message(msg: dict) -> int:
    return estimate_tokens(str(msg.get("content") or "")) + _MESSAGE_OVERHEAD


def split_turn_groups(body: list[dict]) -> list[list[dict]]:
    """把消息体按「轮组」切分,每组以 user 开头、含其后的 assistant/tool 直到下一个 user。

    轮组 = 单独一条 user(如失败轮只落了 user),或
    `user, assistant(tool_calls), tool..., assistant` 的完整工具往返。
    user 之前的孤立行(理论不该出现)丢弃,保证每组首条必为 user。
    """
    groups: list[list[dict]] = []
    for m in body:
        if m.get("role") == "user":
            groups.append([m])
        elif groups:
            groups[-1].append(m)
        # else: 孤立行(首个 user 之前)→ 丢弃,它不能作为窗口开头
    return groups


def trim_to_budget(messages: list[dict], budget: int) -> list[dict]:
    """超预算时按「轮组」从最旧端整组删除,绝不劈开一轮。

    轮组见 `split_turn_groups`。只删整组保证:窗口紧跟 system 后必以 user 开头,
    且不残留孤立的 tool / assistant(tool_calls)(OpenAI 协议上游会拒收
    "messages with role 'tool' must be a response to a preceding message with 'tool_calls'")。
    不变量:首条 system(若有)与最后一组(当前轮)永不删除;仅剩 system+当前轮
    仍超预算则原样返回该最小窗口。(无 user 行的退化输入只保留 system。)
    """
    if not messages:
        return messages

    head = [messages[0]] if messages[0].get("role") == "system" else []
    groups = split_turn_groups(messages[len(head):])

    while len(groups) > 1 and total_tokens(head + [m for g in groups for m in g]) > budget:
        groups.pop(0)  # 删最旧一整组
    return head + [m for g in groups for m in g]


def build_messages(system_prompt: str | None, history: list[dict], current_user_msg: str, budget: int) -> list[dict]:
    full: list[dict] = []
    if system_prompt:
        full.append({"role": "system", "content": system_prompt})
    full.extend(history)
    full.append({"role": "user", "content": current_user_msg})
    return trim_to_budget(full, budget)


def to_langchain_messages(messages: list[dict]) -> list:
    """dict 行 → LangChain 消息。支持 assistant 带 tool_calls 与 role=tool 的 ToolMessage。"""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    out = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            out.append(SystemMessage(m.get("content") or ""))
        elif role == "assistant":
            tcs = m.get("tool_calls") or []
            out.append(AIMessage(content=m.get("content") or "", tool_calls=list(tcs)))
        elif role == "tool":
            out.append(ToolMessage(m.get("content") or "", tool_call_id=m.get("tool_call_id") or ""))
        else:
            out.append(HumanMessage(m.get("content") or ""))
    return out

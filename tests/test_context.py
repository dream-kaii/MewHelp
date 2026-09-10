from app.context import build_messages, estimate_tokens, total_tokens, trim_to_budget

SYS = {"role": "system", "content": "你是客服"}
U = lambda i: {"role": "user", "content": f"用户问题{i},内容补长一些"}
A = lambda i: {"role": "assistant", "content": f"客服回复{i},同样补长一些"}


def _tool_group(tag=""):
    """一段工具轮组:user, assistant(tool_calls), tool, assistant。"""
    return [
        {"role": "user", "content": f"订单1001到哪了{tag}"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "query_order", "args": {"order_id": "1001"}}]},
        {"role": "tool", "content": "已发货", "tool_call_id": "c1"},
        {"role": "assistant", "content": f"已发货,请留意查收{tag}"},
    ]


def test_estimate_tokens_empty_is_zero():
    assert estimate_tokens("") == 0


def test_estimate_tokens_scales_with_length():
    assert estimate_tokens("你好") >= 1
    assert estimate_tokens("你好" * 100) > estimate_tokens("你好")


def test_estimate_tokens_rounds_up():
    assert estimate_tokens("abc") == 2  # 3 字符 / 2 → 向上取整
    assert estimate_tokens("a") == 1


def test_trim_keeps_system_and_current_message():
    history = [U(0), A(0), U(1), A(1), U(2), A(2)]
    out = trim_to_budget([SYS] + history + [U(3)], budget=50)
    assert out[0]["role"] == "system"
    assert out[-1] == U(3)


def test_trim_drops_oldest_pair_first():
    history = [U(0), A(0), U(1), A(1)]
    # 预算 45:整窗 6+10*5=56 超限;删最旧 U0/A0 后剩 36<=45 停
    out = trim_to_budget([SYS] + history + [U(2)], budget=45)
    roles = [m["role"] for m in out]
    assert roles[0] == "system"
    assert U(1) in out and A(1) in out and U(2) in out
    assert U(0) not in out and A(0) not in out


def test_trim_returns_everything_when_within_budget():
    msgs = [SYS, U(0), A(0), U(1)]
    assert trim_to_budget(msgs, budget=10_000) == msgs


def test_build_messages_order_system_history_current():
    history = [U(0), A(0)]
    out = build_messages("你是客服", history, "当前问题", budget=10_000)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "user"]
    assert out[-1]["content"] == "当前问题"


def test_trim_tool_window_starts_with_user_never_orphans_tool():
    """带工具的历史被裁剪后,窗口必须以 user 开头、不留孤儿 tool/assistant(tool_calls)。

    预算刻取「旧算法恰好停在 body[2:]」这一步 —— 旧实现会丢掉最旧的 user+assistant(tool_calls),
    使窗口以 tool 行开头,OpenAI 协议上游会拒收。
    """
    group = _tool_group()
    current = U(9)
    # 旧算法的坏窗口:丢弃最旧两条后剩下的 body[2:] + 当前轮
    bad_window = [SYS, *group[2:], current]
    budget = total_tokens(bad_window)

    out = trim_to_budget([SYS, *group, current], budget=budget)
    roles = [m["role"] for m in out]
    assert roles[0] == "system"
    assert roles[1] == "user"  # 旧实现此处为 "tool"
    assert "tool" not in roles  # 整组被删,不留孤儿
    assert out[-1] == current  # 当前轮永不删除


def test_trim_keeps_whole_tool_group_when_it_fits():
    group = _tool_group()
    full = [SYS, *group, U(9)]
    assert trim_to_budget(full, budget=total_tokens(full)) == full


def test_trim_drops_whole_tool_group_not_part_of_it():
    group = _tool_group()
    tail = [SYS, *group, U(9)]
    out = trim_to_budget([SYS, U(0), A(0), *group, U(9)], budget=total_tokens(tail))
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool", "assistant", "user"]


def test_trim_never_drops_last_message_on_even_length_body():
    """失败轮只落 user → 历史奇偶翻转成偶数长度;旧实现会整窗删空,只剩 system。"""
    msgs = [SYS, U(0), A(0), U(1), U(2)]
    out = trim_to_budget(msgs, budget=1)
    assert out[0]["role"] == "system"
    assert out[-1] == U(2)
    assert len(out) == 2

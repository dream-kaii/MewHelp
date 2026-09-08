from app.context import build_messages, estimate_tokens, trim_to_budget

SYS = {"role": "system", "content": "你是客服"}
U = lambda i: {"role": "user", "content": f"用户问题{i},内容补长一些"}
A = lambda i: {"role": "assistant", "content": f"客服回复{i},同样补长一些"}


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

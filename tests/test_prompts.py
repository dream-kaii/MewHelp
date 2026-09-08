from app.config import Settings
from app.prompts import EXTRACT_SYSTEM, format_chat_system_prompt, format_extract_user


def test_chat_system_prompt_renders_and_constrains():
    text = format_chat_system_prompt(Settings(cs_shop_name="测试店", cs_staff_name="阿喵"))
    assert "测试店" in text and "阿喵" in text
    for key in ("客服", "不编造", "订单号"):
        assert key in text


def test_extract_user_prompt_embeds_text():
    out = format_extract_user("订单20260908001漏气想退")
    assert "20260908001" in out


def test_extract_system_lists_enum_and_null_rule():
    assert "退货退款" in EXTRACT_SYSTEM and "仅退款" in EXTRACT_SYSTEM
    assert "换货" in EXTRACT_SYSTEM and "维修" in EXTRACT_SYSTEM
    assert "null" in EXTRACT_SYSTEM

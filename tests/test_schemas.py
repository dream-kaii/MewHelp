import pytest
from pydantic import ValidationError

from app.schemas import AfterSalesExtract, ChatRequest, ExtractRequest, RequestType


def test_chat_request_defaults_session_none():
    r = ChatRequest(message="你好")
    assert r.session_id is None
    assert r.message == "你好"


def test_chat_request_rejects_empty_message():
    with pytest.raises(ValidationError):
        ChatRequest(message="")


def test_request_type_has_exactly_four_members():
    assert {e.value for e in RequestType} == {"退货退款", "仅退款", "换货", "维修"}


def test_extract_allows_null_fields():
    e = AfterSalesExtract()
    assert e.order_no is None and e.request_type is None and e.desired_solution is None


def test_extract_accepts_valid_request_type():
    e = AfterSalesExtract(order_no="20260908001", request_type=RequestType.RETURN_REFUND)
    assert e.request_type == RequestType.RETURN_REFUND
    assert e.order_no == "20260908001"


def test_extract_coerces_literal_null_strings_to_none():
    # 真实模型有时把 null 输出成字符串 "null"/"none",应宽容归一为 None 而非炸枚举/类型
    e = AfterSalesExtract.model_validate(
        {"order_no": "null", "request_type": "none", "desired_solution": ""}
    )
    assert e.order_no is None and e.request_type is None and e.desired_solution is None


def test_extract_valid_enum_not_clobbered_by_coercion():
    e = AfterSalesExtract.model_validate(
        {"order_no": "20260908001", "request_type": "退货退款", "desired_solution": "上门取件"}
    )
    assert e.request_type == RequestType.RETURN_REFUND and e.desired_solution == "上门取件"


def test_chat_request_rejects_whitespace_only_message():
    with pytest.raises(ValidationError):
        ChatRequest(message="   ")


def test_chat_request_rejects_oversized_session_id():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="a" * 129, message="hi")


def test_extract_request_rejects_whitespace_only_text():
    with pytest.raises(ValidationError):
        ExtractRequest(text="\n\t ")

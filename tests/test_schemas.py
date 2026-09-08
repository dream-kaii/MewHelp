import pytest
from pydantic import ValidationError

from app.schemas import AfterSalesExtract, ChatRequest, RequestType


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

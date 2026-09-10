from enum import Enum

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    session_id: str | None = Field(default=None, max_length=128, description="(ch01 遗留,忽略)")
    conversation_id: int | None = Field(default=None, description="会话 id;缺省则新建并回传")
    user_id: str = Field(default="web-anonymous", max_length=64)
    message: str = Field(min_length=1, max_length=4000)

    @field_validator("message")
    @classmethod
    def _message_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message 不能为空白")
        return v


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text 不能为空白")
        return v


class RequestType(str, Enum):
    RETURN_REFUND = "退货退款"
    REFUND_ONLY = "仅退款"
    EXCHANGE = "换货"
    REPAIR = "维修"


class AfterSalesExtract(BaseModel):
    """售后诉求抽取结果。只从原文抽取,禁止编造;无法判断一律 null。"""

    order_no: str | None = Field(default=None, description="原文出现的订单号;未提及为 null")
    request_type: RequestType | None = Field(
        default=None,
        description="诉求类型,仅可取 退货退款/仅退款/换货/维修;无法归类为 null",
    )
    desired_solution: str | None = Field(default=None, description="用户期望的处理方案(自由文本);未明说为 null")

    @field_validator("order_no", "request_type", "desired_solution", mode="before")
    @classmethod
    def _literal_null_to_none(cls, v):
        """把模型误输出成字符串的 "null"/"none"/"" 归一为 JSON null,避免枚举/类型校验崩溃。"""
        if isinstance(v, str) and v.strip().lower() in {"null", "none", ""}:
            return None
        return v

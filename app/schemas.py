from enum import Enum

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str | None = Field(default=None, description="会话 id;缺省由服务端生成并回传")
    message: str = Field(min_length=1, max_length=4000)


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


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

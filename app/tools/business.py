"""订单 / 商品 / 物流三个查询工具(演示用 mock 数据,不接真实接口)。"""
import json

from langchain_core.tools import tool

from app.tools.mockdata import mock_logistics, mock_order, mock_product


@tool
def query_order(order_id: str) -> str:
    """按订单号查询订单信息(状态、金额、购买的商品)。当用户提到具体订单、订单号、买的东西时使用。"""
    return json.dumps(mock_order(order_id), ensure_ascii=False)


@tool
def query_product(product_id: str) -> str:
    """按商品编号查询某件商品的信息(名称、价格、库存、该商品的保修规格)。仅当用户给出具体商品编号、问这件商品本身时才用;问通用规则/政策请用 query_faq。"""
    return json.dumps(mock_product(product_id), ensure_ascii=False)


@tool
def query_logistics(order_id: str) -> str:
    """按订单号查询物流进度(承运商、运单号、当前节点、预计到达)。当用户问发货没、到哪了、什么时候到、物流进度时使用。"""
    return json.dumps(mock_logistics(order_id), ensure_ascii=False)

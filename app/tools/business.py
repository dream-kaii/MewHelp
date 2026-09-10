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
    """按商品编号查询商品信息(名称、价格、库存、保修)。当用户问商品本身的价格/库存/保修时使用。"""
    return json.dumps(mock_product(product_id), ensure_ascii=False)


@tool
def query_logistics(order_id: str) -> str:
    """按订单号查询物流进度(承运商、运单号、当前节点、预计到达)。当用户问发货没、到哪了、什么时候到、物流进度时使用。"""
    return json.dumps(mock_logistics(order_id), ensure_ascii=False)

import json

from app.tools.business import query_logistics, query_order, query_product
from app.tools.mockdata import mock_logistics, mock_order


def test_mock_is_deterministic_per_input():
    assert mock_order("1001") == mock_order("1001")
    assert mock_order("1001") != mock_order("1002")
    assert mock_logistics("1001") == mock_logistics("1001")


def test_order_shape():
    o = mock_order("1001")
    assert o["order_id"] == "1001" and o["status"] and o["amount"] > 0 and o["product_id"]


def test_tools_expose_name_and_schema():
    assert query_order.name == "query_order"
    assert query_product.name == "query_product"
    assert query_logistics.name == "query_logistics"
    assert "order_id" in query_order.args_schema.model_fields
    assert query_order.description  # 供模型选型的说明不能为空


def test_tool_invoke_returns_json_string():
    out = json.loads(query_order.invoke({"order_id": "1001"}))
    assert out["order_id"] == "1001"

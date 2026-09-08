from app.main import app
from app.routers.extract import resolve_extract_service
from app.schemas import AfterSalesExtract, RequestType


class _StubOk:
    async def extract(self, text: str) -> AfterSalesExtract:
        return AfterSalesExtract(
            order_no="20260908001",
            request_type=RequestType.RETURN_REFUND,
            desired_solution="上门取件退货",
        )


class _StubBoom:
    async def extract(self, text: str):
        raise RuntimeError("upstream boom")


def test_extract_route_returns_shaped_json(client):
    app.dependency_overrides[resolve_extract_service] = lambda: _StubOk()
    try:
        r = client.post("/api/extract", json={"text": "订单20260908001漏气想退货上门取件"})
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 200
    assert r.json() == {
        "order_no": "20260908001",
        "request_type": "退货退款",
        "desired_solution": "上门取件退货",
    }


def test_extract_route_service_error_maps_to_502(client):
    app.dependency_overrides[resolve_extract_service] = lambda: _StubBoom()
    try:
        r = client.post("/api/extract", json={"text": "我要退货"})
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 502

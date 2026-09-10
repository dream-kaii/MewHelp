"""确定性 mock 数据:同一输入恒返回同一结果,便于演示与评测断言。"""
import hashlib
import random

from app.config import get_settings

_PRODUCTS = [
    ("猫粮 5kg", "宠物主粮", 129.0),
    ("自动喂食器", "宠物电器", 299.0),
    ("猫抓板", "宠物玩具", 49.0),
    ("逗猫棒", "宠物玩具", 19.0),
    ("跑步机", "宠物器材", 1899.0),
]
_CARRIERS = ["顺丰速运", "中通快递", "圆通速递", "京东物流"]
_NODES = ["已揽收", "运输中", "到达分拨中心", "派送中", "已签收"]


def _rng(*parts: object) -> random.Random:
    raw = "|".join(str(p) for p in parts) + "|" + get_settings().mock_seed_salt
    seed = int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12], 16)
    return random.Random(seed)


def mock_order(order_id: str) -> dict:
    r = _rng("order", order_id)
    name, cat, price = r.choice(_PRODUCTS)
    return {
        "order_id": str(order_id),
        "status": r.choice(["已发货", "待发货", "已完成", "已取消"]),
        "amount": round(price * r.randint(1, 3), 2),
        "product_id": f"P{r.randint(100, 999)}",
        "product_name": name,
        "category": cat,
        "created_at": f"2026-{r.randint(1, 9):02d}-{r.randint(1, 28):02d}",
    }


def mock_product(product_id: str) -> dict:
    r = _rng("product", product_id)
    name, cat, price = r.choice(_PRODUCTS)
    return {
        "product_id": str(product_id),
        "name": name,
        "category": cat,
        "price": price,
        "stock": r.randint(0, 200),
        "warranty": r.choice(["7 天无理由", "15 天包换", "一年质保"]),
    }


def mock_logistics(order_id: str) -> dict:
    r = _rng("logistics", order_id)
    steps = _NODES[: r.randint(2, len(_NODES))]
    return {
        "order_id": str(order_id),
        "carrier": r.choice(_CARRIERS),
        "tracking_no": "SF" + "".join(str(r.randint(0, 9)) for _ in range(12)),
        "current_node": steps[-1],
        "traces": [
            {"node": n, "time": f"2026-09-{r.randint(1, 9):02d} {r.randint(8, 20)}:00"} for n in steps
        ],
        "eta_days": r.randint(1, 4),
    }

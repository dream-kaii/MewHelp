"""FAQ 种子数据(幂等)。用法:python -m scripts.seed_faq

向 faq 表灌 6 条常见问答(退货政策 / 发票 / 运费 / 保修 / 发货时效)。
用应用自己的 app.db.base.get_sessionmaker()(异步),落在 .env 指向的库,
不直接用 pymysql —— 保证与线上读写同一连接配置。

幂等:以 question 是否已存在为准,已存在的行跳过,可反复执行。
"""
import sys

from sqlalchemy import select

from app.db.base import dispose_engine, get_sessionmaker
from app.db.models import Faq

# 注意:「运费」类目一律用「运费」措辞,不出现「邮费」二字 ——
# 验收③ 用「邮费是多少」触发关键词漏召回(预期),DB 里不应命中。
FAQ_SEED: list[dict[str, str]] = [
    {
        "question": "退货政策是怎么规定的?",
        "answer": "签收后 7 天内,商品未使用、包装完好可无理由退货;质量问题 30 天内可退换。"
        "定制商品、拆封的食品及贴身用品不支持无理由退货。",
        "category": "退货政策",
    },
    {
        "question": "怎么申请退货?流程是什么?",
        "answer": "在「我的订单」里点申请退货,选择原因并上传凭证;审核通过后按提示寄回,"
        "签收后 1-3 个工作日退款原路返回。",
        "category": "退货政策",
    },
    {
        "question": "可以开发票吗?怎么开?",
        "answer": "支持开具电子普通发票和增值税专用发票。下单时在备注里填写抬头与税号,"
        "发货后 24 小时内发送到预留邮箱。",
        "category": "发票",
    },
    {
        "question": "运费怎么算?",
        "answer": "单笔订单满 99 元包运费(偏远地区除外);未满 99 元收取 8 元基础运费,"
        "具体以结算页显示为准。",
        "category": "运费",
    },
    {
        "question": "商品保修期多久?",
        "answer": "电子类商品保修 12 个月,家电类保修 24 个月,以商品详情页标注为准;"
        "保修期内非人为损坏免费维修。",
        "category": "保修",
    },
    {
        "question": "下单后多久发货?",
        "answer": "现货商品 48 小时内发货,预售商品按详情页标注时间发货;"
        "发货后可在订单里查看物流。",
        "category": "发货时效",
    },
]


async def main() -> int:
    # Win 控制台默认 GBK,中文统计行会乱码 → 显式切 UTF-8(重定向时无此属性,跳过)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    session_factory = get_sessionmaker()
    inserted = 0
    skipped = 0
    try:
        async with session_factory() as session:
            for row in FAQ_SEED:
                exists = (
                    await session.execute(select(Faq.id).where(Faq.question == row["question"]))
                ).scalar_one_or_none()
                if exists is not None:
                    skipped += 1
                    continue
                session.add(Faq(**row))
                inserted += 1
            await session.commit()
    finally:
        await dispose_engine()

    print(f"[seed_faq] 新增 {inserted} 条,跳过(已存在){skipped} 条,共 {len(FAQ_SEED)} 条种子")
    return 0


if __name__ == "__main__":
    import anyio

    sys.exit(anyio.run(main))

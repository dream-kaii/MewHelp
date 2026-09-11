"""挖知识 job:历史对话 → LLM 抽 QA → 暂存 → 去重 → 入库。可 cron / 任务计划调用。

用法:
  python -m scripts.mine_knowledge                 # 默认近 30 天、20 个会话
  python -m scripts.mine_knowledge --limit 50 --lookback 60
  python -m scripts.mine_knowledge --no-promote    # 只挖到暂存区,便于人工审核
"""
import argparse
import asyncio
import sys

from app.config import get_settings
from app.db.base import get_sessionmaker
from app.llm import get_chat_model
from app.rag.embedder import get_embedder
from app.rag.miner import mine, promote_staged


async def main(argv: list[str] | None = None) -> int:
    s = get_settings()
    p = argparse.ArgumentParser(description="从历史对话挖知识")
    p.add_argument("--limit", type=int, default=s.mine_batch_size)
    p.add_argument("--lookback", type=int, default=s.mine_lookback_days)
    p.add_argument("--no-promote", action="store_true")
    args = p.parse_args(argv)

    sf = get_sessionmaker()
    stats = await mine(
        sf, get_chat_model(), get_embedder(s),
        limit=args.limit, lookback_days=args.lookback, sim_threshold=s.dedupe_sim_threshold,
    )
    print(f"扫描会话 {stats['conversations']} 个,新入暂存 {stats['staged']} 条")
    if not args.no_promote:
        pr = await promote_staged(sf)
        print(f"提升入库 {pr['promoted']} 条(状态 pending,待向量化)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

"""FAQ 查询工具:优先走向量检索的 retriever;retriever=None 时回退 ch02 的关键词 LIKE 查表。"""
from langchain_core.tools import BaseTool, tool
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.db import repository as repo


def make_kb_tools(session_factory: async_sessionmaker, retriever=None) -> list[BaseTool]:
    @tool
    async def query_faq(keyword: str) -> str:
        """查询店铺通用规则与政策(退货政策、运费/邮费、发票、保修期/保修政策等)。用户问「怎么退、能不能开票、保修多久、政策是什么」这类没有具体商品编号的规则问题时,优先用这个工具。"""
        if retriever is not None:
            return await retriever.search(keyword)
        async with session_factory() as session:  # 回退:ch02 的 SQL LIKE 实现
            hits = await repo.search_faq(session, keyword)
        if not hits:
            return f"FAQ 未找到与「{keyword}」相关条目(关键词检索无命中)"
        return "\n".join(f"问:{h['question']}\n答:{h['answer']}" for h in hits)

    return [query_faq]

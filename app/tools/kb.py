"""FAQ 查询工具:关键词 LIKE 查 faq 表。"""
from langchain_core.tools import BaseTool, tool
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.db import repository as repo


def make_kb_tools(session_factory: async_sessionmaker) -> list[BaseTool]:
    @tool
    async def query_faq(keyword: str) -> str:
        """查询店铺常见问题库(退货政策、发票、运费、保修等规则类问题)。用户问政策/规则/怎么退/能不能开票时使用。"""
        async with session_factory() as session:
            hits = await repo.search_faq(session, keyword)
        if not hits:
            return f"FAQ 未找到与「{keyword}」相关条目(关键词检索无命中)"
        return "\n".join(f"问:{h['question']}\n答:{h['answer']}" for h in hits)

    return [query_faq]

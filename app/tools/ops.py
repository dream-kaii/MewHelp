"""人工工单工具:写 tickets 表。"""
from langchain_core.tools import BaseTool, tool
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.db import repository as repo


def make_ops_tools(session_factory: async_sessionmaker, conversation_id: int) -> list[BaseTool]:
    @tool
    async def create_ticket(description: str, ticket_type: str) -> str:
        """创建人工工单转人工处理。ticket_type 只能取「售后」「投诉」「咨询」。用户要求人工/投诉/机器人解决不了时使用。"""
        if ticket_type not in ("售后", "投诉", "咨询"):
            return f"工单类型非法:{ticket_type},只能是 售后/投诉/咨询"
        async with session_factory() as session:
            no = await repo.create_ticket(
                session, conversation_id=conversation_id, description=description, ticket_type=ticket_type
            )
            await session.commit()
        return f"人工工单已创建,工单号 {no},我们会尽快处理"

    return [create_ticket]

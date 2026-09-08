from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_settings
from app.prompts import EXTRACT_SYSTEM, format_extract_user
from app.schemas import AfterSalesExtract


class ExtractService:
    def __init__(self, model, method: str | None = None):
        method = method or get_settings().extract_method
        self._structured = model.with_structured_output(AfterSalesExtract, method=method)

    async def extract(self, text: str) -> AfterSalesExtract:
        messages = [SystemMessage(EXTRACT_SYSTEM), HumanMessage(format_extract_user(text))]
        return await self._structured.ainvoke(messages)

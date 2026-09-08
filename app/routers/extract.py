import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from app.extract import ExtractService
from app.llm import get_chat_model
from app.schemas import AfterSalesExtract, ExtractRequest

logger = logging.getLogger("mewhelp.extract")

router = APIRouter()


def resolve_extract_service(request: Request) -> ExtractService:
    model = getattr(request.app.state, "extract_model", None)
    if model is None:
        model = get_chat_model()
    return ExtractService(model=model)


@router.post("/api/extract")
async def extract(
    body: ExtractRequest, service: ExtractService = Depends(resolve_extract_service)
) -> AfterSalesExtract:
    try:
        return await service.extract(body.text)
    except Exception:
        logger.exception("extract failed")
        raise HTTPException(status_code=502, detail="抽取服务暂时不可用")

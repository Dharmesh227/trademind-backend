import time
from dataclasses import asdict

from fastapi import APIRouter, HTTPException

from trademind.engines.sentiment.engine import SentimentEngine, SentimentResult

router = APIRouter(prefix="/analytics", tags=["Market Sentiment"])

_engine = SentimentEngine()


@router.get("/sentiment")
async def get_full_sentiment() -> dict:
    """Full market sentiment analysis with all components."""
    try:
        result: SentimentResult = await _engine.analyze()
        data = asdict(result)
        data["components"] = {k: asdict(v) for k, v in result.components.items()}
        return data
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Sentiment analysis failed: {exc}")


@router.get("/sentiment/fear-greed")
async def get_fear_greed() -> dict:
    """Fear/Greed gauge value (0-100)."""
    try:
        return await _engine.get_fear_greed_gauge()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Fear/Greed analysis failed: {exc}")

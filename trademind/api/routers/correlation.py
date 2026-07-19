"""Correlation analysis API endpoints.

Provides pairwise stock correlations, most/least correlated pairs,
and portfolio diversification scoring.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["Correlation Analysis"])


class PairCorrelationResponse(BaseModel):
    symbol1: str
    symbol2: str
    correlation: float
    relationship: str


class CorrelationMatrixResponse(BaseModel):
    matrix: dict
    highly_correlated: List[PairCorrelationResponse]
    negatively_correlated: List[PairCorrelationResponse]
    diversification_score: float
    timestamp: datetime
    symbols_used: int
    data_points: int


class CorrelationPairsResponse(BaseModel):
    highly_correlated: List[PairCorrelationResponse]
    negatively_correlated: List[PairCorrelationResponse]
    most_correlated: Optional[PairCorrelationResponse] = None
    least_correlated: Optional[PairCorrelationResponse] = None
    timestamp: datetime


class DiversificationResponse(BaseModel):
    diversification_score: float
    interpretation: str
    highly_correlated_count: int
    negatively_correlated_count: int
    symbols_analyzed: int
    data_points: int
    timestamp: datetime


def _pair_to_response(pair) -> PairCorrelationResponse:
    return PairCorrelationResponse(
        symbol1=pair.symbol1,
        symbol2=pair.symbol2,
        correlation=pair.correlation,
        relationship=pair.relationship,
    )


def _interpret_diversification(score: float) -> str:
    if score == 0:
        return "Insufficient data to assess diversification"
    if score < 0.3:
        return "Excellent diversification — holdings have low correlation"
    if score < 0.5:
        return "Good diversification — moderate correlation between holdings"
    if score < 0.7:
        return "Moderate diversification — consider adding uncorrelated assets"
    return "Poor diversification — holdings are highly correlated, add uncorrelated assets"


@router.get("/correlation", response_model=CorrelationMatrixResponse)
async def get_correlation_matrix() -> CorrelationMatrixResponse:
    """Full pairwise correlation matrix for all tracked symbols."""
    from trademind.engines.correlation.engine import correlation_engine

    try:
        result = await correlation_engine.compute()
    except Exception as exc:
        logger.error("Correlation computation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(exc)}")

    return CorrelationMatrixResponse(
        matrix=result.matrix,
        highly_correlated=[_pair_to_response(p) for p in result.highly_correlated],
        negatively_correlated=[_pair_to_response(p) for p in result.negatively_correlated],
        diversification_score=result.diversification_score,
        timestamp=result.timestamp,
        symbols_used=result.symbols_used,
        data_points=result.data_points,
    )


@router.get("/correlation/pairs", response_model=CorrelationPairsResponse)
async def get_correlation_pairs(
    limit: int = Query(default=10, ge=1, le=50, description="Max pairs per category"),
) -> CorrelationPairsResponse:
    """Most and least correlated pairs across tracked symbols."""
    from trademind.engines.correlation.engine import correlation_engine

    try:
        result = await correlation_engine.compute()
    except Exception as exc:
        logger.error("Correlation computation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(exc)}")

    high = [_pair_to_response(p) for p in result.highly_correlated[:limit]]
    neg = [_pair_to_response(p) for p in result.negatively_correlated[:limit]]

    most = high[0] if high else None
    least = neg[0] if neg else None

    return CorrelationPairsResponse(
        highly_correlated=high,
        negatively_correlated=neg,
        most_correlated=most,
        least_correlated=least,
        timestamp=result.timestamp,
    )


@router.get("/diversification", response_model=DiversificationResponse)
async def get_diversification_score() -> DiversificationResponse:
    """Portfolio diversification score based on average absolute correlation."""
    from trademind.engines.correlation.engine import correlation_engine

    try:
        result = await correlation_engine.compute()
    except Exception as exc:
        logger.error("Correlation computation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(exc)}")

    return DiversificationResponse(
        diversification_score=result.diversification_score,
        interpretation=_interpret_diversification(result.diversification_score),
        highly_correlated_count=len(result.highly_correlated),
        negatively_correlated_count=len(result.negatively_correlated),
        symbols_analyzed=result.symbols_used,
        data_points=result.data_points,
        timestamp=result.timestamp,
    )

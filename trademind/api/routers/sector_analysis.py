"""Sector rotation and multi-timeframe API endpoints."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["Analytics"])


class SectorRotationResponse(BaseModel):
    rotation_phase: str
    leading_sectors: List[str]
    lagging_sectors: List[str]
    breadth_percentile: float
    rotation_strength: float
    momentum_scores: Dict[str, float]
    momentum_ranks: Dict[str, int]
    rotation_change: Optional[Dict] = None
    data_points: int


class MultiTimeframeRequest(BaseModel):
    symbol: str
    timeframes: List[str] = Field(default=["daily", "hourly", "15min"])
    weights: Optional[Dict[str, float]] = None


class MultiTimeframeResponse(BaseModel):
    symbol: str
    fused_score: float
    confidence: str
    alignment: float
    per_timeframe_scores: Dict[str, float]
    fused_features: Dict[str, float]


@router.get("/sector-rotation", response_model=SectorRotationResponse)
async def get_sector_rotation() -> SectorRotationResponse:
    """Get current sector rotation analysis."""
    from trademind.engines.sector_rotation.tracker import sector_tracker
    from trademind.engines.sector_rotation.detector import rotation_detector

    regime = rotation_detector.detect_current_regime(sector_tracker)
    ranks = sector_tracker.get_momentum_ranks()
    change = rotation_detector.detect_rotation_change(sector_tracker)

    return SectorRotationResponse(
        rotation_phase=regime.rotation_phase,
        leading_sectors=regime.leading_sectors,
        lagging_sectors=regime.lagging_sectors,
        breadth_percentile=regime.breadth_percentile,
        rotation_strength=regime.rotation_strength,
        momentum_scores=regime.momentum_scores,
        momentum_ranks=ranks,
        rotation_change={
            "old_leaders": change.old_leaders,
            "new_leaders": change.new_leaders,
            "direction": change.direction,
            "confidence": change.confidence,
            "description": change.description,
        } if change else None,
        data_points=sector_tracker.data_points,
    )


@router.post("/multi-timeframe", response_model=MultiTimeframeResponse)
async def get_multi_timeframe_analysis(
    request: MultiTimeframeRequest,
) -> MultiTimeframeResponse:
    """Run multi-timeframe analysis for a single symbol."""
    from trademind.engines.multi_timeframe.aggregator import timeframe_aggregator
    from trademind.engines.multi_timeframe.fusion import timeframe_fusion
    from trademind.engines.ai_scoring.scorer import AIScoreEngine

    try:
        all_tf_data = await timeframe_aggregator.fetch_all_timeframes([request.symbol])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data fetch failed: {str(e)}")

    symbol_data = all_tf_data.get(request.symbol, {})
    if not symbol_data:
        raise HTTPException(status_code=404, detail=f"No data for {request.symbol}")

    features_by_tf = timeframe_aggregator.extract_features_per_timeframe(
        request.symbol, symbol_data
    )

    if not features_by_tf:
        raise HTTPException(status_code=404, detail="Feature extraction failed")

    fused_features = timeframe_fusion.fuse_features(features_by_tf, request.weights)

    scorer = AIScoreEngine()
    score_results = {}
    for tf_name, tf_features in features_by_tf.items():
        try:
            result = scorer.score_symbol(request.symbol, tf_features)
            score_results[tf_name] = result
        except Exception as e:
            logger.warning("Scoring failed for %s/%s: %s", request.symbol, tf_name, e)

    fused_score = timeframe_fusion.fuse_scores(score_results, request.weights)
    per_tf_scores = {tf: r.overall_score for tf, r in score_results.items()}

    return MultiTimeframeResponse(
        symbol=request.symbol,
        fused_score=fused_score.overall_score,
        confidence=fused_score.confidence_level,
        alignment=fused_features.get("timeframe_alignment", 50.0),
        per_timeframe_scores=per_tf_scores,
        fused_features=fused_features,
    )

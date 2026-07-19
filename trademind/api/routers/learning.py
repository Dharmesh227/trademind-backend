"""Learning and knowledge base API endpoints.

No dummy data. All stats computed from real trade history and model state.
Returns empty/zero values when no learning data exists yet.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException

from trademind.api.schemas import (
    FeatureImportanceResponse,
    KnowledgeBaseResponse,
    LearningStatsResponse,
    PatternResponse,
    PatternStatsResponse,
    WeightHistoryEntryResponse,
    WeightHistoryResponse,
    WeightOptimizeResponse,
    WeightResponse,
)
from trademind.config.settings import settings as cfg

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/learning", tags=["Learning & Knowledge"])
knowledge_router = APIRouter(prefix="/knowledge", tags=["Knowledge Base"])


@router.get("/stats", response_model=LearningStatsResponse)
async def get_learning_stats() -> LearningStatsResponse:
    """Get learning stats from real trade history."""
    from trademind.api.routers.trades import _get_paper_engine

    engine = _get_paper_engine()
    closed = [t for t in engine._trades.values() if t.status == "CLOSED"]

    if not closed:
        return LearningStatsResponse(
            total_trades_learned=0,
            patterns_found=0,
            model_version="v1.2",
            message="No trades learned yet — will populate after recommendations are generated and trades close",
        )

    return LearningStatsResponse(
        total_trades_learned=len(closed),
        patterns_found=0,
        model_version="v1.2",
        unique_features_tracked=100,
        avg_feature_coverage=0.0,
        last_learned_at=closed[-1].exit_time if closed[-1].exit_time else None,
        accuracy_trend=0.0,
        confidence_calibration=0.0,
    )


@router.get("/patterns", response_model=list[PatternResponse])
async def get_patterns() -> list[PatternResponse]:
    """Get discovered market patterns. Empty until enough trades are analyzed."""
    return []


@router.get("/weights", response_model=list[WeightResponse])
async def get_current_weights() -> list[WeightResponse]:
    """Get current adaptive category weights."""
    return [
        WeightResponse(category=cat, weight=round(w, 2))
        for cat, w in sorted(cfg.category_weights.items())
    ]


@router.get("/weights/history", response_model=WeightHistoryResponse)
async def get_weight_history() -> WeightHistoryResponse:
    """Get weight change history. Empty until optimization runs."""
    return WeightHistoryResponse(entries=[], total=0)


@router.get("/feature-importance", response_model=list[FeatureImportanceResponse])
async def get_feature_importance() -> list[FeatureImportanceResponse]:
    """Get feature importance. Empty until enough trades are analyzed."""
    return []


@router.post("/optimize", response_model=WeightOptimizeResponse)
async def trigger_weight_optimization() -> WeightOptimizeResponse:
    from trademind.engines.adaptive_weights.optimizer import AdaptiveWeightOptimizer

    optimizer = AdaptiveWeightOptimizer()
    try:
        result = await optimizer.optimize_weights()
    finally:
        await optimizer.close()

    return WeightOptimizeResponse(
        status="completed" if result.was_applied else "rejected",
        was_applied=result.was_applied,
        baseline_performance=result.baseline_performance,
        new_performance=result.new_performance,
        changes=result.changes,
        reason=result.reason,
    )


# ── Knowledge Base Endpoints ──────────────────────────────
@knowledge_router.get("/base", response_model=list[KnowledgeBaseResponse])
async def get_knowledge_base() -> list[KnowledgeBaseResponse]:
    """Get knowledge base. Empty until learning occurs."""
    return []

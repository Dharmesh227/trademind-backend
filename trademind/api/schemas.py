"""Pydantic request/response schemas for the TradeMind AI REST API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, List, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


# ── Generic Pagination Wrapper ─────────────────────────────
class PaginatedResponse(BaseModel, Generic[T]):
    items: List[T]
    total: int
    page: int
    page_size: int
    total_pages: int


# ── Market Data ────────────────────────────────────────────
class MarketDataResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None
    delivery_volume: Optional[int] = None
    delivery_percent: Optional[float] = None
    prev_close: Optional[float] = None
    change_percent: Optional[float] = None
    turnover: Optional[float] = None


class OptionStrikeResponse(BaseModel):
    strike_price: float
    expiry_date: str
    option_type: str
    open_interest: int
    change_in_oi: int
    volume: int
    iv: float
    last_price: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None


class OptionChainResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    symbol: str
    timestamp: datetime
    expiry_date: Optional[str] = None
    strikes: List[OptionStrikeResponse] = []
    pcr: Optional[float] = None
    total_ce_oi: Optional[int] = None
    total_pe_oi: Optional[int] = None
    max_pain: Optional[float] = None
    iv_rank: Optional[float] = None
    iv_percentile: Optional[float] = None


class IndexDataResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    symbol: str
    timestamp: datetime
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    change_percent: Optional[float] = None
    volume: Optional[int] = None


class VIXDataResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    timestamp: datetime
    value: float
    change_percent: Optional[float] = None
    prev_close: Optional[float] = None


class MarketBreadthResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    timestamp: datetime
    advances: int
    declines: int
    unchanged: int
    advance_decline_ratio: Optional[float] = None
    total_traded: Optional[int] = None


class DataCollectionResponse(BaseModel):
    status: str
    message: str
    symbols_collected: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)


# ── Feature Vector ─────────────────────────────────────────
class FeatureVectorResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    symbol: str
    timestamp: datetime
    features: dict[str, float]
    feature_count: int
    data_completeness: float
    calculation_time_ms: Optional[float] = None


# ── AI Score ───────────────────────────────────────────────
class CategoryScoreResponse(BaseModel):
    name: str
    score: float
    weight: Optional[float] = None


class AIScoreResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    symbol: str
    timestamp: datetime
    score: float
    confidence: float
    confidence_level: str
    category_scores: List[CategoryScoreResponse] = []
    evidence: List[str] = []
    signal_strength: Optional[float] = None
    time_horizon: str = "intraday"
    model_version: str = "v1"
    rank: Optional[int] = None
    total_scored: Optional[int] = None


class ScoreRankingResponse(BaseModel):
    symbol: str
    score: float
    confidence: float
    confidence_level: str
    rank: int
    category_scores: dict[str, float] = {}
    evidence: List[str] = []


class ScoreRefreshResponse(BaseModel):
    status: str
    scored_count: int
    timestamp: datetime = Field(default_factory=datetime.now)


# ── Trade Recommendation ──────────────────────────────────
class EvidenceItem(BaseModel):
    text: str
    category: Optional[str] = None
    strength: Optional[str] = None


class TradeRecommendationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: Optional[str] = None
    symbol: str
    timestamp: datetime
    action: str
    entry_price: float
    stop_loss: float
    target: float
    confidence: float
    expected_move_percent: Optional[float] = None
    holding_period: str = "intraday"
    risk_reward_ratio: Optional[float] = None
    position_size_percent: Optional[float] = None
    evidence: List[str] = []
    is_active: bool = True


class RecommendationHistoryResponse(BaseModel):
    id: Optional[str] = None
    symbol: str
    timestamp: datetime
    action: str
    entry_price: float
    stop_loss: float
    target: float
    confidence: float
    is_active: bool


class RecommendationGenerateResponse(BaseModel):
    status: str
    generated_count: int
    timestamp: datetime = Field(default_factory=datetime.now)


# ── Paper Trade ────────────────────────────────────────────
class PaperTradeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    recommendation_id: Optional[str] = None
    symbol: str
    action: Optional[str] = None
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    stop_loss: float
    target: float
    quantity: int
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    holding_time_hours: Optional[float] = None
    max_drawdown: Optional[float] = None
    max_profit: Optional[float] = None
    exit_reason: Optional[str] = None
    status: str
    commission: float = 0.0
    slippage: float = 0.0


class PortfolioSummary(BaseModel):
    total_trades: int = 0
    open_positions: int = 0
    closed_trades: int = 0
    total_pnl: float = 0.0
    total_pnl_percent: float = 0.0
    win_rate: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    current_capital: float = 100000.0
    initial_capital: float = 100000.0
    winning_trades: int = 0
    losing_trades: int = 0


class TradeExecuteResponse(BaseModel):
    status: str
    trade: Optional[PaperTradeResponse] = None
    message: str = ""


class TradeCloseResponse(BaseModel):
    status: str
    trade: Optional[PaperTradeResponse] = None
    message: str = ""


# ── Analytics ──────────────────────────────────────────────
class DashboardStatsResponse(BaseModel):
    ai_accuracy: float = 0.0
    win_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    total_trades: int = 0
    best_scanner: str = ""
    worst_scanner: str = ""
    best_sector: str = ""
    best_time: str = ""
    best_day: str = ""
    worst_day: str = ""
    max_drawdown: float = 0.0
    max_win_streak: int = 0
    max_loss_streak: int = 0
    avg_holding_hours: float = 0.0
    expectancy: float = 0.0
    total_pnl: float = 0.0
    avg_trade_pnl: float = 0.0
    kelly_criterion: float = 0.0
    message: Optional[str] = None


class AccuracyTimeResponse(BaseModel):
    timestamps: List[datetime] = []
    accuracy_values: List[float] = []
    trade_count: int = 0


class CategoryPerformanceResponse(BaseModel):
    category: str
    total_trades: int = 0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0
    avg_pnl_pct: float = 0.0
    total_pnl: float = 0.0
    profit_factor: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0


class TimePerformanceResponse(BaseModel):
    hourly: List[CategoryPerformanceResponse] = []
    daily: List[CategoryPerformanceResponse] = []


class DrawdownResponse(BaseModel):
    max_drawdown: float = 0.0
    recovery_trades: float = 0.0


# ── Learning ───────────────────────────────────────────────
class LearningStatsResponse(BaseModel):
    total_trades_learned: int = 0
    patterns_found: int = 0
    model_version: str = "v1"
    unique_features_tracked: int = 0
    avg_feature_coverage: float = 0.0
    last_learned_at: Optional[datetime] = None
    accuracy_trend: float = 0.0
    confidence_calibration: float = 0.0
    message: Optional[str] = None


class PatternInsightResponse(BaseModel):
    feature_name: str
    direction: str
    condition: str
    value: float
    impact_pct: float
    description: str


class PatternResponse(BaseModel):
    name: str
    description: str
    conditions: dict[str, Any] = {}
    category: str
    trade_count: int = 0
    win_count: int = 0
    success_rate: float = 0.0
    avg_return: float = 0.0
    confidence: float = 0.0


class PatternStatsResponse(BaseModel):
    total_patterns: int = 0
    avg_accuracy: float = 0.0
    best_pattern: str = ""
    best_pattern_accuracy: float = 0.0
    active_patterns: int = 0
    avg_trades_per_pattern: float = 0.0


class WeightResponse(BaseModel):
    category: str
    weight: float


class WeightHistoryEntryResponse(BaseModel):
    id: Optional[str] = None
    category: str
    feature_name: str
    old_weight: float
    new_weight: float
    change_reason: str
    effective_date: datetime
    trade_count_sample: Optional[int] = None
    avg_impact: Optional[float] = None


class WeightHistoryResponse(BaseModel):
    entries: List[WeightHistoryEntryResponse] = []
    total: int


class KnowledgeBaseResponse(BaseModel):
    id: Optional[str] = None
    version: str
    model_name: str
    model_type: str = "ensemble"
    trades_learned: int = 0
    patterns_found: int = 0
    accuracy: float = 0.0
    precision_score: Optional[float] = None
    recall_score: Optional[float] = None
    f1_score: Optional[float] = None
    confidence: float = 0.0
    training_data_start: Optional[datetime] = None
    training_data_end: Optional[datetime] = None
    is_active: bool = True


class WeightOptimizeResponse(BaseModel):
    status: str
    was_applied: bool
    baseline_performance: float
    new_performance: float
    changes: dict[str, float] = {}
    reason: str = ""


class FeatureImportanceResponse(BaseModel):
    feature: str
    importance: float
    category: str
    correlation_with_profit: float
    description: str


# ── Health & System ────────────────────────────────────────
class SchedulerJobResponse(BaseModel):
    job_id: str
    name: str
    next_run_time: Optional[str] = None
    trigger: str


class SchedulerStatusResponse(BaseModel):
    status: str
    jobs: List[SchedulerJobResponse] = []


class HealthResponse(BaseModel):
    status: str
    version: str
    database: str = "unknown"
    scheduler: str = "unknown"
    uptime_seconds: float = 0.0


# ── Error ──────────────────────────────────────────────────
class ErrorResponse(BaseModel):
    detail: str
    code: Optional[str] = None
    details: Optional[dict[str, Any]] = None


# ── Request Bodies ─────────────────────────────────────────
class MarketCollectRequest(BaseModel):
    symbols: Optional[List[str]] = None


class ScoreRefreshRequest(BaseModel):
    symbols: Optional[List[str]] = None


class RecommendationGenerateRequest(BaseModel):
    symbols: Optional[List[str]] = None


class TradeCloseRequest(BaseModel):
    exit_price: Optional[float] = None
    exit_reason: str = "manual"

"""SQLAlchemy ORM models and transient dataclasses for TradeMind AI."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    """Declarative base for all TradeMind models."""

    pass


class TimestampMixin:
    """Mixin that adds created_at / updated_at columns."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


# ── Market Data ─────────────────────────────────────────────
class MarketData(Base, TimestampMixin):
    """OHLCV + derived market data for every symbol."""

    __tablename__ = "tm_market_data"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    symbol: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    vwap: Mapped[float | None] = mapped_column(Float, nullable=True)
    delivery_volume: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    delivery_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    turnover: Mapped[float | None] = mapped_column(Float, nullable=True)
    trades: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_adjusted: Mapped[bool] = mapped_column(Boolean, default=False)
    source: Mapped[str] = mapped_column(String(50), default="nse")

    __table_args__ = (
        Index("ix_tm_market_data_symbol_ts", "symbol", "timestamp", unique=True),
    )


# ── Option Chain ────────────────────────────────────────────
class OptionChain(Base, TimestampMixin):
    """Option chain snapshot for a symbol at a point in time."""

    __tablename__ = "tm_option_chain"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    symbol: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    strike: Mapped[float] = mapped_column(Float, nullable=False)
    option_type: Mapped[str] = mapped_column(String(4), nullable=False)  # CE / PE
    ltp: Mapped[float] = mapped_column(Float, nullable=False)
    open_interest: Mapped[int] = mapped_column(BigInteger, default=0)
    change_oi: Mapped[int] = mapped_column(BigInteger, default=0)
    volume: Mapped[int] = mapped_column(BigInteger, default=0)
    iv: Mapped[float | None] = mapped_column(Float, nullable=True)
    delta: Mapped[float | None] = mapped_column(Float, nullable=True)
    gamma: Mapped[float | None] = mapped_column(Float, nullable=True)
    theta: Mapped[float | None] = mapped_column(Float, nullable=True)
    vega: Mapped[float | None] = mapped_column(Float, nullable=True)
    pcr_oi: Mapped[float | None] = mapped_column(Float, nullable=True)
    pcr_volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    underlying_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    expiry: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_tm_option_chain_symbol_ts", "symbol", "timestamp", "strike", "option_type"),
    )


# ── Feature Vector ──────────────────────────────────────────
class FeatureVector(Base, TimestampMixin):
    """Computed feature vectors stored as JSON for auditability."""

    __tablename__ = "tm_feature_vectors"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    symbol: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    features_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    feature_count: Mapped[int] = mapped_column(Integer, default=0)
    data_completeness: Mapped[float] = mapped_column(Float, default=0.0)
    calculation_time_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    __table_args__ = (
        Index("ix_tm_feature_vectors_symbol_ts", "symbol", "timestamp", unique=True),
    )


# ── AI Score ────────────────────────────────────────────────
class AIScore(Base, TimestampMixin):
    """Composite AI score with per-category breakdowns."""

    __tablename__ = "tm_ai_scores"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    symbol: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    category_scores_json: Mapped[str] = mapped_column(Text, default="{}")
    evidence_json: Mapped[str] = mapped_column(Text, default="[]")
    signal_strength: Mapped[float | None] = mapped_column(Float, nullable=True)
    time_horizon: Mapped[str] = mapped_column(String(20), default="intraday")
    model_version: Mapped[str] = mapped_column(String(50), default="v1")

    __table_args__ = (
        Index("ix_tm_ai_scores_symbol_ts", "symbol", "timestamp", unique=True),
    )


# ── Trade Recommendation ───────────────────────────────────
class TradeRecommendation(Base, TimestampMixin):
    """Actionable trade recommendation with entry/SL/target."""

    __tablename__ = "tm_trade_recommendations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    symbol: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(20), nullable=False)  # BUY / SELL / HOLD
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    stop_loss: Mapped[float] = mapped_column(Float, nullable=False)
    target: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    expected_move_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    holding_period: Mapped[str] = mapped_column(String(30), default="intraday")
    risk_reward_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    position_size_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    evidence_json: Mapped[str] = mapped_column(Text, default="[]")
    score_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("tm_ai_scores.id"), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    ai_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    recommendation_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    evidence_panel_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_tm_trade_recommendations_symbol_ts", "symbol", "timestamp"),
    )

    score: Mapped[AIScore | None] = relationship(back_populates="recommendations")
    paper_trades: Mapped[list[PaperTrade]] = relationship(back_populates="recommendation")


# ── Paper Trade ─────────────────────────────────────────────
class PaperTrade(Base, TimestampMixin):
    """Paper-trade execution record tracking full lifecycle."""

    __tablename__ = "tm_paper_trades"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    trade_id: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)
    recommendation_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("tm_trade_recommendations.id"), nullable=True
    )
    symbol: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(20), nullable=True)
    entry_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    exit_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop_loss: Mapped[float] = mapped_column(Float, nullable=False)
    target: Mapped[float] = mapped_column(Float, nullable=False)
    initial_sl: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_sl: Mapped[float | None] = mapped_column(Float, nullable=True)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    holding_time_hours: Mapped[float | None] = mapped_column(Float, nullable=True)
    holding_period_hours: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_drawdown: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_reason: Mapped[str | None] = mapped_column(String(30), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="active", index=True)
    commission: Mapped[float] = mapped_column(Float, default=0.0)
    slippage: Mapped[float] = mapped_column(Float, default=0.0)
    ai_score_at_entry: Mapped[float | None] = mapped_column(Float, nullable=True)
    evidence_at_entry_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    trailing_stop_activated: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (
        Index("ix_tm_paper_trades_symbol_entry", "symbol", "entry_time"),
    )

    recommendation: Mapped[TradeRecommendation | None] = relationship(back_populates="paper_trades")
    learning_records: Mapped[list[LearningRecord]] = relationship(back_populates="trade")


# ── Learning Record ─────────────────────────────────────────
class LearningRecord(Base, TimestampMixin):
    """Audit trail of self-learning weight adjustments per trade."""

    __tablename__ = "tm_learning_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    trade_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("tm_paper_trades.id"), nullable=True
    )
    features_snapshot_json: Mapped[str] = mapped_column(Text, default="{}")
    predicted_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_outcome: Mapped[float | None] = mapped_column(Float, nullable=True)
    outcome: Mapped[str] = mapped_column(String(20), nullable=False)  # win / loss / breakeven
    pnl_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    weight_changes_json: Mapped[str] = mapped_column(Text, default="{}")
    feature_importance_json: Mapped[str] = mapped_column(Text, default="{}")
    learning_rate_applied: Mapped[float | None] = mapped_column(Float, nullable=True)

    trade: Mapped[PaperTrade | None] = relationship(back_populates="learning_records")


# ── Adaptive Weight ─────────────────────────────────────────
class AdaptiveWeight(Base, TimestampMixin):
    """Tracks every weight change made by the learning system."""

    __tablename__ = "tm_adaptive_weights"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    category: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    feature_name: Mapped[str] = mapped_column(String(60), nullable=False, index=True)
    old_weight: Mapped[float] = mapped_column(Float, nullable=False)
    new_weight: Mapped[float] = mapped_column(Float, nullable=False)
    change_reason: Mapped[str] = mapped_column(String(200), default="")
    effective_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    trade_count_sample: Mapped[int | None] = mapped_column(Integer, nullable=True)
    avg_impact: Mapped[float | None] = mapped_column(Float, nullable=True)


# ── Pattern ─────────────────────────────────────────────────
class Pattern(Base, TimestampMixin):
    """Discovered market patterns with performance statistics."""

    __tablename__ = "tm_patterns"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    pattern_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    conditions_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    sector: Mapped[str | None] = mapped_column(String(50), nullable=True)
    occurrences: Mapped[int] = mapped_column(Integer, default=0)
    wins: Mapped[int] = mapped_column(Integer, default=0)
    losses: Mapped[int] = mapped_column(Integer, default=0)
    success_rate: Mapped[float] = mapped_column(Float, default=0.0)
    avg_return_percent: Mapped[float] = mapped_column(Float, default=0.0)
    max_return_percent: Mapped[float] = mapped_column(Float, default=0.0)
    min_return_percent: Mapped[float] = mapped_column(Float, default=0.0)
    sharpe_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    discovered_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


# ── Knowledge Base ──────────────────────────────────────────
class KnowledgeBase(Base, TimestampMixin):
    """Versioned model / strategy knowledge archive."""

    __tablename__ = "tm_knowledge_base"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    version: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_type: Mapped[str] = mapped_column(String(50), default="ensemble")
    trades_learned: Mapped[int] = mapped_column(Integer, default=0)
    patterns_found: Mapped[int] = mapped_column(Integer, default=0)
    accuracy: Mapped[float] = mapped_column(Float, default=0.0)
    precision_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    f1_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    training_data_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    training_data_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    model_path: Mapped[str | None] = mapped_column(String(300), nullable=True)
    changelog_json: Mapped[str] = mapped_column(Text, default="[]")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


# ── Transient / In-Memory Models ────────────────────────────
# These dataclasses are for in-memory data transport and API responses.
# They are NOT stored in the database directly.

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional


@dataclass
class PriceData:
    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    vwap: Decimal
    delivery_volume: Optional[int] = None
    delivery_percent: Optional[Decimal] = None
    prev_close: Optional[Decimal] = None
    change_percent: Optional[Decimal] = None
    turnover: Optional[Decimal] = None


@dataclass
class OptionChainData:
    symbol: str
    timestamp: datetime
    expiry_date: Optional[str] = None
    strikes: List[Dict] = field(default_factory=list)
    pcr: Optional[Decimal] = None
    total_ce_oi: Optional[int] = None
    total_pe_oi: Optional[int] = None
    max_pain: Optional[Decimal] = None
    iv_rank: Optional[Decimal] = None
    iv_percentile: Optional[Decimal] = None


@dataclass
class OptionStrike:
    strike_price: Decimal
    expiry_date: str
    option_type: str
    open_interest: int
    change_in_oi: int
    volume: int
    iv: Decimal
    last_price: Decimal
    delta: Optional[Decimal] = None
    gamma: Optional[Decimal] = None
    theta: Optional[Decimal] = None
    vega: Optional[Decimal] = None


@dataclass
class MarketBreadthData:
    timestamp: datetime
    advances: int
    declines: int
    unchanged: int
    advance_decline_ratio: Optional[Decimal] = None
    total_traded: Optional[int] = None


@dataclass
class InstitutionalFlowData:
    timestamp: datetime
    fii_cash: Optional[Decimal] = None
    fii_fno: Optional[Decimal] = None
    dii_cash: Optional[Decimal] = None
    total_fii: Optional[Decimal] = None
    total_dii: Optional[Decimal] = None


@dataclass
class IndexData:
    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    change_percent: Optional[Decimal] = None
    volume: Optional[int] = None


@dataclass
class VIXData:
    timestamp: datetime
    value: Decimal
    change_percent: Optional[Decimal] = None
    prev_close: Optional[Decimal] = None


@dataclass
class FeatureSet:
    symbol: str
    timestamp: datetime
    features: Dict[str, float]
    feature_count: int = 0

    def __post_init__(self):
        self.feature_count = len(self.features)


@dataclass
class AIScoreResult:
    symbol: str
    timestamp: datetime
    overall_score: float
    trend_score: float = 0.0
    momentum_score: float = 0.0
    volume_score: float = 0.0
    options_score: float = 0.0
    sector_score: float = 0.0
    market_score: float = 0.0
    volatility_score: float = 0.0
    confidence_level: str = "Low"
    confidence_score: float = 0.0
    evidence: List[str] = field(default_factory=list)
    rank: Optional[int] = None
    total_scored: Optional[int] = None


@dataclass
class TradeRecData:
    symbol: str
    timestamp: datetime
    action: str
    ai_score: float
    confidence_level: str
    entry_price: Decimal
    stop_loss: Decimal
    target: Decimal
    expected_move_percent: float
    holding_period_suggestion: str
    risk_assessment: str
    evidence_panel: List[str] = field(default_factory=list)
    probability: Optional[float] = None
    recommendation_id: Optional[str] = None
    scores: Optional[AIScoreResult] = None


@dataclass
class PaperTradeData:
    trade_id: str
    symbol: str
    action: str
    entry_price: Decimal
    quantity: int
    entry_time: datetime
    stop_loss: Decimal
    target: Decimal
    initial_sl: Decimal
    current_sl: Optional[Decimal] = None
    exit_price: Optional[Decimal] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    pnl: Optional[Decimal] = None
    pnl_percent: Optional[float] = None
    holding_period_hours: Optional[float] = None
    max_drawdown: Optional[float] = None
    max_profit: Optional[float] = None
    trailing_stop_activated: bool = False
    status: str = "OPEN"
    ai_score_at_entry: Optional[float] = None
    recommendation_id: Optional[str] = None
    evidence_at_entry: List[str] = field(default_factory=list)


@dataclass
class PortfolioSummary:
    total_trades: int = 0
    open_positions: int = 0
    closed_trades: int = 0
    total_pnl: Decimal = Decimal("0.00")
    total_pnl_percent: float = 0.0
    win_rate: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    current_capital: Decimal = Decimal("100000.00")
    initial_capital: Decimal = Decimal("100000.00")
    winning_trades: int = 0
    losing_trades: int = 0
    open_positions_list: List[PaperTradeData] = field(default_factory=list)


# ── Rebuild relationships after all models are defined ─────
AIScore.recommendations = relationship("TradeRecommendation", back_populates="score")  # type: ignore[attr-defined]

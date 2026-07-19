"""Custom exception hierarchy for TradeMind AI."""

from __future__ import annotations

from typing import Any


class TradeMindError(Exception):
    """Base exception for all TradeMind errors."""

    def __init__(self, message: str = "An unexpected error occurred", details: dict[str, Any] | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r})"


# ── Data Layer ──────────────────────────────────────────────
class DataCollectionError(TradeMindError):
    """Raised when market data collection fails."""


class DataValidationError(TradeMindError):
    """Raised when collected data fails validation."""


class NSEConnectionError(DataCollectionError):
    """Raised when connection to NSE APIs fails."""


class RateLimitError(DataCollectionError):
    """Raised when NSE rate limits are exceeded."""


class CookieExpiredError(DataCollectionError):
    """Raised when NSE session cookies have expired."""


# ── Feature Engineering ─────────────────────────────────────
class FeatureEngineeringError(TradeMindError):
    """Raised during feature computation or transformation."""


class FeatureCalculationError(FeatureEngineeringError):
    """Raised when a specific feature calculation fails."""


class FeatureStoreError(FeatureEngineeringError):
    """Raised when feature persistence fails."""


# ── Scoring & ML ───────────────────────────────────────────
class ScoringError(TradeMindError):
    """Raised when the scoring engine encounters an error."""


class ModelNotLoadedError(ScoringError):
    """Raised when a required ML model cannot be loaded."""


class ModelPredictionError(ScoringError):
    """Raised when model inference fails."""


class InsufficientDataError(ScoringError):
    """Raised when insufficient data is available for scoring."""


# ── Trading ─────────────────────────────────────────────────
class TradingError(TradeMindError):
    """Raised during trade recommendation or execution logic."""


class RiskLimitExceeded(TradingError):
    """Raised when a risk management limit is breached."""


class PositionSizeError(TradingError):
    """Raised when position sizing calculation fails."""


class StopLossError(TradingError):
    """Raised when stop-loss logic encounters an error."""


# ── Paper Trading ───────────────────────────────────────────
class PaperTradeError(TradeMindError):
    """Raised during paper trade lifecycle management."""


class InsufficientCapitalError(PaperTradeError):
    """Raised when paper trading capital is exhausted."""


# ── Learning ────────────────────────────────────────────────
class LearningError(TradeMindError):
    """Raised during the self-learning / adaptation cycle."""


class WeightUpdateError(LearningError):
    """Raised when adaptive weight updates fail."""


class PatternDiscoveryError(LearningError):
    """Raised when pattern mining encounters an error."""


# ── Scheduler ───────────────────────────────────────────────
class SchedulerError(TradeMindError):
    """Raised when the APScheduler encounters an error."""


# ── Configuration ───────────────────────────────────────────
class ConfigurationError(TradeMindError):
    """Raised for invalid or missing configuration."""

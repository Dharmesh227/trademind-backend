"""Correlation Matrix engine — pairwise stock correlations and diversification metrics.

Uses yfinance for 3-month daily close prices, pandas for Pearson correlation,
and asyncio.to_thread to keep the event loop responsive.
"""

from __future__ import annotations

import asyncio
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from trademind.config.settings import settings

logger = logging.getLogger(__name__)

_NS_SUFFIX = ".NS"
_CACHE_TTL: float = 3600.0  # 1 hour


@dataclass
class PairCorrelation:
    symbol1: str
    symbol2: str
    correlation: float
    relationship: str  # "strong_positive" / "moderate" / "negative" / "strong_negative"


@dataclass
class CorrelationResult:
    matrix: Dict[str, Dict[str, float]]
    highly_correlated: List[PairCorrelation]
    negatively_correlated: List[PairCorrelation]
    diversification_score: float
    timestamp: datetime
    symbols_used: int
    data_points: int


class CorrelationEngine:
    """Computes pairwise Pearson correlations for tracked NSE symbols."""

    def __init__(self) -> None:
        self._cache: Optional[CorrelationResult] = None
        self._cache_time: float = 0.0

    async def compute(self, force: bool = False) -> CorrelationResult:
        if not force and self._cache is not None and (time.time() - self._cache_time) < _CACHE_TTL:
            return self._cache

        result = await asyncio.to_thread(self._compute_sync)
        self._cache = result
        self._cache_time = time.time()
        return result

    def _compute_sync(self) -> CorrelationResult:
        symbols = settings.fno_symbols
        yahoo_symbols = [f"{s}{_NS_SUFFIX}" for s in symbols]

        logger.info("Fetching 3-month daily prices for %d symbols via yfinance", len(symbols))

        raw = yf.download(
            yahoo_symbols,
            period="3mo",
            interval="1d",
            group_by="ticker",
            progress=False,
            threads=True,
        )

        closes: Dict[str, pd.Series] = {}
        for sym in symbols:
            yahoo_sym = f"{sym}{_NS_SUFFIX}"
            try:
                df = raw[yahoo_sym] if len(yahoo_symbols) > 1 else raw
                if df is None or df.empty:
                    continue
                series = df["Close"].dropna()
                if len(series) < 5:
                    continue
                closes[sym] = series
            except Exception:
                continue

        if len(closes) < 2:
            logger.warning("Not enough symbols with data (%d) to build correlation matrix", len(closes))
            return CorrelationResult(
                matrix={},
                highly_correlated=[],
                negatively_correlated=[],
                diversification_score=0.0,
                timestamp=datetime.now(),
                symbols_used=len(closes),
                data_points=0,
            )

        prices_df = pd.DataFrame(closes)
        prices_df = prices_df.dropna(axis=1, how="all")
        prices_df = prices_df.fillna(method="ffill").fillna(method="bfill")

        corr_matrix = prices_df.corr(method="pearson")

        matrix_dict: Dict[str, Dict[str, float]] = {}
        for sym1 in corr_matrix.columns:
            matrix_dict[sym1] = {}
            for sym2 in corr_matrix.columns:
                val = corr_matrix.loc[sym1, sym2]
                matrix_dict[sym1][sym2] = round(float(val), 4) if not np.isnan(val) else 0.0

        highly_correlated: List[PairCorrelation] = []
        negatively_correlated: List[PairCorrelation] = []

        syms = list(corr_matrix.columns)
        for i in range(len(syms)):
            for j in range(i + 1, len(syms)):
                val = float(corr_matrix.iloc[i, j])
                if np.isnan(val):
                    continue

                pair = PairCorrelation(
                    symbol1=syms[i],
                    symbol2=syms[j],
                    correlation=round(val, 4),
                    relationship=_classify_relationship(val),
                )

                if abs(val) > 0.7:
                    highly_correlated.append(pair)
                if val < -0.3:
                    negatively_correlated.append(pair)

        highly_correlated.sort(key=lambda p: abs(p.correlation), reverse=True)
        negatively_correlated.sort(key=lambda p: p.correlation)

        upper_vals = []
        for i in range(len(syms)):
            for j in range(i + 1, len(syms)):
                v = float(corr_matrix.iloc[i, j])
                if not np.isnan(v):
                    upper_vals.append(abs(v))

        diversification_score = round(float(np.mean(upper_vals)), 4) if upper_vals else 0.0

        n_data_points = len(prices_df)

        logger.info(
            "Correlation matrix computed: %d symbols, %d pairs, diversification=%.4f",
            len(syms), len(upper_vals), diversification_score,
        )

        return CorrelationResult(
            matrix=matrix_dict,
            highly_correlated=highly_correlated,
            negatively_correlated=negatively_correlated,
            diversification_score=diversification_score,
            timestamp=datetime.now(),
            symbols_used=len(syms),
            data_points=n_data_points,
        )


def _classify_relationship(corr: float) -> str:
    if corr >= 0.7:
        return "strong_positive"
    elif corr <= -0.7:
        return "strong_negative"
    elif corr < -0.3:
        return "negative"
    else:
        return "moderate"


correlation_engine = CorrelationEngine()

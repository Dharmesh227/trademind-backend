"""Sector Heatmap engine — generates visual heatmap data for sector performance.

Fetches sector index data from BhavcopyEngine (allIndices), enriches with
yfinance historical data for weekly/monthly changes, computes momentum scores,
volume trends, relative strength vs NIFTY 50, and top 3 stocks per sector.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Yahoo Finance ticker mapping for sector indices and NIFTY 50
_YAHOO_INDEX_TICKERS = {
    "NIFTY 50": "^NSEI",
    "NIFTY BANK": "^NSEBANK",
    "NIFTY IT": "^CNXIT",
    "NIFTY PHARMA": "^CNXPHARMA",
    "NIFTY AUTO": "^CNXAUTO",
    "NIFTY FMCG": "^CNXFMCG",
    "NIFTY METAL": "^CNXMETAL",
    "NIFTY REALTY": "^CNXREALTY",
    "NIFTY ENERGY": "^CNXENERGY",
    "NIFTY MEDIA": "^CNXMEDIA",
    "NIFTY INFRA": "^CNXINFRA",
    "NIFTY PSU": "^CNXPSU",
}

SECTOR_INDICES = {
    "NIFTY BANK": {"name": "Banking", "color": "#2196F3"},
    "NIFTY IT": {"name": "IT", "color": "#9C27B0"},
    "NIFTY PHARMA": {"name": "Pharma", "color": "#4CAF50"},
    "NIFTY AUTO": {"name": "Auto", "color": "#FF9800"},
    "NIFTY FMCG": {"name": "FMCG", "color": "#E91E63"},
    "NIFTY METAL": {"name": "Metal", "color": "#795548"},
    "NIFTY REALTY": {"name": "Realty", "color": "#607D8B"},
    "NIFTY ENERGY": {"name": "Energy", "color": "#FF5722"},
    "NIFTY MEDIA": {"name": "Media", "color": "#00BCD4"},
    "NIFTY INFRA": {"name": "Infrastructure", "color": "#FFC107"},
    "NIFTY PSU": {"name": "PSU", "color": "#3F51B5"},
}

# Top constituents per sector for "top 3 stocks" display
SECTOR_STOCKS = {
    "NIFTY BANK": ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "SBIN", "AXISBANK"],
    "NIFTY IT": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM"],
    "NIFTY PHARMA": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB"],
    "NIFTY AUTO": ["MARUTI", "M&M", "TATAMOTORS", "EICHERMOT", "HEROMOTOCO", "BAJAJ-AUTO"],
    "NIFTY FMCG": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA"],
    "NIFTY METAL": ["TATASTEEL", "JSWSTEEL", "HINDALCO"],
    "NIFTY REALTY": ["DLF", "GODREJPROP", "OBEROIRLTY", "PRESTIGE"],
    "NIFTY ENERGY": ["RELIANCE", "ONGC", "NTPC", "POWERGRID", "BPCL", "IOC"],
    "NIFTY MEDIA": ["ZEEL", "SUNTV", "PVRINOX", "TV18BRDCST"],
    "NIFTY INFRA": ["LT", "ADANIPORTS", "ADANIENT", "UltraTech"],
    "NIFTY PSU": ["SBIN", "COALINDIA", "NTPC", "POWERGRID", "ONGC"],
}

_MOMENTUM_WEIGHTS = {"1d": 0.50, "1w": 0.30, "1m": 0.20}


@dataclass
class SectorHeatCell:
    """Single cell in the sector heatmap."""
    sector_index: str
    sector_name: str
    color: str
    change_1d: float = 0.0
    change_1w: float = 0.0
    change_1m: float = 0.0
    momentum_score: float = 0.0
    heat_value: float = 0.0
    volume_trend: str = "stable"
    relative_strength: float = 0.0
    top_stocks: List[Dict[str, float]] = field(default_factory=list)
    current_price: float = 0.0


@dataclass
class HeatmapResult:
    """Complete heatmap data structure."""
    timestamp: str = ""
    sectors: List[SectorHeatCell] = field(default_factory=list)
    nifty50_change_1d: float = 0.0
    nifty50_change_1w: float = 0.0
    nifty50_change_1m: float = 0.0
    generated_at: str = ""


class SectorHeatmapEngine:
    """Generates sector heatmap data by combining BhavcopyEngine index data
    with yfinance historical prices."""

    def __init__(self) -> None:
        self._cache: Optional[HeatmapResult] = None
        self._cache_time: float = 0.0
        self._cache_ttl: float = 900.0  # 15 minutes

    async def generate_heatmap(self) -> HeatmapResult:
        """Generate or return cached heatmap."""
        if self._cache and (time.time() - self._cache_time) < self._cache_ttl:
            return self._cache

        from trademind.engines.bhavcopy.engine import BhavcopyEngine

        bhavcopy = BhavcopyEngine()
        data = await bhavcopy.get_bhavcopy()

        # Collect all tickers we need from yfinance (sector indices + NIFTY 50)
        tickers_needed = list(_YAHOO_INDEX_TICKERS.values())
        stock_tickers = set()
        for stocks in SECTOR_STOCKS.values():
            for s in stocks:
                stock_tickers.add(f"{s}.NS")
        tickers_needed.extend(stock_tickers)

        # Fetch historical data from yfinance in a thread
        history_map = await asyncio.to_thread(
            self._fetch_yfinance_history, tickers_needed
        )

        # Compute NIFTY 50 changes as baseline
        nifty_ticker = _YAHOO_INDEX_TICKERS.get("NIFTY 50", "^NSEI")
        nifty_closes = history_map.get(nifty_ticker, [])
        nifty_1d = self._pct_change(nifty_closes, 1)
        nifty_1w = self._pct_change(nifty_closes, 5)
        nifty_1m = self._pct_change(nifty_closes, 22)

        sectors: List[SectorHeatCell] = []

        for idx_name, meta in SECTOR_INDICES.items():
            cell = SectorHeatCell(
                sector_index=idx_name,
                sector_name=meta["name"],
                color=meta["color"],
            )

            # Use BhavcopyEngine current price if available
            idx_data = data.indices.get(idx_name)
            if idx_data:
                cell.current_price = idx_data.last
                cell.change_1d = round(idx_data.change_pct, 2)

            # Override / supplement with yfinance historical
            yahoo_ticker = _YAHOO_INDEX_TICKERS.get(idx_name)
            closes = history_map.get(yahoo_ticker, [])

            if closes:
                cell.change_1w = round(self._pct_change(closes, 5), 2)
                cell.change_1m = round(self._pct_change(closes, 22), 2)

                # If Bhavcopy didn't have daily change, use yfinance
                if not idx_data:
                    cell.change_1d = round(self._pct_change(closes, 1), 2)
                    if closes:
                        cell.current_price = round(closes[-1], 2)

            # Momentum score: weighted combo of recent performance
            cell.momentum_score = round(
                self._compute_momentum(cell.change_1d, cell.change_1w, cell.change_1m), 1
            )

            # Heat value: -1 (deep red) to +1 (deep green)
            cell.heat_value = round(
                max(-1.0, min(1.0, cell.momentum_score / 50.0 - 1.0)), 3
            )

            # Volume trend
            cell.volume_trend = self._volume_trend_from_closes(closes)

            # Relative strength vs NIFTY 50
            if nifty_1m != 0:
                cell.relative_strength = round(
                    self._compute_relative_strength(cell.change_1m, nifty_1m), 2
                )
            else:
                cell.relative_strength = 0.0

            # Top 3 stocks in sector
            sector_stock_syms = SECTOR_STOCKS.get(idx_name, [])
            cell.top_stocks = await self._get_top_sector_stocks(
                sector_stock_syms, history_map
            )

            sectors.append(cell)

        # Sort by momentum descending
        sectors.sort(key=lambda s: s.momentum_score, reverse=True)

        now = datetime.now()
        result = HeatmapResult(
            timestamp=now.isoformat(),
            sectors=sectors,
            nifty50_change_1d=round(nifty_1d, 2),
            nifty50_change_1w=round(nifty_1w, 2),
            nifty50_change_1m=round(nifty_1m, 2),
            generated_at=now.strftime("%Y-%m-%d %H:%M:%S"),
        )

        self._cache = result
        self._cache_time = time.time()
        return result

    async def get_top_sectors(self, n: int = 3) -> List[SectorHeatCell]:
        """Return top N performing sectors by momentum."""
        heatmap = await self.generate_heatmap()
        return heatmap.sectors[:n]

    async def get_worst_sectors(self, n: int = 3) -> List[SectorHeatCell]:
        """Return bottom N performing sectors by momentum."""
        heatmap = await self.generate_heatmap()
        return list(reversed(heatmap.sectors))[:n]

    # ── Private helpers ────────────────────────────────────────────

    @staticmethod
    def _fetch_yfinance_history(
        tickers: List[str],
    ) -> Dict[str, List[float]]:
        """Download 1 month of daily closes for a list of tickers.

        Runs in a thread via asyncio.to_thread to avoid blocking the event loop.
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance not installed — heatmap will use Bhavcopy data only")
            return {}

        result: Dict[str, List[float]] = {}

        try:
            data = yf.download(
                tickers,
                period="1mo",
                interval="1d",
                progress=False,
                threads=True,
                group_by="ticker",
            )

            if data is None or data.empty:
                return {}

            multiple = len(tickers) > 1

            for ticker in tickers:
                try:
                    if multiple:
                        df = data[ticker] if ticker in data.columns.get_level_values(0) else None
                    else:
                        df = data

                    if df is None or df.empty:
                        continue

                    closes_raw = df["Close"].dropna().tolist()
                    closes = [float(c) for c in closes_raw if float(c) > 0]
                    if closes:
                        result[ticker] = closes
                except Exception:
                    continue

        except Exception as exc:
            logger.warning("yfinance download failed for heatmap: %s", exc)

        return result

    @staticmethod
    def _pct_change(closes: List[float], lookback: int) -> float:
        """Compute percentage change over the last `lookback` bars."""
        if not closes or len(closes) < lookback + 1:
            if closes and len(closes) >= 2:
                return ((closes[-1] - closes[0]) / closes[0]) * 100
            return 0.0
        old = closes[-(lookback + 1)]
        new = closes[-1]
        if old <= 0:
            return 0.0
        return ((new - old) / old) * 100

    @staticmethod
    def _compute_momentum(change_1d: float, change_1w: float, change_1m: float) -> float:
        """Weighted momentum score 0-100.

        Maps combined change to 0-100 scale where 50 = flat.
        Weights: 1d=50%, 1w=30%, 1m=20%.
        """
        combined = (
            change_1d * _MOMENTUM_WEIGHTS["1d"]
            + change_1w * _MOMENTUM_WEIGHTS["1w"]
            + change_1m * _MOMENTUM_WEIGHTS["1m"]
        )
        # Map: -5% -> 0, 0% -> 50, +5% -> 100
        score = 50.0 + (combined / 5.0) * 50.0
        return max(0.0, min(100.0, score))

    @staticmethod
    def _volume_trend_from_closes(closes: List[float]) -> str:
        """Infer volume trend from close price trajectory (proxy when volume unavailable).

        Uses simple linear slope of recent closes as a proxy for conviction.
        """
        if len(closes) < 5:
            return "stable"

        recent = closes[-5:]
        first_half = sum(recent[:2]) / 2
        second_half = sum(recent[3:]) / 2
        pct_diff = ((second_half - first_half) / first_half * 100) if first_half > 0 else 0

        if pct_diff > 1.5:
            return "increasing"
        elif pct_diff < -1.5:
            return "decreasing"
        return "stable"

    @staticmethod
    def _compute_relative_strength(sector_1m: float, nifty_1m: float) -> float:
        """Relative strength = sector return - NIFTY 50 return over 1 month.

        Positive means sector outperformed the benchmark.
        """
        return sector_1m - nifty_1m

    @staticmethod
    async def _get_top_sector_stocks(
        stock_symbols: List[str],
        history_map: Dict[str, List[float]],
    ) -> List[Dict[str, float]]:
        """Return top 3 performing stocks in a sector based on 1-week return."""
        performances: List[Dict[str, float]] = []

        for sym in stock_symbols:
            ticker = f"{sym}.NS"
            closes = history_map.get(ticker, [])
            if not closes or len(closes) < 2:
                performances.append({"symbol": sym, "change_1w": 0.0})
                continue

            change = ((closes[-1] - closes[0]) / closes[0]) * 100 if closes[0] > 0 else 0
            performances.append({"symbol": sym, "change_1w": round(change, 2)})

        performances.sort(key=lambda x: x["change_1w"], reverse=True)
        return performances[:3]

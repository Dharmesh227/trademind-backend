"""Yahoo Finance data provider — delayed (15 min) live data during market hours.

Used as fallback when Bhavcopy EOD data isn't yet available (before 8 PM IST).
Fetches all F&O symbols + major indices in batch using yfinance.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import yfinance as yf

from trademind.engines.bhavcopy.engine import IndexData, StockData

logger = logging.getLogger(__name__)

# NSE symbol -> Yahoo Finance symbol suffix
_NS_SUFFIX = ".NS"

# Major NSE indices on Yahoo
_YAHOO_INDICES = {
    "^NSEI": "NIFTY 50",
    "^NSEBANK": "NIFTY BANK",
    "^CNXIT": "NIFTY IT",
    "^CNXAUTO": "NIFTY AUTO",
    "^CNXFMCG": "NIFTY FMCG",
    "^CNXPHARMA": "NIFTY PHARMA",
    "^CNXMETAL": "NIFTY METAL",
    "^CNXREALTY": "NIFTY REALTY",
}


class YahooProvider:
    """Fetches delayed stock + index data from Yahoo Finance."""

    def __init__(self) -> None:
        self._cache_stocks: Dict[str, StockData] = {}
        self._cache_indices: Dict[str, IndexData] = {}
        self._cache_time: float = 0.0
        self._cache_ttl: float = 300.0  # 5 min
        self._last_fetch_count: int = 0

    @property
    def is_fresh(self) -> bool:
        return bool(self._cache_stocks) and (time.time() - self._cache_time) < self._cache_ttl

    async def fetch_all(
        self, fno_symbols: List[str], index_symbols: List[str]
    ) -> tuple[Dict[str, StockData], Dict[str, IndexData]]:
        """Fetch all stocks + indices from Yahoo Finance.

        Returns (stocks, indices) in the same format as BhavcopyEngine.
        """
        if self.is_fresh:
            return self._cache_stocks, self._cache_indices

        try:
            stocks = await self._fetch_stocks(fno_symbols)
            indices = await self._fetch_indices(index_symbols)
            self._cache_stocks = stocks
            self._cache_indices = indices
            self._cache_time = time.time()
            self._last_fetch_count = len(stocks) + len(indices)
            logger.info(
                "Yahoo Finance: %d stocks, %d indices fetched",
                len(stocks), len(indices),
            )
            return stocks, indices
        except Exception as exc:
            logger.error("Yahoo Finance fetch failed: %s", exc)
            return self._cache_stocks, self._cache_indices

    async def _fetch_stocks(self, symbols: List[str]) -> Dict[str, StockData]:
        """Batch-fetch stock data from Yahoo."""
        if not symbols:
            return {}

        yahoo_symbols = [f"{s}{_NS_SUFFIX}" for s in symbols]

        try:
            data = yf.download(
                yahoo_symbols,
                period="5d",
                interval="1d",
                group_by="ticker",
                progress=False,
                threads=True,
            )

            stocks: Dict[str, StockData] = {}
            multiple = len(yahoo_symbols) > 1

            for sym in symbols:
                yahoo_sym = f"{sym}{_NS_SUFFIX}"
                try:
                    df = data[yahoo_sym] if multiple else data
                    if df is None or df.empty or len(df) < 1:
                        continue

                    last = df.iloc[-1]
                    prev = df.iloc[-2] if len(df) > 1 else last

                    close = float(last["Close"])
                    prev_close = float(prev["Close"])
                    change = close - prev_close
                    change_pct = (change / prev_close * 100) if prev_close > 0 else 0.0
                    volume = int(last.get("Volume", 0) or 0)

                    stocks[sym] = StockData(
                        symbol=sym,
                        close=close,
                        prev_close=prev_close,
                        change=round(change, 2),
                        change_pct=round(change_pct, 2),
                        open=float(last.get("Open", 0) or 0),
                        high=float(last.get("High", 0) or 0),
                        low=float(last.get("Low", 0) or 0),
                        volume=volume,
                        oi=0,
                        turnover=0,
                        settlement=close,
                    )
                except Exception:
                    continue

            return stocks

        except Exception as exc:
            logger.warning("Yahoo stock fetch failed: %s", exc)
            return {}

    async def _fetch_indices(self, index_symbols: List[str]) -> Dict[str, IndexData]:
        """Fetch major indices from Yahoo."""
        yahoo_indices = list(_YAHOO_INDICES.keys())
        if not yahoo_indices:
            return {}

        try:
            data = yf.download(
                yahoo_indices,
                period="5d",
                interval="1d",
                group_by="ticker",
                progress=False,
                threads=True,
            )

            indices: Dict[str, IndexData] = {}
            multiple = len(yahoo_indices) > 1

            for yahoo_sym, nse_name in _YAHOO_INDICES.items():
                if nse_name not in index_symbols:
                    continue
                try:
                    df = data[yahoo_sym] if multiple else data
                    if df is None or df.empty or len(df) < 1:
                        continue

                    last = df.iloc[-1]
                    prev = df.iloc[-2] if len(df) > 1 else last

                    close = float(last["Close"])
                    prev_close = float(prev["Close"])
                    change_pct = ((close - prev_close) / prev_close * 100) if prev_close > 0 else 0.0
                    volume = int(last.get("Volume", 0) or 0)

                    # Yahoo doesn't provide PE/PB/DY for indices — leave 0
                    indices[nse_name] = IndexData(
                        symbol=nse_name,
                        last=close,
                        open=float(last.get("Open", 0) or 0),
                        high=float(last.get("High", 0) or 0),
                        low=float(last.get("Low", 0) or 0),
                        prev_close=prev_close,
                        change_pct=round(change_pct, 2),
                        pe=0,
                        pb=0,
                        dy=0,
                        advances=0,
                        declines=0,
                        unchanged=0,
                        per_change_30d=0,
                        per_change_365d=0,
                        year_high=0,
                        year_low=0,
                        traded_volume=volume,
                    )
                except Exception:
                    continue

            return indices

        except Exception as exc:
            logger.warning("Yahoo index fetch failed: %s", exc)
            return {}


# Module-level singleton
yahoo_provider = YahooProvider()

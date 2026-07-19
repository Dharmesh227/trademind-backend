"""Yahoo Finance historical data adapter for FeatureExtractor.

Fetches 6 months of daily OHLCV from Yahoo Finance and converts it to
PriceData objects that FeatureExtractor can use to compute 100+ technical
features (RSI, MACD, EMA, VWAP, ADX, Bollinger, Supertrend, etc.).

This bridges the gap between Yahoo's delayed data and the full
AIScoreEngine scoring pipeline.
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

import yfinance as yf

from trademind.database.models import PriceData

logger = logging.getLogger(__name__)


class YahooHistoricalAdapter:
    """Fetches historical OHLCV from Yahoo and feeds FeatureExtractor."""

    def __init__(self) -> None:
        self._cache: Dict[str, List[PriceData]] = {}
        self._cache_date: Optional[str] = None

    async def populate_history(
        self, symbols: List[str], feature_extractor
    ) -> int:
        """Fetch 6-month history for all symbols and populate FeatureExtractor.

        Returns the number of symbols with data populated.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        if self._cache_date == today and self._cache:
            # Already populated today — just feed cached data
            for sym, prices in self._cache.items():
                feature_extractor._price_history[sym] = list(prices)
            return len(self._cache)

        yahoo_syms = [f"{s}.NS" for s in symbols]
        logger.info("Fetching 6-month history for %d symbols from Yahoo...", len(symbols))

        try:
            data = yf.download(
                yahoo_syms,
                period="6mo",
                interval="1d",
                group_by="ticker",
                progress=False,
                threads=True,
            )
        except Exception as exc:
            logger.error("Yahoo history download failed: %s", exc)
            return 0

        populated = 0
        multiple = len(yahoo_syms) > 1

        for sym in symbols:
            yahoo_sym = f"{sym}.NS"
            try:
                df = data[yahoo_sym] if multiple else data
                if df is None or df.empty or len(df) < 20:
                    continue

                prices: List[PriceData] = []
                closes = df["Close"].values
                highs = df["High"].values
                lows = df["Low"].values
                opens = df["Open"].values
                volumes = df["Volume"].values

                for i in range(len(df)):
                    try:
                        close_val = float(closes[i])
                        high_val = float(highs[i])
                        low_val = float(lows[i])
                        open_val = float(opens[i])
                        vol_val = int(volumes[i]) if not (volumes[i] != volumes[i]) else 0  # NaN check

                        if close_val <= 0:
                            continue

                        prev_close = float(closes[i - 1]) if i > 0 else close_val
                        change_pct = ((close_val - prev_close) / prev_close * 100) if prev_close > 0 else 0

                        prices.append(PriceData(
                            symbol=sym,
                            timestamp=datetime.now(),
                            open=Decimal(str(round(open_val, 2))),
                            high=Decimal(str(round(high_val, 2))),
                            low=Decimal(str(round(low_val, 2))),
                            close=Decimal(str(round(close_val, 2))),
                            volume=vol_val,
                            vwap=Decimal(str(round((high_val + low_val + close_val) / 3, 2))),
                            prev_close=Decimal(str(round(prev_close, 2))),
                            change_percent=Decimal(str(round(change_pct, 2))),
                        ))
                    except Exception:
                        continue

                if prices:
                    feature_extractor._price_history[sym] = prices
                    self._cache[sym] = prices
                    populated += 1

            except Exception:
                continue

        self._cache_date = today

        for sym in symbols:
            hist = feature_extractor._price_history.get(sym)
            if not hist or len(hist) < 2:
                continue
            cum_tp_vol = 0.0
            cum_vol = 0
            for p in hist:
                tp = (float(p.high) + float(p.low) + float(p.close)) / 3
                cum_tp_vol += tp * p.volume
                cum_vol += p.volume
                p.vwap = Decimal(str(round(cum_tp_vol / cum_vol, 2))) if cum_vol > 0 else p.close
        logger.info(
            "Yahoo history populated: %d/%d symbols (%d avg bars)",
            populated, len(symbols),
            sum(len(v) for v in self._cache.values()) // max(populated, 1),
        )
        return populated


# Module-level singleton
yahoo_history = YahooHistoricalAdapter()

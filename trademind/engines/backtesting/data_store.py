"""Historical data store — download and cache daily OHLCV for backtesting."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import yfinance as yf

from trademind.database.models import PriceData
from trademind.config.settings import settings as cfg
from decimal import Decimal

logger = logging.getLogger(__name__)


class HistoricalDataStore:
    """Downloads and caches daily OHLCV history for backtesting.

    Uses yfinance to fetch historical data and caches results in memory.
    """

    _cache: Dict[str, List[PriceData]] = {}

    async def download_history(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
    ) -> Dict[str, List[PriceData]]:
        """Download daily OHLCV for all symbols between start and end dates."""
        results: Dict[str, List[PriceData]] = {}

        missing = [s for s in symbols if self._cache_key(s, start_date, end_date) not in self._cache]
        if missing:
            await self._batch_download(missing, start_date, end_date)

        for symbol in symbols:
            key = self._cache_key(symbol, start_date, end_date)
            if key in self._cache:
                results[symbol] = self._cache[key]

        return results

    async def _batch_download(
        self, symbols: List[str], start_date: str, end_date: str
    ) -> None:
        """Batch download using yf.download for efficiency."""
        try:
            tickers = " ".join(f"{s}.NS" for s in symbols)
            data = yf.download(
                tickers=tickers,
                start=start_date,
                end=end_date,
                interval="1d",
                group_by="ticker",
                threads=True,
                progress=False,
            )

            if data is None or data.empty:
                logger.warning("No data returned for %d symbols", len(symbols))
                return

            for symbol in symbols:
                ticker = f"{symbol}.NS"
                try:
                    if len(symbols) == 1:
                        df = data
                    else:
                        df = data[ticker]

                    if df is None or df.empty:
                        continue

                    if hasattr(df.columns, 'levels') and len(df.columns.levels) > 1:
                        for lvl_idx in range(len(df.columns.levels)):
                            vals = {str(v) for v in df.columns.levels[lvl_idx]}
                            if vals & {"Open", "High", "Low", "Close", "Volume"}:
                                df = df.copy()
                                df.columns = df.columns.get_level_values(lvl_idx)
                                break

                    prices = []
                    for date_idx, row in df.iterrows():
                        try:
                            ts = date_idx.to_pydatetime().replace(tzinfo=None)
                            open_val = float(row.get("Open", 0))
                            high_val = float(row.get("High", 0))
                            low_val = float(row.get("Low", 0))
                            close_val = float(row.get("Close", 0))
                            vol = int(row.get("Volume", 0))

                            if close_val <= 0:
                                continue

                            vwap = round((high_val + low_val + close_val) / 3, 2)

                            prices.append(PriceData(
                                symbol=symbol,
                                timestamp=ts,
                                open=Decimal(str(open_val)),
                                high=Decimal(str(high_val)),
                                low=Decimal(str(low_val)),
                                close=Decimal(str(close_val)),
                                volume=vol,
                                vwap=Decimal(str(vwap)),
                                prev_close=None,
                                change_percent=None,
                            ))
                        except Exception:
                            continue

                    if prices:
                        prices.sort(key=lambda p: p.timestamp)
                        self._cache[self._cache_key(symbol, start_date, end_date)] = prices
                        logger.info("Cached %d bars for %s", len(prices), symbol)

                except Exception as e:
                    logger.warning("Failed to parse data for %s: %s", symbol, e)

        except Exception as e:
            logger.error("Batch download failed: %s", e)

    def get_cached_history(
        self, symbol: str, start_date: str, end_date: str
    ) -> List[PriceData]:
        """Serve from cache."""
        key = self._cache_key(symbol, start_date, end_date)
        return self._cache.get(key, [])

    def get_bars_for_date(
        self, symbol: str, date: str, all_data: Dict[str, List[PriceData]]
    ) -> Optional[PriceData]:
        """Get a single bar for a symbol on a specific date."""
        prices = all_data.get(symbol, [])
        target = datetime.strptime(date, "%Y-%m-%d")
        for p in prices:
            if p.timestamp.date() == target.date():
                return p
        return None

    @staticmethod
    def _cache_key(symbol: str, start: str, end: str) -> str:
        return f"{symbol}_{start}_{end}"

    def clear_cache(self) -> None:
        self._cache.clear()


historical_data_store = HistoricalDataStore()

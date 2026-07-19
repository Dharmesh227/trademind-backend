"""Timeframe aggregator — fetch and run feature extraction per timeframe."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import yfinance as yf

from trademind.database.models import PriceData
from trademind.engines.feature_engine.extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class TimeframeAggregator:
    """Fetches OHLCV at multiple intervals and computes features per timeframe."""

    TIMEFRAMES = {
        "daily": {"interval": "1d", "period": "6mo"},
        "hourly": {"interval": "1h", "period": "60d"},
        "15min": {"interval": "15m", "period": "60d"},
    }

    async def fetch_all_timeframes(
        self, symbols: List[str]
    ) -> Dict[str, Dict[str, List[PriceData]]]:
        """Fetch OHLCV for all symbols at all timeframes.

        Returns: {symbol: {timeframe: [PriceData]}}
        """
        results: Dict[str, Dict[str, List[PriceData]]] = {}

        for tf_name, tf_config in self.TIMEFRAMES.items():
            tf_data = await self._fetch_timeframe(
                symbols, tf_config["interval"], tf_config["period"]
            )
            for symbol, prices in tf_data.items():
                if symbol not in results:
                    results[symbol] = {}
                results[symbol][tf_name] = prices

        return results

    async def _fetch_timeframe(
        self, symbols: List[str], interval: str, period: str
    ) -> Dict[str, List[PriceData]]:
        """Fetch data for one timeframe across all symbols."""
        results: Dict[str, List[PriceData]] = {}

        try:
            tickers = " ".join(f"{s}.NS" for s in symbols)
            data = yf.download(
                tickers=tickers,
                period=period,
                interval=interval,
                group_by="ticker",
                threads=True,
                progress=False,
            )

            if data is None or data.empty:
                return results

            for symbol in symbols:
                ticker = f"{symbol}.NS"
                try:
                    if len(symbols) == 1:
                        df = data
                    else:
                        df = data[ticker]

                    if df is None or df.empty:
                        continue

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
                                open=open_val,
                                high=high_val,
                                low=low_val,
                                close=close_val,
                                volume=vol,
                                vwap=vwap,
                            ))
                        except Exception:
                            continue

                    if prices:
                        prices.sort(key=lambda p: p.timestamp)
                        results[symbol] = prices

                except Exception as e:
                    logger.debug("Parse failed for %s (%s): %s", symbol, interval, e)

        except Exception as e:
            logger.warning("Fetch failed for interval=%s: %s", interval, e)

        return results

    def extract_features_per_timeframe(
        self,
        symbol: str,
        prices_by_tf: Dict[str, List[PriceData]],
    ) -> Dict[str, Dict[str, float]]:
        """Run feature extraction for each timeframe independently.

        Returns: {timeframe: {feature_name: value}}
        """
        features_by_tf: Dict[str, Dict[str, float]] = {}

        for tf_name, prices in prices_by_tf.items():
            if not prices or len(prices) < 20:
                continue

            extractor = FeatureExtractor()
            extractor._price_history[symbol] = prices

            current = prices[-1]
            try:
                features = extractor.compute_all_features(
                    symbol=symbol,
                    market_data=current,
                    option_data=None,
                )
                features_by_tf[tf_name] = features
            except Exception as e:
                logger.debug("Feature extraction failed for %s/%s: %s", symbol, tf_name, e)

        return features_by_tf


timeframe_aggregator = TimeframeAggregator()

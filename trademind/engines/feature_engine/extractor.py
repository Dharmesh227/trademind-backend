import logging
from datetime import datetime
from decimal import Decimal
from math import sqrt
from typing import Dict, List, Optional

import numpy as np

from trademind.database.models import (
    AIScoreResult,
    IndexData,
    InstitutionalFlowData,
    MarketBreadthData,
    OptionChainData,
    OptionStrike,
    PriceData,
    VIXData,
)

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extracts 100+ normalized features from raw market data.

    All features are normalized to 0-100 scale for consistent scoring.
    Features span technical, volume, options, momentum, sector, market,
    volatility, and quality categories.
    """

    def __init__(self) -> None:
        self._price_history: Dict[str, List[PriceData]] = {}
        self._max_history: int = 250

    def _clamp_normalize(self, value: float, min_val: float, max_val: float) -> float:
        """Clamp and normalize a value to 0-100 scale."""
        if max_val <= min_val:
            return 50.0
        clamped = max(min_val, min(max_val, value))
        return round((clamped - min_val) / (max_val - min_val) * 100, 2)

    def _safe_float(self, value, default: float = 0.0) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def _safe_decimal_float(self, value) -> float:
        if value is None:
            return 0.0
        try:
            return float(str(value))
        except (ValueError, TypeError):
            return 0.0

    # ------------------------------------------------------------------
    # Technical Features
    # ------------------------------------------------------------------

    def _ema(
        self, prices: List[float], period: int
    ) -> List[float]:
        if not prices:
            return []
        ema: List[float] = []
        multiplier = 2.0 / (period + 1)
        for i, price in enumerate(prices):
            if i == 0:
                ema.append(price)
            else:
                ema.append((price - ema[i - 1]) * multiplier + ema[i - 1])
        return ema

    def _sma(self, prices: List[float], period: int) -> List[float]:
        if len(prices) < period:
            return []
        sma_vals = []
        for i in range(len(prices)):
            if i < period - 1:
                sma_vals.append(0.0)
            else:
                sma_vals.append(sum(prices[i - period + 1 : i + 1]) / period)
        return sma_vals

    def _rsi(self, prices: List[float], period: int = 14) -> List[float]:
        if len(prices) < period + 1:
            return [50.0] * len(prices)
        rsi_vals: List[float] = [50.0] * period
        gains = []
        losses = []
        for i in range(1, period + 1):
            diff = prices[i] - prices[i - 1]
            gains.append(max(diff, 0))
            losses.append(max(-diff, 0))
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        for i in range(period, len(prices)):
            diff = prices[i] - prices[i - 1]
            gain = max(diff, 0)
            loss = max(-diff, 0)
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period
            if avg_loss == 0:
                rs = 100.0
            else:
                rs = avg_gain / avg_loss
            rsi_vals.append(100.0 - 100.0 / (1.0 + rs))
        return rsi_vals

    def _macd(
        self, prices: List[float]
    ) -> Dict[str, List[float]]:
        ema12 = self._ema(prices, 12)
        ema26 = self._ema(prices, 26)
        macd_line = [
            e12 - e26 if e12 != 0 and e26 != 0 else 0.0
            for e12, e26 in zip(ema12, ema26)
        ]
        signal = self._ema(macd_line, 9)
        histogram = [m - s for m, s in zip(macd_line, signal)]
        return {
            "macd": macd_line,
            "signal": signal,
            "histogram": histogram,
        }

    def _bollinger_bands(
        self, prices: List[float], period: int = 20
    ) -> Dict[str, List[float]]:
        sma = self._sma(prices, period)
        bb_upper: List[float] = []
        bb_lower: List[float] = []
        bb_width: List[float] = []
        bb_percent: List[float] = []
        for i in range(len(prices)):
            if i < period - 1 or sma[i] == 0:
                bb_upper.append(0.0)
                bb_lower.append(0.0)
                bb_width.append(0.0)
                bb_percent.append(50.0)
                continue
            window = prices[i - period + 1 : i + 1]
            std = float(np.std(window))
            upper = sma[i] + 2 * std
            lower = sma[i] - 2 * std
            band_width = upper - lower if upper > lower else 0.0
            pct_b = 50.0
            if band_width > 0:
                pct_b = (prices[i] - lower) / band_width * 100
            bb_upper.append(upper)
            bb_lower.append(lower)
            bb_width.append(band_width)
            bb_percent.append(pct_b)
        return {
            "upper": bb_upper,
            "lower": bb_lower,
            "width": bb_width,
            "percent_b": bb_percent,
        }

    def _atr(
        self, high: List[float], low: List[float], close: List[float], period: int = 14
    ) -> List[float]:
        if len(close) < 2:
            return [0.0] * len(close)
        tr_vals: List[float] = [0.0]
        for i in range(1, len(close)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr_vals.append(max(hl, hc, lc))
        atr = [0.0] * (period - 1)
        atr.append(sum(tr_vals[1 : period + 1]) / period)
        for i in range(period, len(tr_vals)):
            atr.append((atr[-1] * (period - 1) + tr_vals[i]) / period)
        return atr

    def _adx(
        self,
        high: List[float],
        low: List[float],
        close: List[float],
        period: int = 14,
    ) -> Dict[str, List[float]]:
        n = len(close)
        plus_dm: List[float] = [0.0] * n
        minus_dm: List[float] = [0.0] * n
        tr_vals: List[float] = [0.0] * n

        for i in range(1, n):
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            else:
                plus_dm[i] = 0.0
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
            else:
                minus_dm[i] = 0.0
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr_vals[i] = max(hl, hc, lc)

        atr_val = sum(tr_vals[1 : period + 1]) / period
        sum_plus = sum(plus_dm[1 : period + 1])
        sum_minus = sum(minus_dm[1 : period + 1])

        plus_di: List[float] = [0.0] * (period)
        minus_di: List[float] = [0.0] * (period)
        adx_vals: List[float] = [0.0] * (period)

        if atr_val > 0:
            plus_di.append((sum_plus / atr_val) * 100)
            minus_di.append((sum_minus / atr_val) * 100)
        else:
            plus_di.append(0.0)
            minus_di.append(0.0)

        for i in range(period + 1, n):
            atr_val = (atr_val * (period - 1) + tr_vals[i]) / period
            sum_plus = (sum_plus * (period - 1) + plus_dm[i]) / period
            sum_minus = (sum_minus * (period - 1) + minus_dm[i]) / period
            if atr_val > 0:
                plus_di.append((sum_plus / atr_val) * 100)
                minus_di.append((sum_minus / atr_val) * 100)
            else:
                plus_di.append(0.0)
                minus_di.append(0.0)

        for i in range(len(plus_di)):
            dx = 0.0
            sum_di = plus_di[i] + minus_di[i]
            if sum_di > 0:
                dx = abs(plus_di[i] - minus_di[i]) / sum_di * 100
            if i < period * 2 - 1:
                adx_vals.append(0.0)
            elif i == period * 2 - 1:
                adx_vals.append(dx)
            else:
                smoothed = (adx_vals[-1] * (period - 1) + dx) / period
                adx_vals.append(smoothed)

        while len(adx_vals) < n:
            adx_vals.append(0.0)

        return {
            "adx": adx_vals[:n],
            "plus_di": plus_di[:n],
            "minus_di": minus_di[:n],
        }

    # ------------------------------------------------------------------
    # Technical Feature Extraction
    # ------------------------------------------------------------------

    def _extract_ema_features(self, close: List[float]) -> Dict[str, float]:
        features: Dict[str, float] = {}
        if len(close) < 5:
            return self._default_ema_features()

        ema5 = self._ema(close, 5)
        ema10 = self._ema(close, 10)
        ema20 = self._ema(close, 20)
        ema50 = self._ema(close, 50)
        ema200 = self._ema(close, 200)

        cp = close[-1]

        features["ema_price_vs_ema5"] = self._clamp_normalize(
            (cp - ema5[-1]) / cp * 100, -5, 5
        )
        features["ema_price_vs_ema20"] = self._clamp_normalize(
            (cp - ema20[-1]) / cp * 100 if ema20[-1] else 0, -8, 8
        )
        features["ema_price_vs_ema50"] = self._clamp_normalize(
            (cp - ema50[-1]) / cp * 100 if ema50[-1] else 0, -10, 10
        )

        ema_alignment = 0
        if ema5[-1] > ema10[-1] > ema20[-1]:
            ema_alignment = 1
        elif ema5[-1] < ema10[-1] < ema20[-1]:
            ema_alignment = -1
        features["ema_alignment"] = self._clamp_normalize(ema_alignment, -1, 1)

        features["ema_alignment_score"] = self._clamp_normalize(
            ema_alignment * (
                1 if (len(ema50) >= 2 and cp > ema50[-1]) else -1
            ) * 50, -50, 50
        )

        bull_count = sum([
            cp > ema5[-1] if ema5[-1] else False,
            cp > ema10[-1] if ema10[-1] else False,
            cp > ema20[-1] if ema20[-1] else False,
            cp > ema50[-1] if len(ema50) >= 2 else False,
            cp > ema200[-1] if len(ema200) >= 2 else False,
        ])
        features["ema_bull_count"] = self._clamp_normalize(bull_count, 0, 5)

        features["ema5_slope"] = self._clamp_normalize(
            (ema5[-1] - ema5[-3]) / (ema5[-3] or 1) * 100, -2, 2
        )
        features["ema20_slope"] = self._clamp_normalize(
            (ema20[-1] - ema20[-3]) / (ema20[-3] or 1) * 100, -2, 2
        )

        return features

    def _default_ema_features(self) -> Dict[str, float]:
        return {
            "ema_price_vs_ema5": 50.0,
            "ema_price_vs_ema20": 50.0,
            "ema_price_vs_ema50": 50.0,
            "ema_alignment": 50.0,
            "ema_alignment_score": 50.0,
            "ema_bull_count": 50.0,
            "ema5_slope": 50.0,
            "ema20_slope": 50.0,
        }

    def _extract_rsi_features(self, close: List[float]) -> Dict[str, float]:
        if len(close) < 15:
            return {"rsi_14": 50.0, "rsi_slope": 50.0, "rsi_strength": 50.0}

        rsi = self._rsi(close)
        rsi_val = rsi[-1]
        rsi_slope_val = (rsi[-1] - rsi[-5]) if len(rsi) >= 5 else 0

        strength = 50.0
        if rsi_val > 60:
            strength = self._clamp_normalize(rsi_val, 60, 100)
        elif rsi_val < 40:
            strength = self._clamp_normalize(100 - rsi_val, 60, 100)
        else:
            strength = self._clamp_normalize(rsi_val, 40, 60)

        return {
            "rsi_14": self._clamp_normalize(rsi_val, 0, 100),
            "rsi_slope": self._clamp_normalize(rsi_slope_val, -10, 10),
            "rsi_strength": strength,
        }

    def _extract_macd_features(self, close: List[float]) -> Dict[str, float]:
        if len(close) < 26:
            return {
                "macd_value": 50.0,
                "macd_histogram": 50.0,
                "macd_signal": 50.0,
                "macd_crossover": 50.0,
            }

        macd_data = self._macd(close)
        macd_val = macd_data["macd"][-1]
        signal = macd_data["signal"][-1]
        hist = macd_data["histogram"][-1]

        crossover = 0
        if len(macd_data["macd"]) >= 2 and len(macd_data["signal"]) >= 2:
            prev_macd = macd_data["macd"][-2]
            prev_signal = macd_data["signal"][-2]
            if prev_macd < prev_signal and macd_val > signal:
                crossover = 1
            elif prev_macd > prev_signal and macd_val < signal:
                crossover = -1

        macd_strength = self._clamp_normalize(macd_val, -50, 50)
        if abs(macd_val) < 1:
            macd_strength = self._clamp_normalize(macd_val, -1, 1)

        return {
            "macd_value": macd_strength,
            "macd_histogram": self._clamp_normalize(hist, -10, 10),
            "macd_signal": self._clamp_normalize(macd_val - signal, -10, 10),
            "macd_crossover": self._clamp_normalize(crossover, -1, 1),
        }

    def _extract_bb_features(self, close: List[float]) -> Dict[str, float]:
        if len(close) < 20:
            return {
                "bb_position": 50.0,
                "bb_width": 50.0,
                "bb_compression": 50.0,
            }

        bb = self._bollinger_bands(close)
        pct_b = bb["percent_b"][-1]
        width = bb["width"][-1]

        avg_width = sum(bb["width"][-20:]) / 20 if any(bb["width"][-20:]) else 0
        compression = 50.0
        if avg_width > 0:
            compression_ratio = width / avg_width
            compression = self._clamp_normalize(compression_ratio, 0.5, 2.0)

        return {
            "bb_position": self._clamp_normalize(pct_b, 0, 100),
            "bb_width": self._clamp_normalize(width, 0, width * 2) if width else 50.0,
            "bb_compression": compression,
        }

    def _extract_vwap_features(
        self, close: List[float], vwap_val: float
    ) -> Dict[str, float]:
        if vwap_val == 0:
            return {"vwap_distance": 50.0, "vwap_position": 50.0}
        dist = (close[-1] - vwap_val) / vwap_val * 100
        return {
            "vwap_distance": self._clamp_normalize(dist, -3, 3),
            "vwap_position": 100.0 if close[-1] >= vwap_val else 0.0,
        }

    def _extract_atr_features(
        self, high: List[float], low: List[float], close: List[float]
    ) -> Dict[str, float]:
        if len(close) < 15:
            return {"atr_expansion": 50.0, "atr_value": 50.0}
        atr = self._atr(high, low, close)
        atr_val = atr[-1]
        avg_atr_10 = sum(atr[-10:]) / 10 if len(atr) >= 10 else atr_val
        expansion = 50.0
        if avg_atr_10 > 0:
            expansion_ratio = atr_val / avg_atr_10
            expansion = self._clamp_normalize(expansion_ratio, 0.5, 2.0)
        return {
            "atr_expansion": expansion,
            "atr_value": self._clamp_normalize(atr_val / (close[-1] or 1) * 100, 0, 10),
        }

    def _extract_adx_features(
        self, high: List[float], low: List[float], close: List[float]
    ) -> Dict[str, float]:
        if len(close) < 28:
            return {"adx": 50.0, "plus_di": 50.0, "minus_di": 50.0, "di_cross": 50.0}
        adx_data = self._adx(high, low, close)
        adx = adx_data["adx"][-1]
        plus = adx_data["plus_di"][-1]
        minus = adx_data["minus_di"][-1]

        return {
            "adx": self._clamp_normalize(adx, 0, 100),
            "plus_di": self._clamp_normalize(plus, 0, 100),
            "minus_di": self._clamp_normalize(minus, 0, 100),
            "di_cross": 100.0 if plus > minus else 0.0,
        }

    def _extract_supertrend(
        self, high: List[float], low: List[float], close: List[float]
    ) -> Dict[str, float]:
        if len(close) < 10:
            return {"supertrend": 50.0, "supertrend_direction": 50.0}
        period = 10
        multiplier = 3.0
        atr = self._atr(high, low, close, period)
        supertrend_vals: List[float] = []
        direction: List[float] = []
        for i in range(len(close)):
            if i < period:
                supertrend_vals.append(close[i])
                direction.append(1.0)
                continue
            hl_avg = (high[i] + low[i]) / 2
            basic_upper = hl_avg + multiplier * atr[i]
            basic_lower = hl_avg - multiplier * atr[i]
            if i == period:
                prev_supertrend = basic_upper
                prev_dir = 1
            else:
                prev_supertrend = supertrend_vals[-1]
                prev_dir = direction[-1]
            if close[i] > prev_supertrend:
                new_dir = 1
                new_basic = basic_lower
            else:
                new_dir = -1
                new_basic = basic_upper
            if prev_dir == 1 and new_dir == 1:
                new_supertrend = max(new_basic, prev_supertrend)
            elif prev_dir == -1 and new_dir == -1:
                new_supertrend = min(new_basic, prev_supertrend)
            elif new_dir == 1:
                new_supertrend = basic_lower
            else:
                new_supertrend = basic_upper
            supertrend_vals.append(new_supertrend)
            direction.append(float(new_dir))
        return {
            "supertrend": self._clamp_normalize(
                (close[-1] - supertrend_vals[-1]) / (supertrend_vals[-1] or 1) * 100,
                -5, 5
            ) if len(supertrend_vals) > 0 else 50.0,
            "supertrend_direction": 100.0 if direction[-1] == 1 else 0.0,
        }

    def _extract_stochastic_features(
        self, high: List[float], low: List[float], close: List[float]
    ) -> Dict[str, float]:
        if len(close) < 14:
            return {"stoch_k": 50.0, "stoch_d": 50.0}
        k_vals: List[float] = []
        for i in range(len(close)):
            if i < 13:
                k_vals.append(50.0)
                continue
            window_high = max(high[i - 13 : i + 1])
            window_low = min(low[i - 13 : i + 1])
            if window_high == window_low:
                k_vals.append(50.0)
            else:
                k_vals.append(
                    (close[i] - window_low) / (window_high - window_low) * 100
                )
        d_vals = self._sma(k_vals, 3)
        return {
            "stoch_k": self._clamp_normalize(k_vals[-1], 0, 100),
            "stoch_d": self._clamp_normalize(d_vals[-1] if d_vals else 50.0, 0, 100),
        }

    def _extract_mfi_features(
        self,
        high: List[float],
        low: List[float],
        close: List[float],
        volume: List[int],
    ) -> Dict[str, float]:
        if len(close) < 14:
            return {"mfi": 50.0}
        typical_price = [(h + l + c) / 3 for h, l, c in zip(high, low, close)]
        mfi_vals: List[float] = []
        for i in range(len(typical_price)):
            if i < 14:
                mfi_vals.append(50.0)
                continue
            positive_flow = 0.0
            negative_flow = 0.0
            for j in range(i - 13, i + 1):
                money_flow = typical_price[j] * volume[j]
                if j > 0 and typical_price[j] >= typical_price[j - 1]:
                    positive_flow += money_flow
                else:
                    negative_flow += money_flow
            if negative_flow == 0:
                mfi_vals.append(100.0)
            else:
                mfr = positive_flow / negative_flow
                mfi_vals.append(100.0 - 100.0 / (1.0 + mfr))
        return {"mfi": self._clamp_normalize(mfi_vals[-1], 0, 100)}

    def _extract_cci_features(
        self, high: List[float], low: List[float], close: List[float]
    ) -> Dict[str, float]:
        if len(close) < 20:
            return {"cci": 50.0}
        tp = [(h + l + c) / 3 for h, l, c in zip(high, low, close)]
        sma_tp = self._sma(tp, 20)
        cci_vals: List[float] = []
        for i in range(len(tp)):
            if i < 19:
                cci_vals.append(0.0)
                continue
            mean_dev = sum(abs(tp[j] - sma_tp[i]) for j in range(i - 19, i + 1)) / 20
            if mean_dev == 0:
                cci_vals.append(0.0)
            else:
                cci_vals.append((tp[i] - sma_tp[i]) / (0.015 * mean_dev))
        return {
            "cci": self._clamp_normalize(cci_vals[-1], -200, 200),
        }

    def _extract_williams_r(
        self, high: List[float], low: List[float], close: List[float]
    ) -> Dict[str, float]:
        if len(close) < 14:
            return {"williams_r": 50.0}
        period = 14
        highest = max(high[-period:])
        lowest = min(low[-period:])
        wr = -100.0
        if highest != lowest:
            wr = (highest - close[-1]) / (highest - lowest) * -100
        return {"williams_r": self._clamp_normalize(wr, -100, 0)}

    # ------------------------------------------------------------------
    # Volume Features
    # ------------------------------------------------------------------

    def _extract_volume_features(
        self, volume: List[int], price_data: PriceData,
        close_prices: Optional[List[float]] = None,
        high_prices: Optional[List[float]] = None,
        low_prices: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        features: Dict[str, float] = {}
        if not volume:
            return {
                "volume_expansion_ratio": 50.0,
                "delivery_percent": 50.0,
                "volume_profile": 50.0,
                "accumulation_distribution": 50.0,
            }

        avg_vol_5 = sum(volume[-5:]) / 5 if len(volume) >= 5 else (volume[-1] or 1)
        avg_vol_20 = sum(volume[-20:]) / 20 if len(volume) >= 20 else avg_vol_5
        vol_ratio = volume[-1] / (avg_vol_20 or 1)
        features["volume_expansion_ratio"] = self._clamp_normalize(vol_ratio, 0.2, 5.0)

        del_per = self._safe_decimal_float(price_data.delivery_percent)
        features["delivery_percent"] = self._clamp_normalize(del_per, 0, 100)

        vol_vs_ema = volume[-1] / (avg_vol_5 or 1)
        features["volume_profile"] = self._clamp_normalize(vol_vs_ema, 0.2, 3.0)

        if (close_prices is not None and high_prices is not None and low_prices is not None
                and len(volume) >= 20 and len(close_prices) >= 20
                and len(high_prices) >= 20 and len(low_prices) >= 20):
            try:
                ad_sum = 0.0
                for i in range(-20, 0):
                    hi = high_prices[i]
                    lo = low_prices[i]
                    cl = close_prices[i]
                    hl_range = hi - lo
                    if hl_range > 0:
                        mfm = ((cl - lo) - (hi - cl)) / hl_range
                    else:
                        mfm = 0.0
                    ad_sum += mfm * volume[i]
                ad_norm = ad_sum / 20
                features["accumulation_distribution"] = self._clamp_normalize(ad_norm, -100000, 100000)
            except (ValueError, ZeroDivisionError, IndexError):
                features["accumulation_distribution"] = 50.0
        else:
            features["accumulation_distribution"] = 50.0

        return features

    # ------------------------------------------------------------------
    # Options Features
    # ------------------------------------------------------------------

    def _extract_options_features(
        self, option_data: Optional[OptionChainData]
    ) -> Dict[str, float]:
        features: Dict[str, float] = {}
        if option_data is None:
            return {
                "pcr": 50.0,
                "iv_rank": 50.0,
                "iv_percentile": 50.0,
                "oi_build_type": 50.0,
                "max_pain_distance": 50.0,
                "pcr_change": 50.0,
            }

        pcr_val = self._safe_decimal_float(option_data.pcr)
        features["pcr"] = self._clamp_normalize(pcr_val, 0.2, 2.0)

        iv_rank = self._safe_decimal_float(option_data.iv_rank)
        features["iv_rank"] = self._clamp_normalize(iv_rank, 0, 100)

        iv_pctl = self._safe_decimal_float(option_data.iv_percentile)
        features["iv_percentile"] = self._clamp_normalize(iv_pctl, 0, 100)

        oi_build = self._detect_oi_build_type(option_data)
        features["oi_build_type"] = self._oi_build_to_score(oi_build)

        max_pain = self._safe_decimal_float(option_data.max_pain)
        features["max_pain_distance"] = 50.0
        if max_pain > 0:
            features["max_pain_distance"] = 50.0

        if option_data.total_ce_oi and option_data.total_pe_oi and option_data.total_ce_oi > 0:
            oi_ratio = option_data.total_pe_oi / option_data.total_ce_oi
            features["oi_ratio"] = self._clamp_normalize(float(oi_ratio), 0.2, 2.0)
        else:
            features["oi_ratio"] = 50.0

        return features

    def _detect_oi_build_type(self, option_data: OptionChainData) -> str:
        strikes = option_data.strikes
        if not strikes:
            return "neutral"
        ce_oi_change = sum(
            float(s.get("change_in_oi", 0))
            for s in strikes
            if isinstance(s, dict) and s.get("option_type") == "CE" and s.get("strike_price", 0) > 0
        )
        pe_oi_change = sum(
            float(s.get("change_in_oi", 0))
            for s in strikes
            if isinstance(s, dict) and s.get("option_type") == "PE" and s.get("strike_price", 0) > 0
        )
        # Use dict fields
        ce_change = sum(
            s.get("change_in_oi", 0) for s in strikes
        )
        pe_change = sum(
            s.get("change_in_oi", 0) for s in strikes
        )
        if ce_oi_change > pe_oi_change and ce_oi_change > 0:
            return "short_build"
        elif pe_oi_change > ce_oi_change and pe_oi_change > 0:
            return "long_build"
        elif ce_oi_change < 0 and pe_oi_change < 0:
            if abs(ce_oi_change) > abs(pe_oi_change):
                return "short_covering"
            else:
                return "long_unwinding"
        return "neutral"

    def _oi_build_to_score(self, build_type: str) -> float:
        mapping = {
            "long_build": 90.0,
            "short_covering": 75.0,
            "neutral": 50.0,
            "long_unwinding": 30.0,
            "short_build": 15.0,
        }
        return mapping.get(build_type, 50.0)

    # ------------------------------------------------------------------
    # Momentum Features
    # ------------------------------------------------------------------

    def _extract_momentum_features(
        self, close: List[float]
    ) -> Dict[str, float]:
        if len(close) < 2:
            return {
                "return_1d": 50.0,
                "return_5d": 50.0,
                "return_20d": 50.0,
                "roc": 50.0,
                "momentum_score": 50.0,
            }

        ret_1d = (close[-1] - close[-2]) / (close[-2] or 1) * 100
        ret_5d = (
            (close[-1] - close[-5]) / (close[-5] or 1) * 100 if len(close) >= 5 else ret_1d
        )
        ret_20d = (
            (close[-1] - close[-20]) / (close[-20] or 1) * 100 if len(close) >= 20 else ret_5d
        )
        roc = (
            (close[-1] - close[-10]) / (close[-10] or 1) * 100 if len(close) >= 10 else ret_5d
        )

        mom = ret_1d * 0.5 + ret_5d * 0.3 + ret_20d * 0.2

        features = {
            "return_1d": self._clamp_normalize(ret_1d, -5, 5),
            "return_5d": self._clamp_normalize(ret_5d, -8, 8),
            "return_20d": self._clamp_normalize(ret_20d, -15, 15),
            "roc": self._clamp_normalize(roc, -10, 10),
            "momentum_score": self._clamp_normalize(mom, -10, 10),
        }

        if len(close) >= 5:
            strength_count = sum(
                1 for i in range(1, 6) if close[-i] >= close[-i - 1]
            )
            features["momentum_strength"] = self._clamp_normalize(strength_count, 0, 5)
        else:
            features["momentum_strength"] = 50.0

        return features

    # ------------------------------------------------------------------
    # Sector Features
    # ------------------------------------------------------------------

    def _extract_sector_features(
        self,
        symbol: str,
        indices: Dict[str, IndexData],
        all_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        features: Dict[str, float] = {}
        sector_map = {
            "RELIANCE": "NIFTY OIL & GAS",
            "TCS": "NIFTY IT",
            "INFY": "NIFTY IT",
            "HDFCBANK": "NIFTY BANK",
            "ICICIBANK": "NIFTY BANK",
            "SBIN": "NIFTY BANK",
            "AXISBANK": "NIFTY BANK",
            "KOTAKBANK": "NIFTY BANK",
            "BAJFINANCE": "NIFTY FINANCIAL SERVICES",
            "LT": "NIFTY INFRA",
            "TITAN": "NIFTY CONSUMER DURABLES",
            "MARUTI": "NIFTY AUTO",
            "M&M": "NIFTY AUTO",
            "TRENT": "NIFTY MIDCAP 100",
            "SUNPHARMA": "NIFTY PHARMA",
            "DRREDDY": "NIFTY PHARMA",
            "CIPLA": "NIFTY PHARMA",
            "DIVISLAB": "NIFTY PHARMA",
            "NTPC": "NIFTY ENERGY",
            "POWERGRID": "NIFTY ENERGY",
            "HAL": "NIFTY PSU",
            "BEL": "NIFTY PSU",
            "HINDUNILVR": "NIFTY FMCG",
            "NESTLEIND": "NIFTY FMCG",
            "BRITANNIA": "NIFTY FMCG",
            "ITC": "NIFTY FMCG",
            "TATASTEEL": "NIFTY METAL",
            "HINDALCO": "NIFTY METAL",
            "JSWSTEEL": "NIFTY METAL",
            "TECHM": "NIFTY IT",
            "HCLTECH": "NIFTY IT",
            "WIPRO": "NIFTY IT",
            "BHARTIARTL": "NIFTY MEDIA",
            "ULTRACEMCO": "NIFTY REALTY",
            "ADANIPORTS": "NIFTY INFRA",
            "ONGC": "NIFTY OIL & GAS",
            "BPCL": "NIFTY OIL & GAS",
            "IOC": "NIFTY OIL & GAS",
            "COALINDIA": "NIFTY ENERGY",
            "BAJAJFINSV": "NIFTY FINANCIAL SERVICES",
            "ASIANPAINT": "NIFTY CONSUMER DURABLES",
            "HDFCLIFE": "NIFTY FINANCIAL SERVICES",
            "SBILIFE": "NIFTY FINANCIAL SERVICES",
            "EICHERMOT": "NIFTY AUTO",
            "HEROMOTOCO": "NIFTY AUTO",
            "BAJAJ-AUTO": "NIFTY AUTO",
            "GRASIM": "NIFTY CEMENT",
            "VOLTAS": "NIFTY CONSUMER DURABLES",
            "MUTHOOTFIN": "NIFTY FINANCIAL SERVICES",
            "ICICIPRULI": "NIFTY FINANCIAL SERVICES",
        }

        sector = sector_map.get(symbol, "NIFTY 50")
        idx_data = indices.get(sector)

        if idx_data:
            sector_change = self._safe_decimal_float(idx_data.change_percent)
            features["sector_change"] = self._clamp_normalize(sector_change, -3, 3)
        else:
            features["sector_change"] = 50.0

        nifty = indices.get("NIFTY 50")
        if idx_data and nifty:
            rel_strength = self._safe_decimal_float(idx_data.change_percent) - \
                self._safe_decimal_float(nifty.change_percent)
            features["sector_relative_strength"] = self._clamp_normalize(
                rel_strength, -2, 2
            )
        else:
            features["sector_relative_strength"] = 50.0

        features["sector_rank"] = 50.0
        if all_scores is not None and symbol in all_scores:
            sorted_scores = sorted(all_scores.values(), reverse=True)
            if sorted_scores:
                score = all_scores[symbol]
                rank = sum(1 for s in sorted_scores if s > score) + 1
                total = len(sorted_scores)
                features["sector_rank"] = self._clamp_normalize(
                    (total - rank) / total * 100, 0, 100
                )

        features["sector_momentum"] = features.get("sector_change", 50.0)

        return features

    # ------------------------------------------------------------------
    # Market Features
    # ------------------------------------------------------------------

    def _extract_market_features(
        self,
        breadth: Optional[MarketBreadthData],
        inst: Optional[InstitutionalFlowData],
        vix: Optional[VIXData],
        close: List[float],
    ) -> Dict[str, float]:
        features: Dict[str, float] = {}

        if breadth:
            ad_ratio = self._safe_decimal_float(breadth.advance_decline_ratio)
            features["market_breadth"] = self._clamp_normalize(ad_ratio, 0.2, 3.0)
        else:
            features["market_breadth"] = 50.0

        if inst:
            fii = self._safe_decimal_float(inst.total_fii)
            dii = self._safe_decimal_float(inst.total_dii)
            net = fii + dii
            features["fii_dii_net"] = self._clamp_normalize(net / 1000, -5000, 5000)
        else:
            features["fii_dii_net"] = 50.0

        if vix:
            vix_val = self._safe_decimal_float(vix.value)
            vix_change = self._safe_decimal_float(vix.change_percent)
            features["vix_level"] = self._clamp_normalize(vix_val, 10, 40)
            features["vix_change"] = self._clamp_normalize(vix_change, -20, 20)
        else:
            features["vix_level"] = 50.0
            features["vix_change"] = 50.0

        if len(close) >= 2:
            gap_pct = (close[-1] - close[-2]) / (close[-2] or 1) * 100
            features["gap_percent"] = self._clamp_normalize(gap_pct, -3, 3)
        else:
            features["gap_percent"] = 50.0

        features["opening_range_percent"] = 50.0

        return features

    # ------------------------------------------------------------------
    # Volatility Features
    # ------------------------------------------------------------------

    def _extract_volatility_features(
        self,
        close: List[float],
        high: List[float],
        low: List[float],
        option_data: Optional[OptionChainData],
    ) -> Dict[str, float]:
        features: Dict[str, float] = {}

        if len(close) >= 20:
            log_returns = [
                np.log(close[i] / close[i - 1])
                for i in range(1, len(close))
                if close[i - 1] > 0
            ]
            hv = float(np.std(log_returns) * np.sqrt(252)) * 100 if log_returns else 0
            features["historical_volatility"] = self._clamp_normalize(hv, 0, 80)

            hv_20 = float(np.std(log_returns[-20:]) * np.sqrt(252)) * 100 if len(log_returns) >= 20 else hv
            features["hv_20"] = self._clamp_normalize(hv_20, 0, 80)
        else:
            features["historical_volatility"] = 50.0
            features["hv_20"] = 50.0

        if option_data and option_data.iv_rank:
            iv = self._safe_decimal_float(option_data.iv_rank)
            features["implied_volatility"] = self._clamp_normalize(iv, 0, 100)
            hv = features.get("historical_volatility", 50.0)
            spread = iv - hv
            features["iv_hv_spread"] = self._clamp_normalize(spread, -20, 20)
        else:
            features["implied_volatility"] = 50.0
            features["iv_hv_spread"] = 50.0

        atr_vals = self._atr(high, low, close)
        if len(atr_vals) >= 2:
            features["atr_expansion_ratio"] = self._clamp_normalize(
                atr_vals[-1] / (atr_vals[-2] or 1), 0.5, 2.0
            )
        else:
            features["atr_expansion_ratio"] = 50.0

        return features

    # ------------------------------------------------------------------
    # Quality Features
    # ------------------------------------------------------------------

    def _extract_quality_features(
        self,
        close: List[float],
        high: List[float],
        low: List[float],
        volume: List[int],
    ) -> Dict[str, float]:
        features: Dict[str, float] = {}

        if len(close) < 2:
            return {
                "candle_quality": 50.0,
                "breakout_strength": 50.0,
                "breakdown_strength": 50.0,
                "pattern_strength": 50.0,
                "liquidity_score": 50.0,
            }

        candle_range = high[-1] - low[-1]
        body = abs(close[-1] - close[-2])
        upper_wick = high[-1] - max(close[-1], close[-2])
        lower_wick = min(close[-1], close[-2]) - low[-1]

        if candle_range > 0:
            body_ratio = body / candle_range
            upper_ratio = upper_wick / candle_range
            lower_ratio = lower_wick / candle_range
            if body_ratio > 0.6 and upper_ratio < 0.15 and lower_ratio < 0.15:
                candle_quality = 90.0
            elif body_ratio > 0.5 and upper_ratio < 0.2 and lower_ratio < 0.2:
                candle_quality = 75.0
            elif body_ratio > 0.3:
                candle_quality = 50.0
            else:
                candle_quality = 30.0
        else:
            candle_quality = 50.0

        features["candle_quality"] = candle_quality

        if (
            len(close) >= 21
            and close[-1] > max(close[-21:-1])
            and volume[-1] > sum(volume[-6:-1]) / 5
        ):
            breakout_strength = 85.0
        elif len(close) >= 11 and close[-1] > max(close[-11:-1]):
            breakout_strength = 70.0
        else:
            breakout_strength = 50.0

        features["breakout_strength"] = breakout_strength

        if (
            len(close) >= 21
            and close[-1] < min(close[-21:-1])
        ):
            features["breakdown_strength"] = 85.0
        elif len(close) >= 11 and close[-1] < min(close[-11:-1]):
            features["breakdown_strength"] = 70.0
        else:
            features["breakdown_strength"] = 50.0

        features["pattern_strength"] = 50.0

        avg_vol = sum(volume[-20:]) / 20 if len(volume) >= 20 else (volume[-1] or 1)
        liquidity = volume[-1] / (avg_vol or 1) * 50
        features["liquidity_score"] = self._clamp_normalize(liquidity, 0, 100)

        return features

    # ------------------------------------------------------------------
    # Main Feature Computation
    # ------------------------------------------------------------------

    def compute_all_features(
        self,
        symbol: str,
        market_data: Optional[PriceData],
        option_data: Optional[OptionChainData],
        sector_data: Optional[Dict[str, IndexData]] = None,
        breadth: Optional[MarketBreadthData] = None,
        inst: Optional[InstitutionalFlowData] = None,
        vix: Optional[VIXData] = None,
        all_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        features: Dict[str, float] = {}

        if (
            market_data is None
        ):
            return self._get_default_features()

        close_raw = [self._safe_float(market_data.close)]
        high_raw = [self._safe_float(market_data.high)]
        low_raw = [self._safe_float(market_data.low)]
        volume_raw = [market_data.volume if market_data.volume else 0]

        if symbol in self._price_history and len(self._price_history[symbol]) >= 2:
            hist = self._price_history[symbol]
            close_raw = [self._safe_float(p.close) for p in hist]
            high_raw = [self._safe_float(p.high) for p in hist]
            low_raw = [self._safe_float(p.low) for p in hist]
            volume_raw = [p.volume if p.volume else 0 for p in hist]

        vwap = self._safe_float(market_data.vwap)

        features.update(self._extract_ema_features(close_raw))
        features.update(self._extract_rsi_features(close_raw))
        features.update(self._extract_macd_features(close_raw))
        features.update(self._extract_bb_features(close_raw))
        features.update(self._extract_vwap_features(close_raw, vwap))
        features.update(self._extract_atr_features(high_raw, low_raw, close_raw))
        features.update(self._extract_adx_features(high_raw, low_raw, close_raw))
        features.update(self._extract_supertrend(high_raw, low_raw, close_raw))
        features.update(self._extract_stochastic_features(high_raw, low_raw, close_raw))
        features.update(self._extract_mfi_features(high_raw, low_raw, close_raw, volume_raw))
        features.update(self._extract_cci_features(high_raw, low_raw, close_raw))
        features.update(self._extract_williams_r(high_raw, low_raw, close_raw))

        features.update(self._extract_volume_features(volume_raw, market_data, close_raw, high_raw, low_raw))
        features.update(self._extract_options_features(option_data))
        features.update(self._extract_momentum_features(close_raw))

        indices = sector_data or {}
        features.update(self._extract_sector_features(symbol, indices, all_scores))

        features.update(self._extract_market_features(breadth, inst, vix, close_raw))
        features.update(self._extract_volatility_features(close_raw, high_raw, low_raw, option_data))
        features.update(self._extract_quality_features(close_raw, high_raw, low_raw, volume_raw))

        features["feature_completeness"] = self._calculate_completeness(features)

        return features

    def _get_default_features(self) -> Dict[str, float]:
        categories = [
            "ema", "rsi", "macd", "bb", "vwap", "atr", "adx", "supertrend",
            "stoch", "mfi", "cci", "williams_r",
            "volume", "options", "momentum", "sector", "market", "volatility", "quality",
        ]
        features: Dict[str, float] = {}
        for cat in categories:
            features[f"{cat}_default"] = 50.0
        features["feature_completeness"] = 0.0
        return features

    def _calculate_completeness(self, features: Dict[str, float]) -> float:
        non_default = sum(
            1 for v in features.values() if v != 50.0
        )
        total = len(features)
        return round(non_default / (total or 1) * 100, 2)

    def update_price_history(
        self, symbol: str, price_data: PriceData
    ) -> None:
        if symbol not in self._price_history:
            self._price_history[symbol] = []
        self._price_history[symbol].append(price_data)
        if len(self._price_history[symbol]) > self._max_history:
            self._price_history[symbol] = self._price_history[symbol][
                -self._max_history:
            ]

    def get_feature_names(self) -> List[str]:
        sample = self._get_default_features()
        return list(sample.keys())

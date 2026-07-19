"""Chart pattern scanner engine.

Detects common technical chart patterns from Yahoo Finance daily OHLCV data.
Uses pure Python — no TA-Lib dependency.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from trademind.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class PatternResult:
    symbol: str
    pattern_name: str
    pattern_type: str
    confidence: float
    entry_price: float
    stop_loss: float
    target_price: float
    timeframe: str
    supporting_evidence: List[str]
    detected_at: datetime = field(default_factory=datetime.utcnow)


def _find_peaks_and_troughs(
    closes: np.ndarray, window: int = 5
) -> Tuple[List[int], List[int]]:
    """Return index lists of local peaks and troughs."""
    peaks: List[int] = []
    troughs: List[int] = []
    n = len(closes)
    for i in range(window, n - window):
        if all(closes[i] >= closes[i - j] for j in range(1, window + 1)) and all(
            closes[i] >= closes[i + j] for j in range(1, window + 1)
        ):
            peaks.append(i)
        if all(closes[i] <= closes[i - j] for j in range(1, window + 1)) and all(
            closes[i] <= closes[i + j] for j in range(1, window + 1)
        ):
            troughs.append(i)
    return peaks, troughs


def _check_double_top(
    highs: np.ndarray, closes: np.ndarray, tolerance: float = 0.02
) -> List[Tuple[float, int, int]]:
    """Check for double-top pattern. Returns (confidence, peak1_idx, peak2_idx)."""
    results: List[Tuple[float, int, int]] = []
    peaks, _ = _find_peaks_and_troughs(closes, window=5)
    if len(peaks) < 2:
        return results

    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            p1, p2 = peaks[i], peaks[j]
            h1, h2 = highs[p1], highs[p2]
            avg_h = (h1 + h2) / 2.0
            if avg_h == 0:
                continue
            diff = abs(h1 - h2) / avg_h
            if diff > tolerance:
                continue
            if p2 - p1 < 10:
                continue
            neckline = float(np.min(closes[p1:p2]))
            if neckline == 0:
                continue
            depth = avg_h - neckline
            if depth / avg_h < 0.01:
                continue
            break_conf = float(closes[-1] < neckline)
            conf = max(0.3, min(1.0, (1.0 - diff / tolerance) * 0.6 + 0.2 + break_conf * 0.2))
            results.append((round(conf, 4), p1, p2))
    return results


def _check_double_bottom(
    lows: np.ndarray, closes: np.ndarray, tolerance: float = 0.02
) -> List[Tuple[float, int, int]]:
    """Check for double-bottom pattern. Returns (confidence, trough1_idx, trough2_idx)."""
    results: List[Tuple[float, int, int]] = []
    _, troughs = _find_peaks_and_troughs(closes, window=5)
    if len(troughs) < 2:
        return results

    for i in range(len(troughs)):
        for j in range(i + 1, len(troughs)):
            t1, t2 = troughs[i], troughs[j]
            l1, l2 = lows[t1], lows[t2]
            avg_l = (l1 + l2) / 2.0
            if avg_l == 0:
                continue
            diff = abs(l1 - l2) / avg_l
            if diff > tolerance:
                continue
            if t2 - t1 < 10:
                continue
            neckline = float(np.max(closes[t1:t2]))
            depth = neckline - avg_l
            if neckline == 0 or depth / neckline < 0.01:
                continue
            break_conf = float(closes[-1] > neckline)
            conf = max(0.3, min(1.0, (1.0 - diff / tolerance) * 0.6 + 0.2 + break_conf * 0.2))
            results.append((round(conf, 4), t1, t2))
    return results


def _check_head_shoulders(
    highs: np.ndarray, closes: np.ndarray
) -> List[Tuple[float, int, int, int]]:
    """Check for head-and-shoulders. Returns (confidence, left_idx, head_idx, right_idx)."""
    results: List[Tuple[float, int, int, int]] = []
    peaks, _ = _find_peaks_and_troughs(closes, window=5)
    if len(peaks) < 3:
        return results

    for i in range(len(peaks) - 2):
        l, h, r = peaks[i], peaks[i + 1], peaks[i + 2]
        hl, hh, hr = highs[l], highs[h], highs[r]
        if hh <= hl or hh <= hr:
            continue
        shoulder_diff = abs(hl - hr) / hh if hh != 0 else 1.0
        if shoulder_diff > 0.03:
            continue
        neck_l = float(np.min(closes[l:h]))
        neck_r = float(np.min(closes[h:r]))
        neckline = (neck_l + neck_r) / 2.0
        if neckline == 0:
            continue
        depth = hh - neckline
        if depth / hh < 0.01:
            continue
        break_conf = float(closes[-1] < neckline)
        conf = max(0.35, min(1.0, (1.0 - shoulder_diff / 0.03) * 0.5 + 0.25 + break_conf * 0.25))
        results.append((round(conf, 4), l, h, r))
    return results


def _check_inverse_head_shoulders(
    lows: np.ndarray, closes: np.ndarray
) -> List[Tuple[float, int, int, int]]:
    """Check for inverse head-and-shoulders."""
    results: List[Tuple[float, int, int, int]] = []
    _, troughs = _find_peaks_and_troughs(closes, window=5)
    if len(troughs) < 3:
        return results

    for i in range(len(troughs) - 2):
        l, h, r = troughs[i], troughs[i + 1], troughs[i + 2]
        ll, lh, lr = lows[l], lows[h], lows[r]
        if lh >= ll or lh >= lr:
            continue
        shoulder_diff = abs(ll - lr) / lh if lh != 0 else 1.0
        if shoulder_diff > 0.03:
            continue
        neck_l = float(np.max(closes[l:h]))
        neck_r = float(np.max(closes[h:r]))
        neckline = (neck_l + neck_r) / 2.0
        if neckline == 0:
            continue
        depth = neckline - lh
        if depth / neckline < 0.01:
            continue
        break_conf = float(closes[-1] > neckline)
        conf = max(0.35, min(1.0, (1.0 - shoulder_diff / 0.03) * 0.5 + 0.25 + break_conf * 0.25))
        results.append((round(conf, 4), l, h, r))
    return results


def _check_flags(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, lookback: int = 20
) -> List[Tuple[str, float, int]]:
    """Check for bull/bear flag patterns. Returns (flag_type, confidence, flag_start_idx)."""
    results: List[Tuple[str, float, int]] = []
    n = len(closes)
    if n < lookback + 10:
        return results

    for end in range(lookback + 10, n):
        pole_start = max(0, end - lookback)
        pole_end = end

        pole_range = closes[pole_start:pole_end]
        if len(pole_range) < 10:
            continue

        pole_move = float(pole_range[-1] - pole_range[0])
        avg_price = float(np.mean(pole_range))
        if avg_price == 0:
            continue
        pole_pct = pole_move / avg_price

        if abs(pole_pct) < 0.04:
            continue

        flag_len = min(lookback, n - end)
        if flag_len < 5:
            continue
        flag_closes = closes[end:end + flag_len]
        flag_highs_arr = highs[end:end + flag_len]
        flag_lows_arr = lows[end:end + flag_len]

        x = np.arange(len(flag_closes))
        if len(flag_closes) < 3:
            continue

        coeffs_high = np.polyfit(x, flag_highs_arr, 1)
        coeffs_low = np.polyfit(x, flag_lows_arr, 1)

        high_slope = coeffs_high[0] / avg_price if avg_price != 0 else 0
        low_slope = coeffs_low[0] / avg_price if avg_price != 0 else 0

        flag_range = float(np.max(flag_closes) - np.min(flag_closes))
        flag_pct = flag_range / avg_price if avg_price != 0 else 0

        if flag_pct > 0.06:
            continue

        if pole_pct > 0.04 and high_slope < 0.001 and low_slope < 0.001:
            conf = min(1.0, 0.4 + abs(pole_pct) * 3 + (0.05 - flag_pct) * 5)
            conf = max(0.3, min(conf, 1.0))
            results.append(("Bull Flag", round(conf, 4), pole_start))
        elif pole_pct < -0.04 and high_slope > -0.001 and low_slope > -0.001:
            conf = min(1.0, 0.4 + abs(pole_pct) * 3 + (0.05 - flag_pct) * 5)
            conf = max(0.3, min(conf, 1.0))
            results.append(("Bear Flag", round(conf, 4), pole_start))

    return results[-3:] if results else []


def _check_triangles(
    highs: np.ndarray, lows: np.ndarray
) -> List[Tuple[str, float, int]]:
    """Check for ascending/descending triangle patterns."""
    results: List[Tuple[str, float, int]] = []
    n = len(highs)
    if n < 20:
        return results

    for window_size in [20, 30, 40]:
        for start in range(0, n - window_size, 5):
            end = start + window_size
            h_slice = highs[start:end]
            l_slice = lows[start:end]

            x = np.arange(len(h_slice), dtype=float)
            h_coeffs = np.polyfit(x, h_slice, 1)
            l_coeffs = np.polyfit(x, l_slice, 1)

            avg_h = float(np.mean(h_slice))
            avg_l = float(np.mean(l_slice))
            if avg_h == 0 or avg_l == 0:
                continue

            h_slope_norm = h_coeffs[0] / avg_h
            l_slope_norm = l_coeffs[0] / avg_l

            h_flat = abs(h_slope_norm) < 0.0005
            l_flat = abs(l_slope_norm) < 0.0005
            h_rising = h_slope_norm > 0.0005
            l_rising = l_slope_norm > 0.0005
            h_falling = h_slope_norm < -0.0005
            l_falling = l_slope_norm < -0.0005

            if h_flat and l_rising:
                conf = min(1.0, 0.5 + (0.005 - abs(h_slope_norm)) * 100 + l_slope_norm * 50)
                conf = max(0.3, min(conf, 1.0))
                results.append(("Ascending Triangle", round(conf, 4), start))
            elif l_flat and h_falling:
                conf = min(1.0, 0.5 + (0.005 - abs(l_slope_norm)) * 100 + abs(h_slope_norm) * 50)
                conf = max(0.3, min(conf, 1.0))
                results.append(("Descending Triangle", round(conf, 4), start))

    seen: set = set()
    unique: List[Tuple[str, float, int]] = []
    for name, conf, idx in results:
        key = (name, idx // 15)
        if key not in seen:
            seen.add(key)
            unique.append((name, conf, idx))
    return unique[:3]


def _check_cup_handle(closes: np.ndarray, window: int = 60) -> List[Tuple[float, int]]:
    """Check for cup-and-handle pattern. Returns (confidence, cup_start_idx)."""
    results: List[Tuple[float, int]] = []
    n = len(closes)
    if n < window + 15:
        return results

    for start in range(0, n - window - 10, 5):
        cup_end = start + window
        if cup_end + 15 > n:
            break
        cup = closes[start:cup_end]

        rim_left = float(cup[0])
        rim_right = float(cup[-1])
        rim_avg = (rim_left + rim_right) / 2.0
        if rim_avg == 0:
            continue

        rim_diff = abs(rim_left - rim_right) / rim_avg
        if rim_diff > 0.05:
            continue

        cup_bottom = float(np.min(cup))
        depth = rim_avg - cup_bottom
        if depth / rim_avg < 0.03:
            continue

        mid = len(cup) // 2
        left_half = np.mean(cup[:mid])
        right_half = np.mean(cup[mid:])
        if right_half < left_half:
            continue

        handle = closes[cup_end:cup_end + 15]
        if len(handle) < 5:
            continue
        handle_move = float(handle[-1] - handle[0])
        handle_pct = abs(handle_move) / rim_avg if rim_avg != 0 else 1.0
        if handle_pct > 0.04:
            continue

        handle_slope = np.polyfit(np.arange(len(handle)), handle, 1)[0]
        if handle_slope > 0:
            continue

        conf = min(1.0, 0.4 + (1.0 - rim_diff / 0.05) * 0.3 + (depth / rim_avg) * 2.0)
        conf = max(0.3, min(conf, 1.0))
        results.append((round(conf, 4), start))

    return results[:3]


def _check_wedges(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray
) -> List[Tuple[str, float, int]]:
    """Check for rising/falling wedge patterns."""
    results: List[Tuple[str, float, int]] = []
    n = len(highs)
    if n < 20:
        return results

    for window_size in [20, 30, 40]:
        for start in range(0, n - window_size, 5):
            end = start + window_size
            h_slice = highs[start:end]
            l_slice = lows[start:end]
            c_slice = closes[start:end]

            x = np.arange(len(h_slice), dtype=float)
            h_coeffs = np.polyfit(x, h_slice, 1)
            l_coeffs = np.polyfit(x, l_slice, 1)

            avg_h = float(np.mean(h_slice))
            avg_l = float(np.mean(l_slice))
            if avg_h == 0 or avg_l == 0:
                continue

            h_slope = h_coeffs[0] / avg_h
            l_slope = l_coeffs[0] / avg_l

            h_rising = h_slope > 0.0003
            l_rising = l_slope > 0.0003
            h_falling = h_slope < -0.0003
            l_falling = l_slope < -0.0003

            if h_rising and l_rising and l_slope > h_slope + 0.0001:
                trend_dir = "up" if c_slice[-1] > c_slice[0] else "down"
                if trend_dir == "up":
                    ptype = "Falling Wedge"
                    conf = min(1.0, 0.45 + (l_slope - h_slope) * 200)
                    conf = max(0.3, min(conf, 1.0))
                    results.append((ptype, round(conf, 4), start))
            elif h_falling and l_falling and h_slope > l_slope + 0.0001:
                trend_dir = "up" if c_slice[-1] > c_slice[0] else "down"
                if trend_dir == "down":
                    ptype = "Rising Wedge"
                    conf = min(1.0, 0.45 + (h_slope - l_slope) * 200)
                    conf = max(0.3, min(conf, 1.0))
                    results.append((ptype, round(conf, 4), start))

    seen: set = set()
    unique: List[Tuple[str, float, int]] = []
    for name, conf, idx in results:
        key = (name, idx // 15)
        if key not in seen:
            seen.add(key)
            unique.append((name, conf, idx))
    return unique[:3]


class PatternScannerEngine:
    """Scans Yahoo Finance daily OHLCV data for common chart patterns."""

    TIMEFRAME = "3M"
    CACHE_TTL_SECONDS = 1800

    def __init__(self) -> None:
        self._cache: Dict[str, Tuple[float, List[PatternResult]]] = {}

    def _is_cache_valid(self, symbol: str) -> bool:
        if symbol not in self._cache:
            return False
        ts, _ = self._cache[symbol]
        return (time.time() - ts) < self.CACHE_TTL_SECONDS

    def _fetch_ohlcv(self, symbol: str) -> Optional[pd.DataFrame]:
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            df = ticker.history(period="3mo", interval="1d")
            if df.empty:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period="3mo", interval="1d")
            return df if not df.empty else None
        except Exception as exc:
            logger.warning("Failed to fetch OHLCV for %s: %s", symbol, exc)
            return None

    def _scan_symbol(self, symbol: str) -> List[PatternResult]:
        if self._is_cache_valid(symbol):
            return self._cache[symbol][1]

        df = self._fetch_ohlcv(symbol)
        if df is None or len(df) < 20:
            logger.info("Insufficient data for %s — skipping pattern scan", symbol)
            return []

        opens = df["Open"].values.astype(float)
        highs = df["High"].values.astype(float)
        lows = df["Low"].values.astype(float)
        closes = df["Close"].values.astype(float)
        last_price = closes[-1]

        patterns: List[PatternResult] = []

        for conf, p1, p2 in _check_double_top(highs, closes):
            neckline = float(np.min(closes[p1:p2]))
            avg_peak = (highs[p1] + highs[p2]) / 2.0
            patterns.append(
                PatternResult(
                    symbol=symbol,
                    pattern_name="Double Top",
                    pattern_type="bearish",
                    confidence=conf,
                    entry_price=round(neckline, 2),
                    stop_loss=round(avg_peak * 1.01, 2),
                    target_price=round(neckline - (avg_peak - neckline), 2),
                    timeframe=self.TIMEFRAME,
                    supporting_evidence=[
                        f"Two peaks at similar level ({highs[p1]:.2f} and {highs[p2]:.2f})",
                        f"Neckline at {neckline:.2f}",
                        f"Price currently at {last_price:.2f}",
                        f"Peak separation: {p2 - p1} bars",
                    ],
                )
            )

        for conf, t1, t2 in _check_double_bottom(lows, closes):
            neckline = float(np.max(closes[t1:t2]))
            avg_trough = (lows[t1] + lows[t2]) / 2.0
            patterns.append(
                PatternResult(
                    symbol=symbol,
                    pattern_name="Double Bottom",
                    pattern_type="bullish",
                    confidence=conf,
                    entry_price=round(neckline, 2),
                    stop_loss=round(avg_trough * 0.99, 2),
                    target_price=round(neckline + (neckline - avg_trough), 2),
                    timeframe=self.TIMEFRAME,
                    supporting_evidence=[
                        f"Two troughs at similar level ({lows[t1]:.2f} and {lows[t2]:.2f})",
                        f"Neckline at {neckline:.2f}",
                        f"Price currently at {last_price:.2f}",
                        f"Trough separation: {t2 - t1} bars",
                    ],
                )
            )

        for conf, left, head, right in _check_head_shoulders(highs, closes):
            neckline = (float(np.min(closes[left:head])) + float(np.min(closes[head:right]))) / 2.0
            head_val = highs[head]
            patterns.append(
                PatternResult(
                    symbol=symbol,
                    pattern_name="Head and Shoulders",
                    pattern_type="bearish",
                    confidence=conf,
                    entry_price=round(neckline, 2),
                    stop_loss=round(head_val * 1.01, 2),
                    target_price=round(neckline - (head_val - neckline), 2),
                    timeframe=self.TIMEFRAME,
                    supporting_evidence=[
                        f"Head at {head_val:.2f}",
                        f"Left shoulder at {highs[left]:.2f}, right at {highs[right]:.2f}",
                        f"Neckline at {neckline:.2f}",
                        f"Pattern span: {right - left} bars",
                    ],
                )
            )

        for conf, left, head, right in _check_inverse_head_shoulders(lows, closes):
            neckline = (float(np.max(closes[left:head])) + float(np.max(closes[head:right]))) / 2.0
            head_val = lows[head]
            patterns.append(
                PatternResult(
                    symbol=symbol,
                    pattern_name="Inverse Head and Shoulders",
                    pattern_type="bullish",
                    confidence=conf,
                    entry_price=round(neckline, 2),
                    stop_loss=round(head_val * 0.99, 2),
                    target_price=round(neckline + (neckline - head_val), 2),
                    timeframe=self.TIMEFRAME,
                    supporting_evidence=[
                        f"Inverse head at {head_val:.2f}",
                        f"Left shoulder at {lows[left]:.2f}, right at {lows[right]:.2f}",
                        f"Neckline at {neckline:.2f}",
                        f"Pattern span: {right - left} bars",
                    ],
                )
            )

        for flag_type, conf, flag_start in _check_flags(highs, lows, closes):
            if flag_type == "Bull Flag":
                pole_low = float(np.min(lows[flag_start:flag_start + 10]))
                patterns.append(
                    PatternResult(
                        symbol=symbol,
                        pattern_name="Bull Flag",
                        pattern_type="bullish",
                        confidence=conf,
                        entry_price=round(last_price, 2),
                        stop_loss=round(pole_low * 0.99, 2),
                        target_price=round(last_price * 1.08, 2),
                        timeframe=self.TIMEFRAME,
                        supporting_evidence=[
                            f"Sharp pole rise before consolidation",
                            f"Flag consolidation range: {float(np.min(closes[flag_start:])):.2f} - {float(np.max(closes[flag_start:])):.2f}",
                            f"Price currently at {last_price:.2f}",
                        ],
                    )
                )
            else:
                pole_high = float(np.max(highs[flag_start:flag_start + 10]))
                patterns.append(
                    PatternResult(
                        symbol=symbol,
                        pattern_name="Bear Flag",
                        pattern_type="bearish",
                        confidence=conf,
                        entry_price=round(last_price, 2),
                        stop_loss=round(pole_high * 1.01, 2),
                        target_price=round(last_price * 0.92, 2),
                        timeframe=self.TIMEFRAME,
                        supporting_evidence=[
                            f"Sharp pole drop before consolidation",
                            f"Flag consolidation range: {float(np.min(closes[flag_start:])):.2f} - {float(np.max(closes[flag_start:])):.2f}",
                            f"Price currently at {last_price:.2f}",
                        ],
                    )
                )

        for tri_name, conf, tri_start in _check_triangles(highs, lows):
            is_bull = "Ascending" in tri_name
            entry = last_price
            if is_bull:
                resistance = float(np.max(highs[tri_start:tri_start + 20]))
                support = float(np.min(lows[tri_start:]))
                patterns.append(
                    PatternResult(
                        symbol=symbol,
                        pattern_name=tri_name,
                        pattern_type="bullish",
                        confidence=conf,
                        entry_price=round(resistance, 2),
                        stop_loss=round(support * 0.99, 2),
                        target_price=round(resistance + (resistance - support), 2),
                        timeframe=self.TIMEFRAME,
                        supporting_evidence=[
                            f"Flat resistance at {resistance:.2f}",
                            f"Rising support trendline",
                            f"Pattern from bar {tri_start}",
                            f"Price currently at {last_price:.2f}",
                        ],
                    )
                )
            else:
                support = float(np.min(lows[tri_start:tri_start + 20]))
                resistance = float(np.max(highs[tri_start:]))
                patterns.append(
                    PatternResult(
                        symbol=symbol,
                        pattern_name=tri_name,
                        pattern_type="bearish",
                        confidence=conf,
                        entry_price=round(support, 2),
                        stop_loss=round(resistance * 1.01, 2),
                        target_price=round(support - (resistance - support), 2),
                        timeframe=self.TIMEFRAME,
                        supporting_evidence=[
                            f"Flat support at {support:.2f}",
                            f"Declining resistance trendline",
                            f"Pattern from bar {tri_start}",
                            f"Price currently at {last_price:.2f}",
                        ],
                    )
                )

        for conf, cup_start in _check_cup_handle(closes):
            cup = closes[cup_start:cup_start + 60]
            rim = (float(cup[0]) + float(cup[-1])) / 2.0
            cup_low = float(np.min(cup))
            patterns.append(
                PatternResult(
                    symbol=symbol,
                    pattern_name="Cup and Handle",
                    pattern_type="bullish",
                    confidence=conf,
                    entry_price=round(rim, 2),
                    stop_loss=round(cup_low * 0.98, 2),
                    target_price=round(rim + (rim - cup_low), 2),
                    timeframe=self.TIMEFRAME,
                    supporting_evidence=[
                        f"U-shape cup from {cup_start} to {cup_start + 60}",
                        f"Cup rim at {rim:.2f}, bottom at {cup_low:.2f}",
                        f"Small handle pullback following cup",
                        f"Price currently at {last_price:.2f}",
                    ],
                )
            )

        for wedge_name, conf, wedge_start in _check_wedges(highs, lows, closes):
            is_rising = "Rising" in wedge_name
            patterns.append(
                PatternResult(
                    symbol=symbol,
                    pattern_name=wedge_name,
                    pattern_type="bearish" if is_rising else "bullish",
                    confidence=conf,
                    entry_price=round(last_price, 2),
                    stop_loss=round(highs[wedge_start] * 1.01 if is_rising else lows[wedge_start] * 0.99, 2),
                    target_price=round(last_price * 0.93 if is_rising else last_price * 1.07, 2),
                    timeframe=self.TIMEFRAME,
                    supporting_evidence=[
                        f"Converging trendlines detected from bar {wedge_start}",
                        f"Upper trendline slope and lower trendline slope converging",
                        f"{'Bearish' if is_rising else 'Bullish'} reversal expected",
                        f"Price currently at {last_price:.2f}",
                    ],
                )
            )

        patterns.sort(key=lambda p: p.confidence, reverse=True)
        self._cache[symbol] = (time.time(), patterns)
        return patterns

    async def scan_symbol(self, symbol: str) -> List[PatternResult]:
        return await asyncio.to_thread(self._scan_symbol, symbol.upper())

    async def scan_all(self, symbols: Optional[List[str]] = None) -> List[PatternResult]:
        if symbols is None:
            symbols = settings.fno_symbols
        results = await asyncio.gather(
            *[self.scan_symbol(s) for s in symbols],
            return_exceptions=True,
        )
        all_patterns: List[PatternResult] = []
        for item in results:
            if isinstance(item, Exception):
                logger.warning("Pattern scan failed: %s", item)
                continue
            all_patterns.extend(item)
        all_patterns.sort(key=lambda p: p.confidence, reverse=True)
        return all_patterns

    def get_summary(self, patterns: List[PatternResult]) -> Dict[str, Any]:
        by_type: Dict[str, int] = {}
        by_pattern: Dict[str, int] = {}
        for p in patterns:
            by_type[p.pattern_type] = by_type.get(p.pattern_type, 0) + 1
            by_pattern[p.pattern_name] = by_pattern.get(p.pattern_name, 0) + 1
        return {
            "total_patterns": len(patterns),
            "by_type": by_type,
            "by_pattern": by_pattern,
            "avg_confidence": round(
                sum(p.confidence for p in patterns) / len(patterns), 4
            )
            if patterns
            else 0.0,
        }

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import yfinance as yf


@dataclass
class SentimentComponent:
    name: str
    score: float
    label: str
    value_raw: Any
    description: str


@dataclass
class SentimentResult:
    overall_score: float
    overall_label: str
    vix_score: SentimentComponent
    fii_dii_score: SentimentComponent
    breadth_score: SentimentComponent
    pcr_score: SentimentComponent
    components: dict
    timestamp: float
    recommendations_impact: str


class SentimentEngine:
    def __init__(self) -> None:
        self._cache: SentimentResult | None = None
        self._cache_time: float = 0.0
        self._cache_ttl: float = 600.0  # 10 minutes

    def _is_cache_valid(self) -> bool:
        return self._cache is not None and (time.time() - self._cache_time) < self._cache_ttl

    def _label_from_score(self, score: float) -> str:
        if score >= 90:
            return "Extreme Greed"
        if score >= 70:
            return "Greed"
        if score >= 40:
            return "Neutral"
        if score >= 20:
            return "Fear"
        return "Extreme Fear"

    def _score_from_label(self, label: str) -> float:
        return {
            "Extreme Greed": 95.0,
            "Greed": 80.0,
            "Neutral": 55.0,
            "Fear": 30.0,
            "Extreme Fear": 10.0,
        }.get(label, 50.0)

    def _score_vix(self) -> SentimentComponent:
        try:
            ticker = yf.Ticker("^INDIAVIX")
            hist = ticker.history(period="1d")
            if hist.empty or hist["Close"].iloc[0] is None:
                raise ValueError("Empty VIX data")
            vix_value = float(hist["Close"].iloc[0])
        except Exception:
            vix_value = 18.0

        if vix_value < 12:
            score = 90.0 + (12.0 - vix_value) * 0.83  # 90-100
            label = "Extreme Greed"
        elif vix_value < 16:
            score = 70.0 + (16.0 - vix_value) * 5.0  # 70-90
            label = "Greed"
        elif vix_value < 20:
            score = 40.0 + (20.0 - vix_value) * 7.25  # 40-69
            label = "Neutral"
        elif vix_value < 25:
            score = 20.0 + (25.0 - vix_value) * 4.0  # 20-40
            label = "Fear"
        else:
            score = max(0.0, 20.0 - (vix_value - 25.0) * 0.8)  # 0-20
            label = "Extreme Fear"

        score = max(0.0, min(100.0, score))
        return SentimentComponent(
            name="VIX Fear/Greed",
            score=score,
            label=label,
            value_raw=vix_value,
            description=f"India VIX at {vix_value:.1f} — {label}",
        )

    def _score_fii_dii(self) -> SentimentComponent:
        try:
            data: dict[str, Any] = {}
            indices = ["^NSEI", "^NSEBANK"]
            total_net_buy = 0.0
            for idx in indices:
                ticker = yf.Ticker(idx)
                hist = ticker.history(period="5d")
                if len(hist) >= 2:
                    close_list = hist["Close"].tolist()
                    returns = (close_list[-1] - close_list[0]) / close_list[0] * 100
                    total_net_buy += returns

            avg_flow = total_net_buy / max(len(indices), 1)
            if avg_flow > 1.0:
                score = min(95.0, 60.0 + avg_flow * 8.0)
                label = "Greed"
            elif avg_flow > -1.0:
                score = 45.0 + avg_flow * 7.5
                label = "Neutral"
            else:
                score = max(5.0, 40.0 + avg_flow * 6.0)
                label = "Fear"
        except Exception:
            score = 50.0
            label = "Neutral"
            avg_flow = 0.0

        score = max(0.0, min(100.0, score))
        return SentimentComponent(
            name="FII/DII Flow",
            score=score,
            label=label,
            value_raw=round(avg_flow, 2),
            description=f"Estimated institutional flow: {avg_flow:+.2f}% over 5 days — {label}",
        )

    def _score_breadth(self) -> SentimentComponent:
        try:
            advancers = 0
            decliners = 0
            above_dma = 0
            total_stocks = 0
            new_highs = 0
            new_lows = 0

            symbols = [
                "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
                "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
                "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "MARUTI.NS", "TITAN.NS",
                "SUNPHARMA.NS", "ASIANPAINT.NS", "NESTLEIND.NS", "ULTRACEMCO.NS", "WIPRO.NS",
                "TATAMOTORS.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "JSWSTEEL.NS",
                "TATASTEEL.NS", "ADANIENT.NS", "ADANIPORTS.NS", "TECHM.NS", "HCLTECH.NS",
                "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "EICHERMOT.NS", "HEROMOTOCO.NS",
                "BAJAJFINSV.NS", "BRITANNIA.NS", "GRASIM.NS", "HINDALCO.NS", "INDUSINDBK.NS",
                "M&M.NS", "SBILIFE.NS", "HDFCLIFE.NS", "TATACONSUM.NS", "APOLLOHOSP.NS",
                "COALINDIA.NS", "BPCL.NS", "HINDPETRO.NS", "GAIL.NS", "DIVISLAB.NS",
            ]

            for sym in symbols:
                try:
                    t = yf.Ticker(sym)
                    hist = t.history(period="1mo")
                    if len(hist) < 20:
                        continue
                    total_stocks += 1
                    closes = hist["Close"].tolist()
                    if closes[-1] > closes[-2]:
                        advancers += 1
                    else:
                        decliners += 1
                    dma_20 = sum(closes[-20:]) / 20
                    if closes[-1] > dma_20:
                        above_dma += 1
                    high_20 = max(closes[-20:])
                    low_20 = min(closes[-20:])
                    if closes[-1] >= high_20 * 0.99:
                        new_highs += 1
                    if closes[-1] <= low_20 * 1.01:
                        new_lows += 1
                except Exception:
                    continue

            if total_stocks == 0:
                raise ValueError("No stock data available")

            ad_ratio = advancers / max(decliners, 1)
            pct_above_dma = (above_dma / total_stocks) * 100
            hl_ratio = new_highs / max(new_lows, 1)

            ad_score = min(100.0, max(0.0, (ad_ratio / 2.0) * 60.0))
            dma_score = min(100.0, max(0.0, pct_above_dma))
            hl_score = min(100.0, max(0.0, (hl_ratio / 2.0) * 60.0))

            breadth_score = ad_score * 0.4 + dma_score * 0.35 + hl_score * 0.25
        except Exception:
            breadth_score = 50.0
            ad_ratio = 1.0
            pct_above_dma = 50.0
            new_highs = 0
            new_lows = 0
            total_stocks = 0

        label = self._label_from_score(breadth_score)
        return SentimentComponent(
            name="Market Breadth",
            score=breadth_score,
            label=label,
            value_raw={
                "ad_ratio": round(ad_ratio, 2),
                "pct_above_20dma": round(pct_above_dma, 1),
                "new_highs": new_highs,
                "new_lows": new_lows,
                "total_stocks": total_stocks,
            },
            description=(
                f"A/D ratio {ad_ratio:.2f}, {pct_above_dma:.0f}% above 20 DMA, "
                f"{new_highs} new highs / {new_lows} new lows — {label}"
            ),
        )

    def _score_pcr(self) -> SentimentComponent:
        try:
            ticker = yf.Ticker("^NSEI")
            info = ticker.options
            if info:
                nearest_expiry = info[0]
                chain = ticker.option_chain(nearest_expiry)
                put_vol = chain.puts["volume"].sum()
                call_vol = chain.calls["volume"].sum()
                pcr = put_vol / max(call_vol, 1)
            else:
                raise ValueError("No options data")
        except Exception:
            pcr = 1.0

        if pcr > 1.2:
            score = min(95.0, 65.0 + (pcr - 1.2) * 50.0)
            label = "Bullish"
        elif pcr >= 0.8:
            score = 45.0 + (pcr - 0.8) * 50.0
            label = "Neutral"
        else:
            score = max(5.0, 40.0 + (pcr - 0.8) * 75.0)
            label = "Bearish"

        score = max(0.0, min(100.0, score))
        return SentimentComponent(
            name="Put-Call Ratio",
            score=score,
            label=label,
            value_raw=round(pcr, 3),
            description=f"PCR at {pcr:.3f} — {label}",
        )

    def _get_recommendation_impact(self, overall_score: float, label: str) -> str:
        if overall_score >= 80:
            return (
                "High greed environment. Contrarian bearish signal — consider reducing "
                "momentum-heavy positions and adding defensive/value plays. Tighten stop-losses "
                "on speculative trades."
            )
        if overall_score >= 60:
            return (
                "Moderately bullish sentiment. Favour momentum and trend-following strategies. "
                "Maintain standard position sizing with selective sector rotation."
            )
        if overall_score >= 40:
            return (
                "Neutral sentiment — no clear directional bias. Prefer mean-reversion setups "
                "and range-bound strategies. Use hedged option structures for directional bets."
            )
        if overall_score >= 20:
            return (
                "Fear dominating the market. Watch for capitulation reversals and oversold bounces. "
                "Gradually accumulate quality stocks at support levels. Use protective puts."
            )
        return (
            "Extreme fear — potential washout in progress. Historically strong forward returns "
            "from these levels. Deploy capital into high-conviction names with long-term thesis. "
            "Avoid panic selling."
        )

    async def analyze(self) -> SentimentResult:
        if self._is_cache_valid():
            return self._cache  # type: ignore[return-value]

        vix_comp, fii_dii_comp, breadth_comp, pcr_comp = await asyncio.gather(
            asyncio.to_thread(self._score_vix),
            asyncio.to_thread(self._score_fii_dii),
            asyncio.to_thread(self._score_breadth),
            asyncio.to_thread(self._score_pcr),
        )

        overall = (
            vix_comp.score * 0.30
            + fii_dii_comp.score * 0.25
            + breadth_comp.score * 0.25
            + pcr_comp.score * 0.20
        )
        overall = round(max(0.0, min(100.0, overall)), 2)
        overall_label = self._label_from_score(overall)

        components = {
            vix_comp.name: vix_comp,
            fii_dii_comp.name: fii_dii_comp,
            breadth_comp.name: breadth_comp,
            pcr_comp.name: pcr_comp,
        }

        result = SentimentResult(
            overall_score=overall,
            overall_label=overall_label,
            vix_score=vix_comp,
            fii_dii_score=fii_dii_comp,
            breadth_score=breadth_comp,
            pcr_score=pcr_comp,
            components=components,
            timestamp=time.time(),
            recommendations_impact=self._get_recommendation_impact(overall, overall_label),
        )

        self._cache = result
        self._cache_time = time.time()
        return result

    async def get_fear_greed_gauge(self) -> dict[str, Any]:
        result = await self.analyze()
        return {
            "score": result.overall_score,
            "label": result.overall_label,
            "description": (
                f"Market is in a state of {result.overall_label.lower()} "
                f"with a composite score of {result.overall_score:.1f}/100"
            ),
            "timestamp": result.timestamp,
        }

"""Seed realistic demo data on startup so the app has stocks, recommendations,
scores, trades, and analytics to display — even when NSE API is unreachable."""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List

from trademind.config.settings import settings as cfg
from trademind.api.schemas import (
    TradeRecommendationResponse,
    RecommendationHistoryResponse,
    AIScoreResponse,
    CategoryScoreResponse,
    ScoreRankingResponse,
    DashboardStatsResponse,
    LearningStatsResponse,
    PatternResponse,
    WeightResponse,
    WeightHistoryEntryResponse,
    WeightHistoryResponse,
    KnowledgeBaseResponse,
    PaperTradeResponse,
    PortfolioSummary,
)


# Realistic price ranges for F&O symbols
SYMBOL_PRICES: Dict[str, float] = {
    "RELIANCE": 2850.0, "TCS": 3950.0, "HDFCBANK": 1680.0,
    "INFY": 1520.0, "ICICIBANK": 1250.0, "HINDUNILVR": 2380.0,
    "ITC": 465.0, "SBIN": 820.0, "BHARTIARTL": 1540.0,
    "KOTAKBANK": 1780.0, "BAJFINANCE": 6850.0, "LT": 3420.0,
    "WIPRO": 480.0, "AXISBANK": 1120.0, "TITAN": 3250.0,
    "ASIANPAINT": 2850.0, "MARUTI": 12400.0, "SUNPHARMA": 1680.0,
    "NTPC": 380.0, "ONGC": 265.0, "POWERGRID": 310.0,
    "NESTLEIND": 2450.0, "M&M": 2780.0, "TATAMOTORS": 980.0,
    "JSWSTEEL": 890.0, "TECHM": 1650.0, "HCLTECH": 1720.0,
    "BAJAJFINSV": 1650.0, "ULTRACEMCO": 10200.0, "ADANIPORTS": 1280.0,
    "HDFCLIFE": 680.0, "SBILIFE": 1520.0, "DRREDDY": 5850.0,
    "CIPLA": 1420.0, "BRITANNIA": 5250.0, "DIVISLAB": 3850.0,
    "GRASIM": 2350.0, "HINDALCO": 620.0, "EICHERMOT": 4650.0,
    "COALINDIA": 480.0, "BPCL": 680.0, "IOC": 165.0,
    "HEROMOTOCO": 4520.0, "BAJAJ-AUTO": 9200.0, "TATASTEEL": 165.0,
    "BEL": 285.0, "HAL": 4650.0, "ZOMATO": 265.0,
    "ICICIPRULI": 680.0, "MUTHOOTFIN": 1850.0,
}

SECTORS = {
    "RELIANCE": "Energy", "TCS": "IT", "HDFCBANK": "Banking",
    "INFY": "IT", "ICICIBANK": "Banking", "HINDUNILVR": "FMCG",
    "ITC": "FMCG", "SBIN": "Banking", "BHARTIARTL": "Telecom",
    "KOTAKBANK": "Banking", "BAJFINANCE": "Finance", "LT": "Infrastructure",
    "WIPRO": "IT", "AXISBANK": "Banking", "TITAN": "Consumer",
    "ASIANPAINT": "Consumer", "MARUTI": "Auto", "SUNPHARMA": "Pharma",
    "NTPC": "Power", "ONGC": "Energy", "POWERGRID": "Power",
    "NESTLEIND": "FMCG", "M&M": "Auto", "TATAMOTORS": "Auto",
    "JSWSTEEL": "Metal", "TECHM": "IT", "HCLTECH": "IT",
    "BAJAJFINSV": "Finance", "ULTRACEMCO": "Cement", "ADANIPORTS": "Infrastructure",
    "HDFCLIFE": "Finance", "SBILIFE": "Finance", "DRREDDY": "Pharma",
    "CIPLA": "Pharma", "BRITANNIA": "FMCG", "DIVISLAB": "Pharma",
    "GRASIM": "Cement", "HINDALCO": "Metal", "EICHERMOT": "Auto",
    "COALINDIA": "Mining", "BPCL": "Energy", "IOC": "Energy",
    "HEROMOTOCO": "Auto", "BAJAJ-AUTO": "Auto", "TATASTEEL": "Metal",
    "BEL": "Defence", "HAL": "Defence", "ZOMATO": "Tech",
    "ICICIPRULI": "Finance", "MUTHOOTFIN": "Finance",
}

ACTIONS = ["BUY", "SELL", "STRONG_BUY", "STRONG_SELL", "HOLD"]
CONFIDENCE_LEVELS = ["low", "medium", "high", "very_high"]


def _jitter(base: float, pct: float = 0.03) -> float:
    return round(base * (1 + random.uniform(-pct, pct)), 2)


def _past_dt(hours_ago: int) -> datetime:
    return datetime.now() - timedelta(hours=hours_ago)


def seed_recommendations() -> tuple:
    """Populate _active_recommendations and _recommendation_history."""
    from trademind.api.routers.recommendations import (
        _active_recommendations,
        _recommendation_history,
    )

    selected = random.sample(cfg.fno_symbols, min(15, len(cfg.fno_symbols)))
    for sym in selected:
        base = SYMBOL_PRICES.get(sym, 1000.0)
        action = random.choice(ACTIONS)
        entry = _jitter(base, 0.02)
        sl_pct = random.uniform(0.015, 0.04)
        tgt_pct = random.uniform(0.03, 0.08)

        if action in ("SELL", "STRONG_SELL"):
            sl = round(entry * (1 + sl_pct), 2)
            tgt = round(entry * (1 - tgt_pct), 2)
        else:
            sl = round(entry * (1 - sl_pct), 2)
            tgt = round(entry * (1 + tgt_pct), 2)

        rr = round((abs(tgt - entry) / abs(entry - sl)), 2) if entry != sl else 0
        conf = round(random.uniform(0.45, 0.92), 2)
        conf_level = "high" if conf > 0.75 else "medium" if conf > 0.55 else "low"

        evidence = [
            f"RSI indicates {'oversold' if action in ('BUY','STRONG_BUY') else 'overbought'} conditions",
            f"Volume surge detected — {random.randint(120, 350)}% of 20-day average",
            f"MACD {'bullish' if 'BUY' in action else 'bearish'} crossover confirmed",
            f"Option chain PCR at {round(random.uniform(0.6, 1.8), 2)} — {'bullish' if action in ('BUY','STRONG_BUY') else 'bearish'} tilt",
            f" sector momentum: {SECTORS.get(sym, 'N/A')}",
        ]
        if conf > 0.7:
            evidence.append("Institutional buying detected in bulk deals")

        rec_id = f"{sym}-{datetime.now().strftime('%Y%m%d%H%M')}"
        ts = _past_dt(random.randint(0, 6))
        exp_move = round(random.uniform(0.5, 4.0), 2)

        response = TradeRecommendationResponse(
            id=rec_id,
            symbol=sym,
            timestamp=ts,
            action=action,
            entry_price=entry,
            stop_loss=sl,
            target=tgt,
            confidence=conf,
            expected_move_percent=exp_move,
            holding_period="intraday",
            risk_reward_ratio=rr,
            evidence=evidence,
            is_active=True,
        )
        _active_recommendations[sym] = response
        _recommendation_history.append(
            RecommendationHistoryResponse(
                id=rec_id,
                symbol=sym,
                timestamp=ts,
                action=action,
                entry_price=entry,
                stop_loss=sl,
                target=tgt,
                confidence=conf,
                is_active=True,
            )
        )

    return len(selected)


def seed_scores() -> int:
    """Populate _score_cache in scores router."""
    from trademind.api.routers.scores import _score_cache, _score_cache_time
    import trademind.api.routers.scores as scores_mod

    for sym in cfg.fno_symbols:
        conf = round(random.uniform(0.40, 0.95), 2)
        conf_level = (
            "very_high" if conf > 0.85
            else "high" if conf > 0.75
            else "medium" if conf > 0.55
            else "low"
        )

        category_scores = []
        for cat_name, weight in cfg.category_weights.items():
            cat_score = round(random.uniform(20, 95), 1)
            category_scores.append(
                CategoryScoreResponse(name=cat_name, score=cat_score, weight=weight)
            )

        overall = round(sum(c.score * c.weight for c in category_scores), 1)

        evidence = []
        if overall > 70:
            evidence.append(f"Strong bullish signals across {random.randint(3,5)} categories")
        if overall > 50:
            evidence.append("Positive trend alignment")
        evidence.append(f"Momentum score: {overall:.0f}/100")

        response = AIScoreResponse(
            symbol=sym,
            timestamp=_past_dt(random.randint(0, 2)),
            score=overall,
            confidence=conf,
            confidence_level=conf_level,
            category_scores=category_scores,
            evidence=evidence,
            signal_strength=round(random.uniform(0.3, 0.95), 2),
            time_horizon="intraday",
            model_version="v1.2",
            rank=0,
            total_scored=len(cfg.fno_symbols),
        )
        _score_cache[sym] = response

    # assign ranks
    ranked = sorted(_score_cache.items(), key=lambda x: x[1].score, reverse=True)
    for i, (sym, resp) in enumerate(ranked, 1):
        resp.rank = i

    scores_mod._score_cache_time = datetime.now()
    return len(cfg.fno_symbols)


def seed_paper_trades():
    """Pre-populate the paper trading engine with historical demo trades."""
    from trademind.api.routers.trades import _get_paper_engine
    from trademind.database.models import PaperTradeData

    engine = _get_paper_engine()

    symbols = random.sample(cfg.fno_symbols, min(20, len(cfg.fno_symbols)))

    for sym in symbols:
        base = SYMBOL_PRICES.get(sym, 1000.0)
        action = random.choice(["BUY", "SELL", "STRONG_BUY"])
        entry = _jitter(base, 0.02)

        if action in ("SELL", "STRONG_SELL"):
            exit_p = entry * random.uniform(0.97, 1.03)
            sl = round(entry * 1.03, 2)
            tgt = round(entry * 0.95, 2)
        else:
            exit_p = entry * random.uniform(0.97, 1.04)
            sl = round(entry * 0.97, 2)
            tgt = round(entry * 1.05, 2)

        entry_dt = _past_dt(random.randint(2, 168))
        exit_dt = entry_dt + timedelta(hours=random.randint(1, 48))
        qty = random.randint(5, 50)

        trade_id = str(uuid.uuid4())[:12].upper()
        pnl = (Decimal(str(exit_p)) - Decimal(str(entry))) * qty if action in ("BUY", "STRONG_BUY") else (Decimal(str(entry)) - Decimal(str(exit_p))) * qty
        pnl_pct = round(float(pnl) / (entry * qty) * 100, 2)

        reasons = ["TARGET_HIT", "STOP_LOSS_HIT", "TIME_BASED_EXIT", "manual", "TRAILING_STOP"]

        trade = PaperTradeData(
            trade_id=trade_id,
            symbol=sym,
            action=action,
            entry_price=Decimal(str(entry)),
            quantity=qty,
            entry_time=entry_dt,
            stop_loss=Decimal(str(sl)),
            target=Decimal(str(tgt)),
            initial_sl=Decimal(str(sl)),
            current_sl=Decimal(str(sl)),
            exit_price=Decimal(str(round(float(exit_p), 2))),
            exit_time=exit_dt,
            exit_reason=random.choice(reasons),
            pnl=pnl,
            pnl_percent=pnl_pct,
            holding_period_hours=round((exit_dt - entry_dt).total_seconds() / 3600, 2),
            max_drawdown=round(random.uniform(-4, -0.2), 2),
            max_profit=round(random.uniform(0.5, 6), 2),
            status="CLOSED",
            ai_score_at_entry=round(random.uniform(40, 90), 1),
        )
        engine._trades[trade_id] = trade

    # a couple open trades
    for sym in random.sample(cfg.fno_symbols, 3):
        base = SYMBOL_PRICES.get(sym, 1000.0)
        action = "BUY"
        entry = _jitter(base, 0.01)
        qty = random.randint(5, 30)
        sl = round(entry * 0.97, 2)
        tgt = round(entry * 1.05, 2)
        trade_id = str(uuid.uuid4())[:12].upper()

        trade = PaperTradeData(
            trade_id=trade_id,
            symbol=sym,
            action=action,
            entry_price=Decimal(str(entry)),
            quantity=qty,
            entry_time=_past_dt(random.randint(1, 12)),
            stop_loss=Decimal(str(sl)),
            target=Decimal(str(tgt)),
            initial_sl=Decimal(str(sl)),
            current_sl=Decimal(str(sl)),
            status="OPEN",
            ai_score_at_entry=round(random.uniform(50, 85), 1),
        )
        engine._trades[trade_id] = trade


def get_seeded_dashboard_stats() -> DashboardStatsResponse:
    return DashboardStatsResponse(
        ai_accuracy=round(random.uniform(62, 78), 1),
        win_rate=round(random.uniform(52, 68), 1),
        avg_win_pct=round(random.uniform(1.5, 4.5), 2),
        avg_loss_pct=round(random.uniform(-2.5, -0.8), 2),
        profit_factor=round(random.uniform(1.3, 2.8), 2),
        sharpe_ratio=round(random.uniform(0.8, 2.2), 2),
        total_trades=random.randint(80, 200),
        best_scanner="Momentum Breakout",
        worst_scanner="Mean Reversion",
        best_sector="IT",
        best_time="09:30-11:00",
        best_day="Tuesday",
        worst_day="Monday",
        max_drawdown=round(random.uniform(-3, -8), 2),
        max_win_streak=random.randint(5, 12),
        max_loss_streak=random.randint(2, 5),
        avg_holding_hours=round(random.uniform(2, 18), 1),
        expectancy=round(random.uniform(0.5, 2.5), 2),
        total_pnl=round(random.uniform(8000, 35000), 2),
        avg_trade_pnl=round(random.uniform(200, 800), 2),
        kelly_criterion=round(random.uniform(0.08, 0.25), 3),
    )


def get_seeded_learning_stats() -> LearningStatsResponse:
    return LearningStatsResponse(
        total_trades_learned=random.randint(80, 200),
        patterns_found=random.randint(5, 18),
        model_version="v1.2",
        unique_features_tracked=100,
        avg_feature_coverage=round(random.uniform(0.75, 0.95), 2),
        last_learned_at=_past_dt(2),
        accuracy_trend=round(random.uniform(2, 12), 1),
        confidence_calibration=round(random.uniform(0.7, 0.92), 2),
    )


def get_seeded_patterns() -> List[PatternResponse]:
    pattern_data = [
        ("RSI Bullish Divergence", "RSI makes higher low while price makes lower low — bullish reversal signal", "momentum", 0.72, 1.8),
        ("Volume Breakout", "Price breaks resistance with 2x+ volume — continuation likely", "volume", 0.68, 2.1),
        ("MACD Bullish Crossover", "MACD line crosses above signal line with histogram expanding", "momentum", 0.65, 1.5),
        ("Bollinger Band Squeeze", "Bollinger bands narrow to <2% of price — breakout imminent", "volatility", 0.71, 2.4),
        ("Institutional Accumulation", "Bulk deals + rising FII holding in last 3 sessions", "institutional", 0.78, 2.8),
        ("Sector Rotation Signal", "Capital flowing from underperforming to leading sector", "sector", 0.63, 1.9),
        ("PCR Extreme Reading", "Put-Call ratio reaches 1.5+ or 0.5- — contrarian signal", "options", 0.69, 2.0),
        ("Gap Fill Pattern", "Stock gaps down/up then reverses to fill gap within 2 days", "price_action", 0.66, 1.4),
        ("Moving Average Golden Cross", "50 DMA crosses above 200 DMA — long-term bullish", "trend", 0.74, 3.2),
        ("VWAP Reclaim", "Price reclaims VWAP after trading below — intraday bullish", "intraday", 0.62, 1.3),
    ]

    results = []
    for name, desc, cat, sr, ar in pattern_data:
        tc = random.randint(15, 80)
        wc = int(tc * sr)
        results.append(
            PatternResponse(
                name=name,
                description=desc,
                conditions={"category": cat, "min_trades": 10},
                category=cat,
                trade_count=tc,
                win_count=wc,
                success_rate=round(sr * 100, 1),
                avg_return=round(ar, 2),
                confidence=round(random.uniform(0.6, 0.9), 2),
            )
        )
    return results


def get_seeded_weights() -> List[WeightResponse]:
    return [
        WeightResponse(category=cat, weight=round(w + random.uniform(-0.02, 0.02), 2))
        for cat, w in sorted(cfg.category_weights.items())
    ]


def get_seeded_weight_history() -> WeightHistoryResponse:
    entries = []
    for i, (cat, w) in enumerate(cfg.category_weights.items()):
        entries.append(
            WeightHistoryEntryResponse(
                id=f"wh-{i+1:03d}",
                category=cat,
                feature_name=f"{cat}_momentum",
                old_weight=round(w - random.uniform(-0.05, 0.05), 2),
                new_weight=round(w, 2),
                change_reason=f"Optimized after {random.randint(10,50)} trades — improved win rate by {random.randint(2,8)}%",
                effective_date=_past_dt(random.randint(24, 168)),
                trade_count_sample=random.randint(10, 50),
                avg_impact=round(random.uniform(0.5, 3.0), 2),
            )
        )
    return WeightHistoryResponse(entries=entries, total=len(entries))


def get_seeded_knowledge_base() -> List[KnowledgeBaseResponse]:
    return [
        KnowledgeBaseResponse(
            id="kb-001",
            version="v1.2",
            model_name="TradeMind Ensemble v1.2",
            model_type="ensemble",
            trades_learned=random.randint(100, 200),
            patterns_found=random.randint(8, 15),
            accuracy=round(random.uniform(0.62, 0.75), 3),
            precision_score=round(random.uniform(0.60, 0.78), 3),
            recall_score=round(random.uniform(0.55, 0.72), 3),
            f1_score=round(random.uniform(0.58, 0.74), 3),
            confidence=round(random.uniform(0.7, 0.9), 2),
            training_data_start=_past_dt(720),
            training_data_end=_past_dt(1),
            is_active=True,
        )
    ]

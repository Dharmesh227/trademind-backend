"""APScheduler task definitions for TradeMind AI background jobs.

These are async functions called by AsyncIOScheduler which handles the event loop.
Each task wraps an engine call and logs success/failure.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from trademind.config.settings import settings as cfg
from trademind.config.constants import MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE, MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE
from trademind.scheduler.cache import data_cache

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))


# ── NSE Session Refresh ────────────────────────────────────
async def nse_session_refresh_task() -> None:
    """Runs every 4 minutes — keeps NSE cookies fresh."""
    try:
        from trademind.engines.nse_client import NSEClient
        client = await NSEClient.get()
        ok = await client._refresh_cookies()
        if ok:
            logger.info("NSE session refresh: cookies updated")
        else:
            logger.warning("NSE session refresh: failed to obtain cookies")
    except Exception as exc:
        logger.error("NSE session refresh task failed: %s", exc)


def _is_market_hours() -> bool:
    """Check if current IST time is within market hours."""
    now = datetime.now(IST)
    market_open = now.replace(
        hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0, microsecond=0
    )
    market_close = now.replace(
        hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0, microsecond=0
    )
    return market_open <= now <= market_close


# ── Sector Rotation Update ──────────────────────────────────
async def sector_rotation_update_task() -> None:
    """Runs every 5 min during market hours — updates sector momentum tracker."""
    if not _is_market_hours():
        return

    from trademind.engines.bhavcopy.engine import BhavcopyEngine

    try:
        engine = BhavcopyEngine()
        data = await engine.get_bhavcopy()

        if not data.indices:
            return

        from trademind.engines.sector_rotation.tracker import sector_tracker

        today = datetime.now(IST).strftime("%Y-%m-%d")
        index_returns = {}
        for name, idx in data.indices.items():
            if idx.change_percent is not None:
                index_returns[name] = float(idx.change_percent)

        if index_returns:
            sector_tracker.record_daily_returns(today, index_returns)
            logger.info(
                "Sector rotation: recorded returns for %d sectors (total data points: %d)",
                len(index_returns), sector_tracker.data_points,
            )
    except Exception as e:
        logger.error("Sector rotation update failed: %s", e)


# ── Market Data Collection ─────────────────────────────────
async def market_data_collection_task() -> None:
    """Runs every 5 minutes during market hours — collects OHLCV + options."""
    if not _is_market_hours():
        return

    from trademind.engines.market_data.collector import MarketDataCollector

    collector = MarketDataCollector()
    try:
        tasks = [collector.collect_all_for_symbol(sym) for sym in cfg.fno_symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success = sum(
            1 for r in results
            if not isinstance(r, Exception) and r[0] is not None
        )
        logger.info(
            "Market data collection: {}/{} symbols",
            success,
            len(cfg.fno_symbols),
        )
    except Exception as e:
        logger.error("Market data collection failed: {}", e)
    finally:
        await collector.close()


# ── Option Chain Collection ─────────────────────────────────
async def option_chain_collection_task() -> None:
    """Runs every 5 min during market hours — collects option chain data for top symbols.

    Results are stored in data_cache.option_chains for use by feature engineering.
    Silently skips when NSE is unreachable.
    """
    if not _is_market_hours():
        return

    from trademind.engines.market_data.collector import MarketDataCollector

    collector = MarketDataCollector()
    try:
        top_symbols = cfg.fno_symbols[:10]
        results = await asyncio.gather(
            *[collector.collect_option_chain(sym) for sym in top_symbols],
            return_exceptions=True,
        )
        collected = 0
        for sym, result in zip(top_symbols, results):
            if not isinstance(result, Exception) and result is not None:
                data_cache.option_chains[sym] = result
                collected += 1

        if collected:
            data_cache.update_timestamp("option_chains")
            logger.info("Option chain collection: %d/%d symbols", collected, len(top_symbols))
    except Exception as e:
        logger.error("Option chain collection task failed: %s", e)
    finally:
        await collector.close()


# ── Market Breadth Collection ───────────────────────────────
async def market_breadth_task() -> None:
    """Runs every 5 min during market hours — collects advance/decline data.

    Result stored in data_cache.market_breadth for feature engineering.
    """
    if not _is_market_hours():
        return

    from trademind.engines.market_data.collector import MarketDataCollector

    collector = MarketDataCollector()
    try:
        breadth = await collector.collect_market_breadth()
        if breadth:
            data_cache.market_breadth = breadth
            data_cache.update_timestamp("market_breadth")
            logger.info(
                "Market breadth: %d advances / %d declines (A/D=%.2f)",
                breadth.advances, breadth.declines,
                float(breadth.advance_decline_ratio) if breadth.advance_decline_ratio else 0,
            )
    except Exception as e:
        logger.error("Market breadth collection failed: %s", e)
    finally:
        await collector.close()


# ── Institutional Flow (FII/DII) Collection ────────────────
async def institutional_flow_task() -> None:
    """Runs hourly during market hours — collects FII/DII cash flow data.

    Result stored in data_cache.institutional_flow for feature engineering.
    """
    if not _is_market_hours():
        return

    from trademind.engines.market_data.collector import MarketDataCollector

    collector = MarketDataCollector()
    try:
        flow = await collector.collect_fii_dii_flow()
        if flow:
            data_cache.institutional_flow = flow
            data_cache.update_timestamp("institutional_flow")
            logger.info("Institutional flow: FII=%.0f, DII=%.0f",
                         float(flow.fii_cash or 0), float(flow.dii_cash or 0))
    except Exception as e:
        logger.error("Institutional flow collection failed: %s", e)
    finally:
        await collector.close()


# ── VIX Collection ─────────────────────────────────────────
async def vix_collection_task() -> None:
    """Runs every 5 min during market hours — collects India VIX."""
    if not _is_market_hours():
        return

    from trademind.engines.market_data.collector import MarketDataCollector

    collector = MarketDataCollector()
    try:
        vix = await collector.collect_vix()
        if vix:
            data_cache.vix_data = vix
            data_cache.update_timestamp("vix")
            logger.info("VIX: %.2f (change=%.2f%%)",
                         float(vix.value), float(vix.change_percent or 0))
    except Exception as e:
        logger.error("VIX collection failed: %s", e)
    finally:
        await collector.close()


# ── Feature Engineering ────────────────────────────────────
async def feature_engineering_task() -> None:
    """Populates FeatureExtractor with Yahoo history + computes 100+ features.

    Stores results in data_cache.feature_cache for use by AI scoring.
    """
    if not _is_market_hours():
        return

    from trademind.engines.feature_engine.extractor import FeatureExtractor
    from trademind.engines.bhavcopy.yahoo_history import yahoo_history
    from trademind.engines.bhavcopy.engine import BhavcopyEngine
    from trademind.database.models import PriceData
    from decimal import Decimal

    extractor = FeatureExtractor()

    try:
        populated = await yahoo_history.populate_history(cfg.fno_symbols, extractor)
        logger.info("Feature engineering: %d symbols populated from Yahoo history", populated)

        bhavcopy = BhavcopyEngine()
        data = await bhavcopy.get_bhavcopy()

        for symbol in cfg.fno_symbols:
            stock = data.stocks.get(symbol)
            if not stock or stock.close <= 0:
                continue

            try:
                current_price = PriceData(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    open=Decimal(str(stock.open or stock.close)),
                    high=Decimal(str(stock.high or stock.close)),
                    low=Decimal(str(stock.low or stock.close)),
                    close=Decimal(str(stock.close)),
                    volume=int(stock.volume),
                    vwap=Decimal(str(round((stock.high + stock.low + stock.close) / 3, 2)) if stock.high and stock.low else stock.close),
                    prev_close=Decimal(str(stock.prev_close)) if stock.prev_close else None,
                    change_percent=Decimal(str(stock.change_pct)),
                )

                if symbol not in extractor._price_history or not extractor._price_history[symbol]:
                    extractor._price_history[symbol] = [current_price]
                elif extractor._price_history[symbol][-1].close != current_price.close:
                    extractor._price_history[symbol].append(current_price)

                option_data = data_cache.option_chains.get(symbol)

                features = extractor.compute_all_features(
                    symbol=symbol,
                    market_data=current_price,
                    option_data=option_data,
                    breadth=data_cache.market_breadth,
                    inst=data_cache.institutional_flow,
                    vix=data_cache.vix_data,
                )

                data_cache.feature_cache[symbol] = features
                data_cache.price_cache[symbol] = current_price
            except Exception as e:
                logger.warning("Feature extraction failed for %s: %s", symbol, e)

        data_cache.update_timestamp("features")
    except Exception as e:
        logger.error("Feature engineering task failed: %s", e)


# ── AI Scoring ─────────────────────────────────────────────
async def ai_scoring_task() -> None:
    """Computes AI scores 0-100 using cached features from feature engineering.

    Stores results in data_cache.scoring_results for use by recommendation generation.
    """
    if not _is_market_hours():
        return

    from trademind.engines.ai_scoring.scorer import AIScoreEngine

    scorer = AIScoreEngine()
    scored = 0
    all_scores: dict[str, float] = {}

    try:
        for symbol in cfg.fno_symbols:
            features = data_cache.feature_cache.get(symbol)
            current_price = data_cache.price_cache.get(symbol)
            if not features or not current_price:
                continue

            try:
                result = scorer.score_symbol(symbol, features)
                all_scores[symbol] = result.overall_score
                scored += 1
            except Exception as e:
                logger.warning("AI scoring failed for %s: %s", symbol, e)

        data_cache.scoring_results = all_scores
        data_cache.update_timestamp("scoring")

        logger.info("AI scoring: %d symbols scored (avg=%.1f)", scored,
                     sum(all_scores.values()) / len(all_scores) if all_scores else 0)
    except Exception as e:
        logger.error("AI scoring task failed: %s", e)


# ── Recommendation Generation ──────────────────────────────
async def recommendation_generation_task() -> None:
    """Generates actionable trade signals from cached AI scores.

    Reads from data_cache.scoring_results + data_cache.feature_cache
    (produced by ai_scoring_task and feature_engineering_task).
    """
    if not _is_market_hours():
        return

    from trademind.engines.ai_scoring.scorer import AIScoreEngine
    from trademind.engines.recommendation.engine import RecommendationEngine
    from trademind.api.routers.recommendations import _active_recommendations
    from decimal import Decimal

    scorer = AIScoreEngine()
    rec_engine = RecommendationEngine()
    generated = 0

    try:
        for symbol in cfg.fno_symbols:
            features = data_cache.feature_cache.get(symbol)
            current_price = data_cache.price_cache.get(symbol)
            ai_score_val = data_cache.scoring_results.get(symbol)

            if not features or not current_price or ai_score_val is None:
                continue

            try:
                from trademind.database.models import AIScoreResult
                from datetime import datetime as _dt
                ai_score = AIScoreResult(
                    symbol=symbol,
                    timestamp=_dt.now(),
                    overall_score=ai_score_val,
                )

                if ai_score_val >= cfg.confidence_medium_threshold * 100:
                    rec = rec_engine.generate_recommendation(
                        symbol=symbol,
                        ai_score=ai_score,
                        features=features,
                        market_data=current_price,
                        all_scores=data_cache.scoring_results,
                    )
                    generated += 1
            except Exception as e:
                logger.warning("Recommendation failed for %s: %s", symbol, e)

        logger.info("Recommendation generation: %d recommendations from %d scored symbols",
                     generated, len(data_cache.scoring_results))
    except Exception as e:
        logger.error("Recommendation generation task failed: %s", e)


# ── Paper Trade Monitoring ─────────────────────────────────
async def paper_trade_monitoring_task() -> None:
    """Runs every minute — checks SL, target, trailing stop, time exit.

    When a trade is closed by monitoring, triggers learning automatically.
    """
    if not _is_market_hours():
        return

    from trademind.api.routers.trades import _get_paper_engine
    from trademind.engines.market_data.collector import MarketDataCollector

    engine = _get_paper_engine()
    open_trades = engine.get_open_positions()

    if not open_trades:
        return

    symbols = list({t.symbol for t in open_trades})
    collector = MarketDataCollector()
    current_prices = {}

    try:
        for symbol in symbols:
            try:
                price_data = await collector.collect_price_data(symbol)
                if price_data:
                    current_prices[symbol] = price_data
            except Exception as e:
                logger.warning("Price fetch failed for {}: {}", symbol, e)

        if current_prices:
            closed = await engine.monitor_open_trades(current_prices)
            if closed:
                logger.info("Closed {} trades via monitoring", len(closed))
                await _trigger_learning_for_closed_trades(closed)
    except Exception as e:
        logger.error("Paper trade monitoring failed: {}", e)
    finally:
        await collector.close()


# ── Learning Task (event-driven) ───────────────────────────
async def learning_task() -> None:
    """Runs after trade closure — updates model knowledge."""
    from trademind.engines.learning.engine import LearningEngine

    engine = LearningEngine()
    try:
        stats = await engine.get_learning_stats()
        logger.info(
            "Learning: {} trades learned, trend: {:.1f}%",
            stats.total_trades_learned,
            stats.accuracy_trend,
        )
    except Exception as e:
        logger.error("Learning task failed: {}", e)
    finally:
        await engine.close()


async def learn_from_closed_trade(trade) -> None:
    """Trigger learning engine for a single closed trade."""
    from trademind.engines.learning.engine import LearningEngine

    engine = LearningEngine()
    try:
        features = data_cache.feature_cache.get(trade.symbol, {})
        if not features:
            features = {}
        await engine.learn_from_trade(trade, features)
    except Exception as e:
        logger.error("Learning from trade %s failed: %s", trade.trade_id, e)
    finally:
        await engine.close()


async def _trigger_learning_for_closed_trades(closed_trades) -> None:
    """Fire-and-forget learning for each newly closed trade."""
    for trade in closed_trades:
        if trade is not None:
            try:
                await learn_from_closed_trade(trade)
            except Exception as e:
                logger.warning("Learning trigger failed for %s: %s", trade.trade_id, e)


# ── Weight Optimization ────────────────────────────────────
async def weight_optimization_task() -> None:
    """Runs weekly (weekend) — optimizes adaptive category weights."""
    from trademind.engines.adaptive_weights.optimizer import AdaptiveWeightOptimizer

    optimizer = AdaptiveWeightOptimizer()
    try:
        should_update = await optimizer._require_minimum_samples_async(
            "all", cfg.min_trades_for_learning
        )
        if should_update:
            result = await optimizer.optimize_weights()
            logger.info(
                "Weight optimization: applied={}, reason={}",
                result.was_applied,
                result.reason,
            )
        else:
            logger.info("Weight optimization skipped: insufficient samples")
    except Exception as e:
        logger.error("Weight optimization task failed: {}", e)
    finally:
        await optimizer.close()


# ── Pattern Discovery ──────────────────────────────────────
async def pattern_discovery_task() -> None:
    """Runs weekly — mines new patterns from completed trades."""
    from trademind.engines.pattern_discovery.engine import PatternDiscoveryEngine

    engine = PatternDiscoveryEngine()
    try:
        patterns = await engine.discover_patterns()
        stats = await engine.get_pattern_stats()
        logger.info(
            "Pattern discovery: {} patterns, best='{}' ({:.0%})",
            stats.total_patterns,
            stats.best_pattern,
            stats.best_pattern_accuracy,
        )
    except Exception as e:
        logger.error("Pattern discovery task failed: {}", e)
    finally:
        await engine.close()


# ── Bhavcopy Refresh ─────────────────────────────────────
async def bhavcopy_refresh_task() -> None:
    """Runs daily at 8 PM IST — refreshes F&O Bhavcopy data from NSE."""
    try:
        from trademind.engines.bhavcopy.engine import BhavcopyEngine
        engine = BhavcopyEngine()
        data = await engine.get_bhavcopy()
        if data.stocks:
            logger.info(
                "Bhavcopy refresh: %d stocks, %d indices, %d options (source=%s)",
                data.fo_count, len(data.indices), data.option_count, data.source,
            )
        else:
            logger.warning("Bhavcopy refresh: no data available")
    except Exception as exc:
        logger.error("Bhavcopy refresh failed: %s", exc)


# ── Yahoo Finance Refresh ──────────────────────────────────
async def yahoo_refresh_task() -> None:
    """Runs every 5 min during market hours — fetches delayed prices from Yahoo Finance.

    Merges live prices into the BhavcopyEngine cache so scores/recommendations
    use the latest available data during market hours.
    """
    if not _is_market_hours():
        return

    try:
        from trademind.engines.bhavcopy.yahoo import yahoo_provider
        from trademind.engines.bhavcopy.engine import BhavcopyEngine

        engine = BhavcopyEngine()
        data = await engine.get_bhavcopy()
        stock_symbols = list(data.stocks.keys()) if data.stocks else cfg.fno_symbols

        stocks, indices = await yahoo_provider.fetch_all(
            stock_symbols, cfg.index_symbols,
        )
        if stocks:
            engine = BhavcopyEngine()
            data = await engine.get_bhavcopy()

            for sym, y_stock in stocks.items():
                if sym in data.stocks:
                    bh = data.stocks[sym]
                    bh.close = y_stock.close
                    bh.open = y_stock.open
                    bh.high = y_stock.high
                    bh.low = y_stock.low
                    bh.change = y_stock.change
                    bh.change_pct = y_stock.change_pct
                    bh.volume = y_stock.volume
                    bh.prev_close = y_stock.prev_close
                else:
                    data.stocks[sym] = y_stock

            data.fo_count = len(data.stocks)
            logger.info("Yahoo refresh: %d stocks updated", len(stocks))
    except Exception as exc:
        logger.error("Yahoo refresh task failed: %s", exc)


# ── Auto Trade Execution ───────────────────────────────────
async def auto_trade_task() -> None:
    """Runs every 5 min during market hours — auto-executes paper trades from high-confidence recommendations."""
    if not _is_market_hours():
        return

    from trademind.api.routers.recommendations import _active_recommendations
    from trademind.api.routers.trades import _get_paper_engine
    from trademind.engines.auto_trader.engine import auto_trader

    if not _active_recommendations:
        return

    paper_engine = _get_paper_engine()

    try:
        actions = auto_trader.process_recommendations(_active_recommendations, paper_engine)
        executed = [a for a in actions if a["status"] == "executed"]
        if executed:
            logger.info(
                "Auto-trade: %d trades executed — %s",
                len(executed),
                ", ".join(f"{a['symbol']}({a['action']})" for a in executed),
            )
    except Exception as e:
        logger.error("Auto-trade task failed: %s", e)

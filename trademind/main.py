"""TradeMind AI — FastAPI application entry point.

No dummy/seed data. All endpoints return real data from NSE or empty with a message.
"""

from __future__ import annotations

import asyncio
import time
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from trademind import __version__
from trademind.config.settings import settings
from trademind.core.exceptions import TradeMindError
from trademind.core.logger import setup_logging
from trademind.core.nse_status import nse_status
from trademind.database.connection import dispose_engine, init_db
from trademind.scheduler.manager import scheduler_manager

logger = logging.getLogger(__name__)

_app_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup / shutdown lifecycle."""
    global _app_start_time  # noqa: PLW0603
    _app_start_time = time.time()

    setup_logging(log_level=settings.log_level if hasattr(settings, "log_level") else "INFO")
    logger.info("TradeMind AI starting up — version={}", __version__)

    await init_db()
    logger.info("Database initialized")

    scheduler_manager.start_scheduler()
    logger.info("Scheduler started")

    # ── Probe NSE reachability + pre-warm Bhavcopy ─────────────
    try:
        from trademind.engines.bhavcopy.engine import BhavcopyEngine
        bhavcopy = BhavcopyEngine()
        data = await bhavcopy.get_bhavcopy()
        if data.stocks or data.indices:
            nse_status.update(
                is_live=True,
                indices_count=len(data.indices),
                message=f"Bhavcopy loaded: {data.fo_count} stocks, {len(data.indices)} indices",
            )
            # Cache index data for market router
            from trademind.api.routers import market as _market_mod
            from trademind.api.schemas import IndexDataResponse
            cached = [
                IndexDataResponse(
                    symbol=idx.symbol,
                    timestamp=datetime.now(),
                    open=idx.open,
                    high=idx.high,
                    low=idx.low,
                    close=idx.last,
                    change_percent=idx.change_pct,
                    volume=int(idx.traded_volume) if idx.traded_volume else None,
                )
                for name, idx in data.indices.items()
                if name in settings.index_symbols
            ]
            _market_mod._indices_cache = cached
            logger.info(
                "NSE live via Bhavcopy — %d stocks, %d indices cached",
                data.fo_count, len(cached),
            )
        else:
            nse_status.update(is_live=False, last_error="No data in Bhavcopy response")
            logger.info("NSE not available — all endpoints return empty with message")
    except Exception as exc:
        nse_status.update(is_live=False, last_error=str(exc))
        logger.info("NSE not reachable: %s", exc)

    yield

    logger.info("TradeMind AI shutting down")
    scheduler_manager.stop_scheduler()
    await dispose_engine()


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title=settings.app_name if hasattr(settings, "app_name") else "TradeMind AI",
        version=__version__,
        description="Self-Improving NSE Stock Trading Intelligence Platform",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS Middleware (Android app support) ───────────────
    cors_origins = getattr(settings, "cors_origins", ["*"])
    if isinstance(cors_origins, str):
        import json
        try:
            cors_origins = json.loads(cors_origins)
        except (json.JSONDecodeError, TypeError):
            cors_origins = [cors_origins]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Total-Count"],
    )

    # ── Global Error Handling Middleware ────────────────────
    @app.middleware("http")
    async def error_handling_middleware(request: Request, call_next):
        try:
            return await call_next(request)
        except TradeMindError as exc:
            logger.error("TradeMindError: {} — {}", exc.message, exc.details)
            return JSONResponse(
                status_code=422,
                content={"detail": exc.message, "code": exc.__class__.__name__, "details": exc.details},
            )
        except Exception as exc:
            logger.exception("Unhandled exception: {}", exc)
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "code": "InternalServerError"},
            )

    # ── Import and Include Routers ─────────────────────────
    from trademind.api.routers import market, scores, recommendations, trades, analytics, learning, backtest, sector_analysis, patterns, heatmap, correlation, notifications, portfolio, sentiment

    api_prefix = "/api/v1"
    app.include_router(market.router, prefix=api_prefix)
    app.include_router(scores.router, prefix=api_prefix)
    app.include_router(recommendations.router, prefix=api_prefix)
    app.include_router(trades.router, prefix=api_prefix)
    app.include_router(analytics.router, prefix=api_prefix)
    app.include_router(learning.router, prefix=api_prefix)
    app.include_router(learning.knowledge_router, prefix=api_prefix)
    app.include_router(backtest.router, prefix=api_prefix)
    app.include_router(sector_analysis.router, prefix=api_prefix)
    app.include_router(patterns.router, prefix=api_prefix)
    app.include_router(heatmap.router, prefix=api_prefix)
    app.include_router(correlation.router, prefix=api_prefix)
    app.include_router(notifications.router, prefix=api_prefix)
    app.include_router(portfolio.router, prefix=api_prefix)
    app.include_router(sentiment.router, prefix=api_prefix)

    # ── Health Check + NSE Status ──────────────────────────
    @app.get("/health", tags=["System"])
    async def health() -> dict:
        uptime = time.time() - _app_start_time if _app_start_time else 0.0
        scheduler_status = scheduler_manager.get_scheduler_status()
        return {
            "status": "ok",
            "version": __version__,
            "database": "connected",
            "scheduler": scheduler_status.get("status", "unknown"),
            "scheduler_jobs": scheduler_status.get("job_count", 0),
            "uptime_seconds": round(uptime, 1),
            "nse": nse_status.to_dict(),
        }

    @app.get("/api/v1/system/nse-status", tags=["System"])
    async def get_nse_status() -> dict:
        return nse_status.to_dict()

    # ── Scheduler Management Endpoints ─────────────────────
    @app.get("/api/v1/system/scheduler", tags=["System"])
    async def get_scheduler_status() -> dict:
        return scheduler_manager.get_scheduler_status()

    @app.post("/api/v1/system/scheduler/{job_id}/pause", tags=["System"])
    async def pause_scheduler_job(job_id: str) -> dict:
        success = scheduler_manager.pause_job(job_id)
        if not success:
            return JSONResponse(status_code=404, content={"detail": f"Job '{job_id}' not found"})
        return {"status": "paused", "job_id": job_id}

    @app.post("/api/v1/system/scheduler/{job_id}/resume", tags=["System"])
    async def resume_scheduler_job(job_id: str) -> dict:
        success = scheduler_manager.resume_job(job_id)
        if not success:
            return JSONResponse(status_code=404, content={"detail": f"Job '{job_id}' not found"})
        return {"status": "resumed", "job_id": job_id}

    @app.post("/api/v1/system/scheduler/{job_id}/run", tags=["System"])
    async def trigger_scheduler_job(job_id: str) -> dict:
        success = scheduler_manager.run_job_now(job_id)
        if not success:
            return JSONResponse(status_code=404, content={"detail": f"Job '{job_id}' not found"})
        return {"status": "triggered", "job_id": job_id}

    @app.get("/api/v1/system/stats", tags=["System"])
    async def system_stats() -> dict:
        uptime = time.time() - _app_start_time if _app_start_time else 0.0
        return {
            "version": __version__,
            "symbols_tracked": len(settings.fno_symbols),
            "index_tracked": len(settings.index_symbols),
            "scheduler_jobs": len(scheduler_manager.get_job_status()),
            "uptime_seconds": round(uptime, 1),
            "collection_interval": settings.collection_interval,
            "max_open_positions": settings.max_open_positions,
            "default_capital": settings.default_capital,
            "nse_live": nse_status.is_live,
        }

    return app


app = create_app()

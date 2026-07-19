"""SchedulerManager — configures and controls all APScheduler jobs."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from trademind.scheduler.tasks import (
    ai_scoring_task,
    auto_trade_task,
    bhavcopy_refresh_task,
    feature_engineering_task,
    institutional_flow_task,
    learning_task,
    market_breadth_task,
    market_data_collection_task,
    nse_session_refresh_task,
    option_chain_collection_task,
    paper_trade_monitoring_task,
    pattern_discovery_task,
    recommendation_generation_task,
    sector_rotation_update_task,
    vix_collection_task,
    weight_optimization_task,
    yahoo_refresh_task,
)

logger = logging.getLogger(__name__)


class SchedulerManager:
    """Manages the APScheduler lifecycle and all scheduled jobs.

    Job Schedule (staggered to avoid thundering herd):
    T+0s   : market_data_collection, nse_session_refresh
    T+30s  : yahoo_refresh
    T+60s  : option_chain_collection, vix_collection
    T+90s  : market_breadth, institutional_flow
    T+120s : feature_engineering
    T+180s : ai_scoring
    T+240s : recommendation_generation
    T+300s : auto_trade
    Every 1m: paper_trade_monitoring
    Daily 20:00 IST: bhavcopy_refresh
    Weekly Sat 02:00 IST: weight_optimization
    Weekly Sun 03:00 IST: pattern_discovery
    """

    def __init__(self) -> None:
        self._scheduler: Optional[AsyncIOScheduler] = None
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    def _stagger(self, seconds: int) -> datetime:
        """Return a next_run_time offset from now by `seconds`."""
        return datetime.now() + timedelta(seconds=seconds)

    def start_scheduler(self) -> None:
        """Configure and start all scheduled jobs."""
        if self._is_running and self._scheduler and self._scheduler.running:
            logger.warning("Scheduler already running")
            return

        self._scheduler = AsyncIOScheduler(
            timezone="Asia/Kolkata",
            job_defaults={
                "coalesce": True,
                "max_instances": 1,
                "misfire_grace_time": 300,
            },
        )

        # ── T+0s: Market Data Collection ───────────────────
        self._scheduler.add_job(
            market_data_collection_task,
            trigger=IntervalTrigger(minutes=5),
            id="market_data_collection",
            name="Market Data Collection",
            replace_existing=True,
        )

        # ── T+0s: NSE Session Refresh (every 4 min) ───────
        self._scheduler.add_job(
            nse_session_refresh_task,
            trigger=IntervalTrigger(minutes=4),
            id="nse_session_refresh",
            name="NSE Session Refresh",
            replace_existing=True,
        )

        # ── T+30s: Yahoo Finance Refresh ──────────────────
        self._scheduler.add_job(
            yahoo_refresh_task,
            trigger=IntervalTrigger(minutes=5),
            id="yahoo_refresh",
            name="Yahoo Finance Refresh",
            replace_existing=True,
            next_run_time=self._stagger(30),
        )

        # ── T+60s: Option Chain Collection ────────────────
        self._scheduler.add_job(
            option_chain_collection_task,
            trigger=IntervalTrigger(minutes=5),
            id="option_chain_collection",
            name="Option Chain Collection",
            replace_existing=True,
            next_run_time=self._stagger(60),
        )

        # ── T+60s: VIX Collection ─────────────────────────
        self._scheduler.add_job(
            vix_collection_task,
            trigger=IntervalTrigger(minutes=5),
            id="vix_collection",
            name="VIX Collection",
            replace_existing=True,
            next_run_time=self._stagger(60),
        )

        # ── T+90s: Market Breadth ─────────────────────────
        self._scheduler.add_job(
            market_breadth_task,
            trigger=IntervalTrigger(minutes=5),
            id="market_breadth",
            name="Market Breadth",
            replace_existing=True,
            next_run_time=self._stagger(90),
        )

        # ── T+90s: Institutional Flow (hourly) ────────────
        self._scheduler.add_job(
            institutional_flow_task,
            trigger=IntervalTrigger(hours=1),
            id="institutional_flow",
            name="Institutional Flow (FII/DII)",
            replace_existing=True,
            next_run_time=self._stagger(90),
        )

        # ── T+90s: Sector Rotation Update ─────────────────
        self._scheduler.add_job(
            sector_rotation_update_task,
            trigger=IntervalTrigger(minutes=5),
            id="sector_rotation_update",
            name="Sector Rotation Update",
            replace_existing=True,
            next_run_time=self._stagger(90),
        )

        # ── T+120s: Feature Engineering ───────────────────
        self._scheduler.add_job(
            feature_engineering_task,
            trigger=IntervalTrigger(minutes=5),
            id="feature_engineering",
            name="Feature Engineering",
            replace_existing=True,
            next_run_time=self._stagger(120),
        )

        # ── T+180s: AI Scoring ────────────────────────────
        self._scheduler.add_job(
            ai_scoring_task,
            trigger=IntervalTrigger(minutes=5),
            id="ai_scoring",
            name="AI Scoring",
            replace_existing=True,
            next_run_time=self._stagger(180),
        )

        # ── T+240s: Recommendation Generation ─────────────
        self._scheduler.add_job(
            recommendation_generation_task,
            trigger=IntervalTrigger(minutes=5),
            id="recommendation_generation",
            name="Recommendation Generation",
            replace_existing=True,
            next_run_time=self._stagger(240),
        )

        # ── T+300s: Auto Trade Execution ──────────────────
        self._scheduler.add_job(
            auto_trade_task,
            trigger=IntervalTrigger(minutes=5),
            id="auto_trade_execution",
            name="Auto Trade Execution",
            replace_existing=True,
            next_run_time=self._stagger(300),
        )

        # ── Every 1 min: Paper Trade Monitoring ────────────
        self._scheduler.add_job(
            paper_trade_monitoring_task,
            trigger=IntervalTrigger(minutes=1),
            id="paper_trade_monitoring",
            name="Paper Trade Monitoring",
            replace_existing=True,
        )

        # ── Daily 8 PM IST: Bhavcopy Refresh ──────────────
        self._scheduler.add_job(
            bhavcopy_refresh_task,
            trigger=CronTrigger(
                hour=20,
                minute=0,
                timezone="Asia/Kolkata",
            ),
            id="bhavcopy_refresh",
            name="Bhavcopy Refresh",
            replace_existing=True,
        )

        # ── Weekly Saturday 02:00 IST: Weight Optimization ─
        self._scheduler.add_job(
            weight_optimization_task,
            trigger=CronTrigger(
                day_of_week="sat",
                hour=2,
                minute=0,
                timezone="Asia/Kolkata",
            ),
            id="weight_optimization",
            name="Weight Optimization",
            replace_existing=True,
        )

        # ── Weekly Sunday 03:00 IST: Pattern Discovery ────
        self._scheduler.add_job(
            pattern_discovery_task,
            trigger=CronTrigger(
                day_of_week="sun",
                hour=3,
                minute=0,
                timezone="Asia/Kolkata",
            ),
            id="pattern_discovery",
            name="Pattern Discovery",
            replace_existing=True,
        )

        self._scheduler.start()
        self._is_running = True

        job_count = len(self._scheduler.get_jobs())
        logger.info(
            "Scheduler started with {} jobs (staggered pipeline)",
            job_count,
        )

    def stop_scheduler(self) -> None:
        """Gracefully stop the scheduler and all jobs."""
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            self._is_running = False
            logger.info("Scheduler stopped")
        else:
            logger.debug("Scheduler not running, nothing to stop")

    def get_job_status(self) -> List[Dict[str, Any]]:
        """Return status of all scheduled jobs."""
        if not self._scheduler:
            return []

        jobs = []
        for job in self._scheduler.get_jobs():
            next_run = None
            if job.next_run_time:
                next_run = job.next_run_time.isoformat()

            jobs.append(
                {
                    "job_id": job.id,
                    "name": job.name or job.id,
                    "next_run_time": next_run,
                    "trigger": str(job.trigger),
                }
            )

        return jobs

    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get overall scheduler status."""
        return {
            "status": "running" if self._is_running else "stopped",
            "job_count": len(self._scheduler.get_jobs()) if self._scheduler else 0,
            "jobs": self.get_job_status(),
        }

    def pause_job(self, job_id: str) -> bool:
        """Pause a specific job."""
        if not self._scheduler:
            return False
        job = self._scheduler.get_job(job_id)
        if job:
            job.pause()
            logger.info("Job '{}' paused", job_id)
            return True
        logger.warning("Job '{}' not found", job_id)
        return False

    def resume_job(self, job_id: str) -> bool:
        """Resume a paused job."""
        if not self._scheduler:
            return False
        job = self._scheduler.get_job(job_id)
        if job:
            job.resume()
            logger.info("Job '{}' resumed", job_id)
            return True
        logger.warning("Job '{}' not found", job_id)
        return False

    def run_job_now(self, job_id: str) -> bool:
        """Trigger a job to run immediately."""
        if not self._scheduler:
            return False
        job = self._scheduler.get_job(job_id)
        if job:
            job.modify(next_run_time=datetime.now())
            logger.info("Job '{}' triggered for immediate execution", job_id)
            return True
        logger.warning("Job '{}' not found", job_id)
        return False


# Module-level singleton
scheduler_manager = SchedulerManager()

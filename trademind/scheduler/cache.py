"""Shared in-memory cache for scheduler data pipeline.

Eliminates duplicate work between tasks by storing intermediate results:
- option_chains: per-symbol option chain data from NSE
- market_breadth: advance/decline ratio from NSE
- institutional_flow: FII/DII flow from NSE
- vix_data: India VIX from NSE
- scoring_results: AI scores from ai_scoring_task (consumed by recommendation_task)
- feature_cache: features from feature_engineering_task (consumed by scoring)
- price_cache: current prices (consumed by scoring)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class DataCache:
    """Module-level singleton that holds intermediate results between tasks."""

    def __init__(self) -> None:
        self.option_chains: Dict[str, Any] = {}
        self.market_breadth: Optional[Any] = None
        self.institutional_flow: Optional[Any] = None
        self.vix_data: Optional[Any] = None
        self.scoring_results: Dict[str, float] = {}
        self.feature_cache: Dict[str, Dict[str, float]] = {}
        self.price_cache: Dict[str, Any] = {}
        self.last_update: Dict[str, datetime] = {}

    def update_timestamp(self, key: str) -> None:
        self.last_update[key] = datetime.now()

    def get_staleness_seconds(self, key: str) -> Optional[float]:
        ts = self.last_update.get(key)
        if ts is None:
            return None
        return (datetime.now() - ts).total_seconds()


# Module-level singleton — import and use directly
data_cache = DataCache()

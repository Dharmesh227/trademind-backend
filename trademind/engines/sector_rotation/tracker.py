"""Sector momentum tracker — store daily sector index returns, compute rolling momentum."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SectorMomentumTracker:
    """Tracks daily returns for NSE sector indices and computes momentum scores."""

    def __init__(self) -> None:
        self._returns: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self._last_update: Optional[datetime] = None

    def record_daily_returns(
        self, date_str: str, index_returns: Dict[str, float]
    ) -> None:
        """Record daily % change for each sector index."""
        for sector, ret in index_returns.items():
            self._returns[sector].append((date_str, ret))

        for sector in self._returns:
            self._returns[sector].sort(key=lambda x: x[0])

        self._last_update = datetime.now()

    def get_momentum_score(
        self, sector_name: str, lookback_days: int = 20
    ) -> float:
        """Weighted sum of recent returns → 0-100 normalized score.

        More recent returns get higher weight.
        """
        history = self._returns.get(sector_name, [])
        if not history:
            return 50.0

        recent = history[-lookback_days:]
        if not recent:
            return 50.0

        weighted_sum = 0.0
        weight_total = 0.0
        for i, (_, ret) in enumerate(recent):
            weight = i + 1
            weighted_sum += ret * weight
            weight_total += weight

        if weight_total == 0:
            return 50.0

        avg_weighted = weighted_sum / weight_total
        normalized = 50 + avg_weighted * 10
        return max(0, min(100, round(normalized, 1)))

    def get_momentum_ranks(self, lookback_days: int = 20) -> Dict[str, int]:
        """Rank all sectors by momentum score. 1 = highest momentum."""
        scores = {}
        for sector in self._returns:
            scores[sector] = self.get_momentum_score(sector, lookback_days)

        sorted_sectors = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return {sector: rank + 1 for rank, (sector, _) in enumerate(sorted_sectors)}

    def get_sector_returns_history(
        self, sector_name: str, days: int = 20
    ) -> List[float]:
        """Raw daily returns for a sector."""
        history = self._returns.get(sector_name, [])
        return [ret for _, ret in history[-days:]]

    def get_all_sectors(self) -> List[str]:
        return list(self._returns.keys())

    def get_cumulative_return(
        self, sector_name: str, days: int = 20
    ) -> float:
        """Compound cumulative return over N days."""
        returns = self.get_sector_returns_history(sector_name, days)
        if not returns:
            return 0.0

        cumulative = 1.0
        for r in returns:
            cumulative *= (1 + r / 100)
        return round((cumulative - 1) * 100, 2)

    @property
    def last_update(self) -> Optional[datetime]:
        return self._last_update

    @property
    def data_points(self) -> int:
        return max((len(v) for v in self._returns.values()), default=0)


sector_tracker = SectorMomentumTracker()

"""Rotation detector — identify leading/lagging sectors and rotation patterns."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from trademind.engines.sector_rotation.tracker import SectorMomentumTracker

logger = logging.getLogger(__name__)


@dataclass
class RotationRegime:
    leading_sectors: List[str] = field(default_factory=list)
    lagging_sectors: List[str] = field(default_factory=list)
    rotation_phase: str = "unknown"
    breadth_percentile: float = 50.0
    rotation_strength: float = 0.0
    momentum_scores: Dict[str, float] = field(default_factory=dict)
    timestamp: str = ""


@dataclass
class RotationChange:
    old_leaders: List[str] = field(default_factory=list)
    new_leaders: List[str] = field(default_factory=list)
    direction: str = "mixed"
    confidence: float = 0.0
    description: str = ""
    timestamp: str = ""


CYCLICAL_SECTORS = {"NIFTY BANK", "NIFTY METAL", "NIFTY AUTO", "NIFTY REALTY"}
DEFENSIVE_SECTORS = {"NIFTY FMCG", "NIFTY PHARMA", "NIFTY IT"}


class RotationDetector:
    """Detects sector rotation phases and signals."""

    LEADING_PERCENTILE = 80
    LAGGING_PERCENTILE = 20

    def detect_current_regime(
        self, tracker: SectorMomentumTracker
    ) -> RotationRegime:
        """Analyze current sector momentum to determine rotation phase."""
        from datetime import datetime

        scores = {}
        for sector in tracker.get_all_sectors():
            scores[sector] = tracker.get_momentum_score(sector)

        if not scores:
            return RotationRegime(timestamp=datetime.now().isoformat())

        sorted_sectors = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        n = len(sorted_sectors)

        leading_cutoff = max(1, n * 20 // 100)
        lagging_cutoff = max(1, n * 20 // 100)

        leading = [s for s, _ in sorted_sectors[:leading_cutoff]]
        lagging = [s for s, _ in sorted_sectors[-lagging_cutoff:]]

        positive_count = sum(1 for s in scores.values() if s > 50)
        breadth = positive_count / n * 100 if n > 0 else 50

        all_scores = list(scores.values())
        score_range = max(all_scores) - min(all_scores) if all_scores else 0
        rotation_strength = min(100, score_range)

        phase = self._determine_phase(leading, lagging, breadth, rotation_strength)

        return RotationRegime(
            leading_sectors=leading,
            lagging_sectors=lagging,
            rotation_phase=phase,
            breadth_percentile=round(breadth, 1),
            rotation_strength=round(rotation_strength, 1),
            momentum_scores=scores,
            timestamp=datetime.now().isoformat(),
        )

    def detect_rotation_change(
        self, tracker: SectorMomentumTracker, lookback_days: int = 20
    ) -> Optional[RotationChange]:
        """Detect if leading sectors have shifted recently."""
        from datetime import datetime

        current = self.detect_current_regime(tracker)

        old_scores = {}
        for sector in tracker.get_all_sectors():
            history = tracker.get_sector_returns_history(sector, lookback_days * 2)
            recent = history[-lookback_days:] if len(history) > lookback_days else history
            older = history[:lookback_days] if len(history) > lookback_days else []

            if older:
                old_avg = sum(older) / len(older)
                old_scores[sector] = 50 + old_avg * 10

        if not old_scores:
            return None

        sorted_old = sorted(old_scores.items(), key=lambda x: x[1], reverse=True)
        n = max(1, len(sorted_old) * 20 // 100)
        old_leaders = [s for s, _ in sorted_old[:n]]

        new_leaders = current.leading_sectors
        old_set = set(old_leaders)
        new_set = set(new_leaders)

        if old_set == new_set:
            return None

        cyclical_new = new_set & CYCLICAL_SECTORS
        defensive_new = new_set & DEFENSIVE_SECTORS
        cyclical_old = old_set & CYCLICAL_SECTORS
        defensive_old = old_set & DEFENSIVE_SECTORS

        if len(cyclical_new) > len(cyclical_old):
            direction = "cyclical"
        elif len(defensive_new) > len(defensive_old):
            direction = "defensive"
        else:
            direction = "mixed"

        overlap = len(old_set & new_set)
        confidence = max(0, min(100, (1 - overlap / max(len(old_set), 1)) * 100))

        return RotationChange(
            old_leaders=old_leaders,
            new_leaders=new_leaders,
            direction=direction,
            confidence=round(confidence, 1),
            description=f"Rotation shifted to {direction}: {', '.join(new_leaders)}",
            timestamp=datetime.now().isoformat(),
        )

    def compute_rotation_signal(
        self, sector_name: str, regime: RotationRegime
    ) -> float:
        """0-100 signal for a sector based on rotation position."""
        score = regime.momentum_scores.get(sector_name, 50.0)

        if sector_name in regime.leading_sectors:
            phase_bonus = {
                "early_cycle": 15,
                "mid_cycle": 10,
                "late_cycle": 0,
                "defensive": 5,
            }.get(regime.rotation_phase, 0)
            return min(100, score + phase_bonus)

        if sector_name in regime.lagging_sectors:
            phase_penalty = {
                "early_cycle": 5,
                "mid_cycle": 10,
                "late_cycle": 15,
                "defensive": 5,
            }.get(regime.rotation_phase, 0)
            return max(0, score - phase_penalty)

        return score

    def _determine_phase(
        self,
        leading: List[str],
        lagging: List[str],
        breadth: float,
        strength: float,
    ) -> str:
        """Determine the market rotation phase."""
        leading_set = set(leading)
        lagging_set = set(lagging)

        cyclical_leading = len(leading_set & CYCLICAL_SECTORS)
        defensive_leading = len(leading_set & DEFENSIVE_SECTORS)

        if breadth > 70 and cyclical_leading > defensive_leading:
            return "early_cycle"
        elif breadth > 50 and strength < 40:
            return "mid_cycle"
        elif breadth < 40 and defensive_leading > cyclical_leading:
            return "defensive"
        elif breadth < 50:
            return "late_cycle"
        else:
            return "mid_cycle"


rotation_detector = RotationDetector()

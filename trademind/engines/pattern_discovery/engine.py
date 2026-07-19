from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from trademind.database.models import (
    LearningRecord,
    Pattern,
    PaperTrade,
)
from trademind.config.settings import settings


@dataclass
class DiscoveredPattern:
    name: str
    description: str
    conditions: Dict[str, Any]
    category: str
    trade_count: int = 0
    win_count: int = 0
    success_rate: float = 0.0
    avg_return: float = 0.0
    confidence: float = 0.0


@dataclass
class PatternMatch:
    pattern: DiscoveredPattern
    confidence: float
    matched_conditions: List[str]


@dataclass
class PatternStats:
    total_patterns: int = 0
    avg_accuracy: float = 0.0
    best_pattern: str = ""
    best_pattern_accuracy: float = 0.0
    active_patterns: int = 0
    avg_trades_per_pattern: float = 0.0


class PatternDiscoveryEngine:
    """Engine 9: Discovers and validates trading patterns from trade data."""

    BUCKET_RANGES: Dict[str, List[Tuple[float, float]]] = {
        "rs_rank": [(0, 30), (30, 50), (50, 70), (70, 85), (85, 100)],
        "volume_ratio": [(0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 100)],
        "delivery_pct": [(0, 40), (40, 55), (55, 70), (70, 85), (85, 100)],
        "vwap_distance_pct": [(-100, -2), (-2, -0.5), (-0.5, 0.5), (0.5, 2), (2, 100)],
        "sector_rank": [(0, 3), (3, 6), (6, 10), (10, 20), (20, 100)],
        "breakout_strength": [(0, 40), (40, 60), (60, 75), (75, 90), (90, 100)],
        "oi_change_pct": [(-100, -10), (-10, 0), (0, 10), (10, 30), (30, 100)],
        "momentum_score": [(0, 30), (30, 50), (50, 65), (65, 80), (80, 100)],
        "trend_score": [(0, 30), (30, 50), (50, 65), (65, 80), (80, 100)],
        "adx_value": [(0, 15), (15, 25), (25, 35), (35, 50), (50, 100)],
    }

    THRESHOLD_FEATURES: List[Tuple[str, float, str]] = [
        ("rs_rank", 85, "high"),
        ("volume_ratio", 2.5, "high"),
        ("delivery_pct", 65, "high"),
        ("breakout_strength", 80, "high"),
        ("momentum_score", 70, "high"),
        ("trend_score", 70, "high"),
        ("oi_change_pct", 15, "high"),
        ("adx_value", 25, "high"),
    ]

    _DB_URL = "sqlite+aiosqlite:///trademind.db"

    def __init__(self) -> None:
        self._engine = create_async_engine(self._DB_URL, echo=False)
        self._factory = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )
        self._patterns_cache: Optional[List[DiscoveredPattern]] = None

    async def discover_patterns(
        self,
        trades_with_features: Optional[List[Tuple[PaperTrade, Dict[str, Any]]]] = None,
    ) -> List[DiscoveredPattern]:
        if trades_with_features is None:
            trades_with_features = await self._load_trades_with_features()

        if len(trades_with_features) < settings.pattern_min_samples:
            logger.info(
                "Not enough trades: {} < {}",
                len(trades_with_features),
                settings.pattern_min_samples,
            )
            return []

        df = self._build_feature_matrix(trades_with_features)
        if df.empty:
            return []

        discovered: List[DiscoveredPattern] = []

        freq_combos = self._find_frequent_combinations(
            trades_with_features, min_support=0.1
        )
        for _key, stats in freq_combos:
            pattern = self._combo_to_pattern(stats)
            if pattern:
                discovered.append(pattern)

        threshold_patterns = self._discover_threshold_patterns(df)
        discovered.extend(threshold_patterns)

        composite_patterns = self._discover_composite_patterns(df)
        existing_names = {p.name for p in discovered}
        for p in composite_patterns:
            if p.name not in existing_names:
                discovered.append(p)
                existing_names.add(p.name)

        for p in discovered:
            sr, avg_ret = self._evaluate_pattern(p, trades_with_features)
            p.success_rate = sr
            p.avg_return = avg_ret
            if p.trade_count == 0:
                p.trade_count = max(3, int(sr * len(trades_with_features) * 0.1))
            p.confidence = min(
                1.0, sr * min(1.0, p.trade_count / max(1, len(trades_with_features)))
            )

        discovered = [p for p in discovered if p.trade_count >= 3]
        discovered.sort(key=lambda x: x.success_rate * x.confidence, reverse=True)

        await self._persist_patterns(discovered)
        self._patterns_cache = discovered

        logger.info(
            "Discovered {} patterns from {} trades",
            len(discovered),
            len(trades_with_features),
        )
        return discovered

    async def match_pattern(
        self,
        current_features: Dict[str, Any],
        patterns: Optional[List[DiscoveredPattern]] = None,
    ) -> List[PatternMatch]:
        if patterns is None:
            patterns = await self.get_top_patterns()

        matches: List[PatternMatch] = []

        for pattern in patterns:
            matched_conditions: List[str] = []
            total_conditions = len(pattern.conditions)
            matched_count = 0

            for fname, condition in pattern.conditions.items():
                current_val = current_features.get(fname)
                if current_val is None:
                    continue
                try:
                    current_val = float(current_val)
                except (TypeError, ValueError):
                    continue

                if self._check_condition(current_val, condition):
                    matched_count += 1
                    matched_conditions.append(
                        f"{fname}={current_val:.1f} matches {condition}"
                    )

            if total_conditions > 0 and matched_count > 0:
                mc = (matched_count / total_conditions) * pattern.confidence
                if mc > 0.2:
                    matches.append(
                        PatternMatch(
                            pattern=pattern,
                            confidence=round(mc, 4),
                            matched_conditions=matched_conditions,
                        )
                    )

        matches.sort(key=lambda x: x.confidence, reverse=True)
        return matches

    def _find_frequent_combinations(
        self,
        trades_with_features: List[Tuple[PaperTrade, Dict[str, Any]]],
        min_support: float = 0.1,
    ) -> List[Tuple[Tuple[str, ...], Dict[str, Any]]]:
        min_count = max(1, int(len(trades_with_features) * min_support))

        feature_buckets: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for _, features in trades_with_features:
            for fname, fval in features.items():
                if fval is None or fname not in self.BUCKET_RANGES:
                    continue
                try:
                    fval_f = float(fval)
                except (TypeError, ValueError):
                    continue
                bucket = self._bucketize(fname, fval_f)
                if bucket is not None:
                    feature_buckets[fname][bucket] += 1

        frequent_singles: List[Tuple[str, str]] = []
        for fname, buckets in feature_buckets.items():
            for bucket, count in buckets.items():
                if count >= min_count:
                    frequent_singles.append((fname, bucket))

        results: List[Tuple[Tuple[str, ...], Dict[str, Any]]] = []

        for i in range(len(frequent_singles)):
            for j in range(i + 1, len(frequent_singles)):
                f1, b1 = frequent_singles[i]
                f2, b2 = frequent_singles[j]
                if f1 == f2:
                    continue

                pair_count = 0
                for _, features in trades_with_features:
                    v1 = features.get(f1)
                    v2 = features.get(f2)
                    if v1 is None or v2 is None:
                        continue
                    try:
                        v1f, v2f = float(v1), float(v2)
                    except (TypeError, ValueError):
                        continue
                    if self._bucketize(f1, v1f) == b1 and self._bucketize(f2, v2f) == b2:
                        pair_count += 1

                if pair_count >= min_count:
                    conds = {f1: b1, f2: b2}
                    win_t, combo_t, pnl_t = self._evaluate_conditions(
                        conds, trades_with_features
                    )
                    if combo_t >= 3:
                        results.append(
                            (
                                (f1, f2),
                                {
                                    "conditions": conds,
                                    "support": pair_count / max(1, len(trades_with_features)),
                                    "success_rate": win_t / combo_t,
                                    "avg_return": pnl_t / combo_t,
                                    "trade_count": combo_t,
                                },
                            )
                        )

        for i in range(len(frequent_singles)):
            for j in range(i + 1, len(frequent_singles)):
                for k in range(j + 1, len(frequent_singles)):
                    f1, b1 = frequent_singles[i]
                    f2, b2 = frequent_singles[j]
                    f3, b3 = frequent_singles[k]
                    if len({f1, f2, f3}) < 3:
                        continue

                    combo_count = 0
                    for _, features in trades_with_features:
                        v1 = features.get(f1)
                        v2 = features.get(f2)
                        v3 = features.get(f3)
                        if v1 is None or v2 is None or v3 is None:
                            continue
                        try:
                            v1f = float(v1)
                            v2f = float(v2)
                            v3f = float(v3)
                        except (TypeError, ValueError):
                            continue
                        if (
                            self._bucketize(f1, v1f) == b1
                            and self._bucketize(f2, v2f) == b2
                            and self._bucketize(f3, v3f) == b3
                        ):
                            combo_count += 1

                    if combo_count >= min_count:
                        conds = {f1: b1, f2: b2, f3: b3}
                        win_t, combo_t, pnl_t = self._evaluate_conditions(
                            conds, trades_with_features
                        )
                        if combo_t >= 3:
                            results.append(
                                (
                                    (f1, f2, f3),
                                    {
                                        "conditions": conds,
                                        "support": combo_count / max(1, len(trades_with_features)),
                                        "success_rate": win_t / combo_t,
                                        "avg_return": pnl_t / combo_t,
                                        "trade_count": combo_t,
                                    },
                                )
                            )

        results.sort(
            key=lambda x: x[1]["success_rate"] * x[1]["support"], reverse=True
        )
        return results[:50]

    def _evaluate_conditions(
        self,
        conditions: Dict[str, str],
        trades_with_features: List[Tuple[PaperTrade, Dict[str, Any]]],
    ) -> Tuple[int, int, float]:
        win_count = 0
        total = 0
        total_pnl = 0.0
        for trade, features in trades_with_features:
            if self._match_bucket_conditions(conditions, features):
                total += 1
                if (trade.pnl or 0) > 0:
                    win_count += 1
                total_pnl += trade.pnl_percent or 0.0
        return win_count, total, total_pnl

    def _evaluate_pattern(
        self,
        pattern: DiscoveredPattern,
        trades_with_features: List[Tuple[PaperTrade, Dict[str, Any]]],
    ) -> Tuple[float, float]:
        matching_trades = []
        for trade, features in trades_with_features:
            if self._match_conditions(pattern.conditions, features):
                matching_trades.append(trade)

        if not matching_trades:
            return 0.0, 0.0

        wins = sum(1 for t in matching_trades if (t.pnl or 0) > 0)
        total = len(matching_trades)
        sr = wins / total if total > 0 else 0.0

        pnls = [t.pnl_percent or 0.0 for t in matching_trades]
        avg_ret = float(np.mean(pnls)) if pnls else 0.0
        return round(sr, 4), round(avg_ret, 2)

    def _discover_threshold_patterns(
        self, df: pd.DataFrame
    ) -> List[DiscoveredPattern]:
        patterns: List[DiscoveredPattern] = []

        for fname, threshold, direction in self.THRESHOLD_FEATURES:
            if fname not in df.columns:
                continue

            if direction == "high":
                mask = df[fname] >= threshold
                label = f"{fname} >= {threshold}"
            else:
                mask = df[fname] <= threshold
                label = f"{fname} <= {threshold}"

            matched = df[mask]
            if len(matched) < 3:
                continue

            win_count = int((matched["pnl"] > 0).sum())
            total = len(matched)
            sr = win_count / total
            avg_ret = float(matched["pnl_percent"].mean())

            cat = self._feature_to_category(fname)
            conditions = {fname: {"min": threshold} if direction == "high" else {"max": threshold}}

            patterns.append(
                DiscoveredPattern(
                    name=label,
                    description=f"{label} -> {sr:.0%} success ({total} trades)",
                    conditions=conditions,
                    category=cat,
                    trade_count=total,
                    win_count=win_count,
                    success_rate=round(sr, 4),
                    avg_return=round(avg_ret, 2),
                    confidence=round(min(1.0, sr * (total / max(1, len(df)))), 4),
                )
            )

        return patterns

    def _discover_composite_patterns(
        self, df: pd.DataFrame
    ) -> List[DiscoveredPattern]:
        patterns: List[DiscoveredPattern] = []

        multi_features = [
            ["rs_rank", "volume_ratio", "sector_rank"],
            ["breakout_strength", "delivery_pct", "volume_ratio"],
            ["vwap_distance_pct", "breakout_strength", "delivery_pct"],
            ["momentum_score", "volume_ratio", "sector_rank"],
        ]

        thresholds_map = {
            "rs_rank": ("min", 80),
            "volume_ratio": ("min", 2.0),
            "delivery_pct": ("min", 60),
            "vwap_distance_pct": ("min", 1.0),
            "breakout_strength": ("min", 75),
            "sector_rank": ("max", 3),
            "momentum_score": ("min", 65),
        }

        for feature_combo in multi_features:
            available = [f for f in feature_combo if f in df.columns]
            if len(available) < 2:
                continue

            conditions: Dict[str, Any] = {}
            labels: List[str] = []
            mask = pd.Series([True] * len(df))

            for fname in available:
                op, val = thresholds_map.get(fname, ("min", 50))
                conditions[fname] = {op: val}
                if op == "min":
                    mask = mask & (df[fname] >= val)
                    labels.append(f"{fname} >= {val}")
                else:
                    mask = mask & (df[fname] <= val)
                    labels.append(f"{fname} <= {val}")

            matched = df[mask]
            if len(matched) < 3:
                continue

            win_count = int((matched["pnl"] > 0).sum())
            total = len(matched)
            sr = win_count / total
            avg_ret = float(matched["pnl_percent"].mean())

            name = " AND ".join(labels)
            patterns.append(
                DiscoveredPattern(
                    name=name,
                    description=f"Composite: {name} -> {sr:.0%} success",
                    conditions=conditions,
                    category="composite",
                    trade_count=total,
                    win_count=win_count,
                    success_rate=round(sr, 4),
                    avg_return=round(avg_ret, 2),
                    confidence=round(min(1.0, sr * (total / max(1, len(df)))), 4),
                )
            )

        return patterns

    async def get_top_patterns(self, n: int = 20) -> List[DiscoveredPattern]:
        if self._patterns_cache is not None:
            return self._patterns_cache[:n]

        async with self._factory() as session:
            result = await session.execute(
                select(Pattern)
                .where(Pattern.is_active == True)
                .order_by(Pattern.success_rate.desc())
                .limit(n)
            )
            db_patterns = list(result.scalars().all())

        if not db_patterns:
            return []

        patterns = []
        for p in db_patterns:
            try:
                conditions = json.loads(p.conditions_json)
            except (json.JSONDecodeError, TypeError):
                continue
            patterns.append(
                DiscoveredPattern(
                    name=p.pattern_name,
                    description=f"Pattern: {p.pattern_name}",
                    conditions=conditions,
                    category=p.sector or "unknown",
                    trade_count=p.occurrences,
                    win_count=p.wins,
                    success_rate=p.success_rate,
                    avg_return=p.avg_return_percent,
                    confidence=p.success_rate,
                )
            )

        self._patterns_cache = patterns
        return patterns[:n]

    async def get_pattern_stats(self) -> PatternStats:
        async with self._factory() as session:
            result = await session.execute(
                select(Pattern).where(Pattern.is_active == True)
            )
            all_patterns = list(result.scalars().all())

        if not all_patterns:
            return PatternStats()

        accuracies = [p.success_rate for p in all_patterns]
        occurrences = [p.occurrences for p in all_patterns]
        best = max(all_patterns, key=lambda p: p.success_rate)

        return PatternStats(
            total_patterns=len(all_patterns),
            avg_accuracy=round(float(np.mean(accuracies)), 4),
            best_pattern=best.pattern_name,
            best_pattern_accuracy=round(best.success_rate, 4),
            active_patterns=sum(1 for p in all_patterns if p.is_active),
            avg_trades_per_pattern=round(float(np.mean(occurrences)), 1),
        )

    async def _load_trades_with_features(
        self,
    ) -> List[Tuple[PaperTrade, Dict[str, Any]]]:
        async with self._factory() as session:
            trades_result = await session.execute(
                select(PaperTrade).where(
                    PaperTrade.status.in_(["closed", "completed"])
                )
            )
            trades = list(trades_result.scalars().all())

            if not trades:
                return []

            trade_ids = [t.id for t in trades]
            lr_result = await session.execute(
                select(LearningRecord).where(
                    LearningRecord.trade_id.in_(trade_ids)
                )
            )
            records = list(lr_result.scalars().all())

        record_map: Dict[str, Dict[str, Any]] = {}
        for lr in records:
            if lr.trade_id:
                try:
                    record_map[lr.trade_id] = json.loads(lr.features_snapshot_json)
                except (json.JSONDecodeError, TypeError):
                    continue

        result: List[Tuple[PaperTrade, Dict[str, Any]]] = []
        for trade in trades:
            features = record_map.get(trade.id, {})
            if features:
                result.append((trade, features))

        return result

    def _build_feature_matrix(
        self, trades_with_features: List[Tuple[PaperTrade, Dict[str, Any]]]
    ) -> pd.DataFrame:
        records = []
        for trade, features in trades_with_features:
            row: Dict[str, Any] = {
                "pnl": trade.pnl or 0.0,
                "pnl_percent": trade.pnl_percent or 0.0,
            }
            for k, v in features.items():
                if isinstance(v, (int, float)):
                    row[k] = v
            records.append(row)
        return pd.DataFrame(records) if records else pd.DataFrame()

    def _bucketize(self, feature_name: str, value: float) -> Optional[str]:
        ranges = self.BUCKET_RANGES.get(feature_name)
        if not ranges:
            return None
        for i, (low, high) in enumerate(ranges):
            if low <= value < high or (i == len(ranges) - 1 and value >= low):
                return f"bucket_{i}"
        return None

    def _match_bucket_conditions(
        self,
        conditions: Dict[str, str],
        features: Dict[str, Any],
    ) -> bool:
        for fname, expected_bucket in conditions.items():
            fval = features.get(fname)
            if fval is None:
                return False
            try:
                fval_f = float(fval)
            except (TypeError, ValueError):
                return False
            if self._bucketize(fname, fval_f) != expected_bucket:
                return False
        return True

    def _match_conditions(
        self, conditions: Dict[str, Any], features: Dict[str, Any]
    ) -> bool:
        for fname, condition in conditions.items():
            fval = features.get(fname)
            if fval is None:
                return False
            try:
                fval_f = float(fval)
            except (TypeError, ValueError):
                return False
            if not self._check_condition(fval_f, condition):
                return False
        return True

    @staticmethod
    def _check_condition(value: float, condition: Any) -> bool:
        if isinstance(condition, dict):
            if "min" in condition and value < condition["min"]:
                return False
            if "max" in condition and value > condition["max"]:
                return False
            return True
        if isinstance(condition, (int, float)):
            return value >= float(condition)
        return True

    def _combo_to_pattern(self, stats: Dict[str, Any]) -> Optional[DiscoveredPattern]:
        conditions = stats.get("conditions", {})
        if not conditions:
            return None

        feature_names = set(conditions.keys())
        cat = self._conditions_to_category(feature_names)

        parts = []
        for fname in sorted(conditions.keys()):
            parts.append(f"{fname}={conditions[fname]}")
        name = " AND ".join(parts)

        return DiscoveredPattern(
            name=name,
            description=f"Frequent combination: {name}",
            conditions=conditions,
            category=cat,
            trade_count=stats.get("trade_count", 0),
            success_rate=stats.get("success_rate", 0.0),
            avg_return=stats.get("avg_return", 0.0),
            confidence=stats.get("support", 0.0),
        )

    def _conditions_to_category(self, feature_names: set) -> str:
        if feature_names & {"sector_rank", "sector_momentum"}:
            return "sector"
        if feature_names & {"oi_change_pct", "put_call_ratio", "pcr_oi"}:
            return "options"
        if feature_names & {"volume_ratio", "delivery_pct", "vwap_distance_pct"}:
            return "volume"
        if feature_names & {"rs_rank", "momentum_score", "macd_signal"}:
            return "momentum"
        if feature_names & {"trend_score", "adx_value", "breakout_strength"}:
            return "trend"
        if feature_names & {"market_breadth", "fii_flow", "dii_flow"}:
            return "market"
        if feature_names & {"vix_level", "atr_pct", "bb_position"}:
            return "volatility"
        return "composite"

    async def _persist_patterns(self, patterns: List[DiscoveredPattern]) -> None:
        async with self._factory() as session:
            existing_result = await session.execute(
                select(Pattern).where(Pattern.is_active == True)
            )
            existing = {p.pattern_name: p for p in existing_result.scalars().all()}

            for p in patterns:
                if p.name in existing:
                    db_pat = existing[p.name]
                    db_pat.occurrences = p.trade_count
                    db_pat.wins = p.win_count
                    db_pat.losses = p.trade_count - p.win_count
                    db_pat.success_rate = p.success_rate
                    db_pat.avg_return_percent = p.avg_return
                    db_pat.last_seen = datetime.utcnow()
                else:
                    session.add(
                        Pattern(
                            pattern_name=p.name,
                            conditions_json=json.dumps(p.conditions),
                            sector=p.category if p.category != "composite" else None,
                            occurrences=p.trade_count,
                            wins=p.win_count,
                            losses=p.trade_count - p.win_count,
                            success_rate=p.success_rate,
                            avg_return_percent=p.avg_return,
                            discovered_date=datetime.utcnow(),
                            last_seen=datetime.utcnow(),
                            is_active=True,
                        )
                    )

            await session.commit()

    @staticmethod
    def _feature_to_category(feature_name: str) -> str:
        category_map = {
            "rs_rank": "momentum",
            "momentum_score": "momentum",
            "macd_signal": "momentum",
            "rsi_value": "momentum",
            "volume_ratio": "volume",
            "delivery_pct": "volume",
            "vwap_distance_pct": "volume",
            "trend_score": "trend",
            "adx_value": "trend",
            "breakout_strength": "trend",
            "sector_rank": "sector",
            "sector_momentum": "sector",
            "oi_change_pct": "options",
            "put_call_ratio": "options",
            "market_breadth": "market",
            "fii_flow": "market",
            "dii_flow": "market",
            "vix_level": "volatility",
            "atr_pct": "volatility",
            "bb_position": "volatility",
        }
        return category_map.get(feature_name, "other")

    async def close(self) -> None:
        await self._engine.dispose()

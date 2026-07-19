from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from trademind.database.models import (
    FeatureVector,
    KnowledgeBase,
    LearningRecord,
    PaperTrade,
)
from trademind.config.settings import settings


@dataclass
class PatternInsight:
    feature_name: str
    direction: str
    condition: str
    value: float
    impact_pct: float
    description: str


@dataclass
class LearningResult:
    record: LearningRecord
    insights: List[PatternInsight]
    similarity_score: float
    is_novel: bool


@dataclass
class FeatureImportanceEntry:
    feature: str
    importance: float
    category: str
    correlation_with_profit: float
    description: str


@dataclass
class LearningStats:
    total_trades_learned: int = 0
    patterns_found: int = 0
    model_version: str = "v1"
    unique_features_tracked: int = 0
    avg_feature_coverage: float = 0.0
    last_learned_at: Optional[datetime] = None
    accuracy_trend: float = 0.0
    confidence_calibration: float = 0.0


class LearningEngine:
    """Engine 7: The brain — learns from every completed trade."""

    _DB_URL = "sqlite+aiosqlite:///trademind.db"
    _MIN_FEATURES = 3

    def __init__(self) -> None:
        self._engine = create_async_engine(self._DB_URL, echo=False)
        self._factory = async_sessionmaker(self._engine, class_=AsyncSession, expire_on_commit=False)

    async def learn_from_trade(
        self, paper_trade: PaperTrade, features_snapshot: Dict[str, Any]
    ) -> LearningResult:
        result_label = "win" if (paper_trade.pnl or 0) > 0 else "loss"
        if paper_trade.pnl is not None and abs(paper_trade.pnl) < 0.01:
            result_label = "breakeven"

        pnl_pct = paper_trade.pnl_percent or 0.0

        insights = self._extract_insight(result_label, features_snapshot)
        insight_texts = [i.description for i in insights]

        predicted_score = self._features_to_score(features_snapshot)
        weight_changes = self._compute_weight_deltas(features_snapshot, result_label, pnl_pct)
        feature_importance = self._quick_feature_importance(features_snapshot, pnl_pct)

        record = LearningRecord(
            trade_id=paper_trade.id,
            features_snapshot_json=json.dumps(features_snapshot, default=str),
            predicted_score=predicted_score,
            actual_outcome=pnl_pct,
            outcome=result_label,
            pnl_percent=pnl_pct,
            weight_changes_json=json.dumps(weight_changes),
            feature_importance_json=json.dumps(feature_importance),
        )

        similar_trades = await self.get_similar_historical_trades(features_snapshot, threshold=0.7)
        similarity_score = 0.0
        if similar_trades:
            best_sim = max(
                self._compute_similarity(features_snapshot, st)
                for st in similar_trades
            )
            similarity_score = best_sim

        is_novel = len(similar_trades) == 0

        async with self._factory() as session:
            session.add(record)
            await session.commit()
            await session.refresh(record)

        await self.update_model_knowledge(record)

        logger.info(
            "Learned trade {} ({}): {} | PnL {:.2f}% | Sim {:.2f} | Novel={}",
            paper_trade.id, paper_trade.symbol, result_label, pnl_pct,
            similarity_score, is_novel,
        )

        return LearningResult(
            record=record,
            insights=insights,
            similarity_score=similarity_score,
            is_novel=is_novel,
        )

    async def analyze_pattern(
        self, trade: PaperTrade, features: Dict[str, Any]
    ) -> List[PatternInsight]:
        result_label = "win" if (trade.pnl or 0) > 0 else "loss"
        return self._extract_insight(result_label, features)

    async def update_model_knowledge(self, learning_record: LearningRecord) -> None:
        async with self._factory() as session:
            count_result = await session.execute(
                select(func.count(LearningRecord.id))
            )
            total_learned = count_result.scalar() or 0

            pattern_result = await session.execute(
                select(func.count(LearningRecord.id)).where(
                    LearningRecord.outcome == "win"
                )
            )
            win_count = pattern_result.scalar() or 0

            kb_result = await session.execute(
                select(KnowledgeBase)
                .where(KnowledgeBase.is_active == True)
                .order_by(KnowledgeBase.id.desc())
                .limit(1)
            )
            kb = kb_result.scalars().first()

            if kb:
                kb.trades_learned = total_learned
                kb.patterns_found = win_count
                kb.accuracy = round(win_count / total_learned * 100, 2) if total_learned > 0 else 0.0
                kb.training_data_end = learning_record.created_at
                changelog = json.loads(kb.changelog_json) if kb.changelog_json else []
                changelog.append({
                    "action": "trade_learned",
                    "trade_id": learning_record.trade_id,
                    "outcome": learning_record.outcome,
                    "pnl_pct": learning_record.pnl_percent,
                    "timestamp": datetime.utcnow().isoformat(),
                })
                kb.changelog_json = json.dumps(changelog[-100:])
            else:
                kb = KnowledgeBase(
                    version="v1",
                    model_name="TradeMind Ensemble",
                    model_type="ensemble",
                    trades_learned=total_learned,
                    patterns_found=win_count,
                    accuracy=round(win_count / total_learned * 100, 2) if total_learned > 0 else 0.0,
                    training_data_start=learning_record.created_at,
                    training_data_end=learning_record.created_at,
                    changelog_json=json.dumps([{
                        "action": "trade_learned",
                        "trade_id": learning_record.trade_id,
                        "outcome": learning_record.outcome,
                        "timestamp": datetime.utcnow().isoformat(),
                    }]),
                )
                session.add(kb)

            await session.commit()

    async def get_similar_historical_trades(
        self, features: Dict[str, Any], threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        async with self._factory() as session:
            result = await session.execute(
                select(LearningRecord).order_by(LearningRecord.id.desc()).limit(500)
            )
            records = list(result.scalars().all())

        similar: List[Dict[str, Any]] = []
        for record in records:
            try:
                record_features = json.loads(record.features_snapshot_json)
            except (json.JSONDecodeError, TypeError):
                continue
            sim = self._compute_similarity(features, record_features)
            if sim >= threshold:
                similar.append(record_features)

        return similar

    async def calculate_feature_importance(
        self, trades: Optional[List[PaperTrade]] = None
    ) -> List[FeatureImportanceEntry]:
        async with self._factory() as session:
            result = await session.execute(
                select(LearningRecord).order_by(LearningRecord.id.desc()).limit(1000)
            )
            records = list(result.scalars().all())

        if len(records) < self._MIN_FEATURES:
            return []

        all_features: Dict[str, List[float]] = {}
        pnl_values: List[float] = []

        for record in records:
            try:
                features = json.loads(record.features_snapshot_json)
            except (json.JSONDecodeError, TypeError):
                continue

            pnl_values.append(record.pnl_percent or 0.0)
            for fname, fval in features.items():
                if fval is not None and isinstance(fval, (int, float)):
                    all_features.setdefault(fname, []).append(float(fval))

        if not pnl_values:
            return []

        pnl_arr = np.array(pnl_values)
        importances: List[FeatureImportanceEntry] = []

        for fname, fvals in all_features.items():
            min_len = min(len(fvals), len(pnl_arr))
            if min_len < 10:
                continue

            feat_arr = np.array(fvals[:min_len])
            pnl_slice = pnl_arr[:min_len]

            if np.std(feat_arr) == 0 or np.std(pnl_slice) == 0:
                corr = 0.0
            else:
                corr = float(np.corrcoef(feat_arr, pnl_slice)[0, 1])
                if np.isnan(corr):
                    corr = 0.0

            importance = abs(corr)
            category = self._feature_to_category(fname)

            if corr > 0.1:
                desc = f"{fname} positively correlates with profit (r={corr:.3f})"
            elif corr < -0.1:
                desc = f"{fname} negatively correlates with profit (r={corr:.3f})"
            else:
                desc = f"{fname} shows no significant correlation with profit"

            importances.append(
                FeatureImportanceEntry(
                    feature=fname,
                    importance=round(importance, 4),
                    category=category,
                    correlation_with_profit=round(corr, 4),
                    description=desc,
                )
            )

        importances.sort(key=lambda x: x.importance, reverse=True)
        return importances

    async def should_update_weights(self) -> bool:
        async with self._factory() as session:
            result = await session.execute(
                select(func.count(LearningRecord.id))
            )
            count = result.scalar() or 0
        return count >= settings.min_trades_for_learning

    async def get_learning_stats(self) -> LearningStats:
        async with self._factory() as session:
            trade_count_result = await session.execute(
                select(func.count(LearningRecord.id))
            )
            total_learned = trade_count_result.scalar() or 0

            win_result = await session.execute(
                select(func.count(LearningRecord.id)).where(
                    LearningRecord.outcome == "win"
                )
            )
            win_count = win_result.scalar() or 0

            kb_result = await session.execute(
                select(KnowledgeBase)
                .where(KnowledgeBase.is_active == True)
                .order_by(KnowledgeBase.id.desc())
                .limit(1)
            )
            kb = kb_result.scalars().first()
            model_version = "v1"
            last_learned = None
            if kb:
                model_version = kb.version
                last_learned = kb.training_data_end

            recent_result = await session.execute(
                select(LearningRecord).order_by(LearningRecord.id.desc()).limit(50)
            )
            recent = list(recent_result.scalars().all())
            accuracy_trend = 0.0
            if recent:
                recent_wins = sum(1 for r in recent if r.outcome == "win")
                accuracy_trend = float(recent_wins / len(recent) * 100)

        return LearningStats(
            total_trades_learned=total_learned,
            patterns_found=win_count,
            model_version=model_version,
            unique_features_tracked=0,
            avg_feature_coverage=0.0,
            last_learned_at=last_learned,
            accuracy_trend=round(accuracy_trend, 1),
        )

    def _compute_similarity(
        self, features_a: Dict[str, Any], features_b: Dict[str, Any]
    ) -> float:
        common_keys = set(features_a.keys()) & set(features_b.keys())
        if not common_keys:
            return 0.0

        valid_keys = [
            k for k in common_keys
            if isinstance(features_a[k], (int, float))
            and isinstance(features_b[k], (int, float))
            and features_a[k] is not None
            and features_b[k] is not None
        ]

        if len(valid_keys) < 3:
            return 0.0

        vec_a = np.array([[float(features_a[k]) for k in valid_keys]])
        vec_b = np.array([[float(features_b[k]) for k in valid_keys]])

        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        cosine_sim = float(np.dot(vec_a[0], vec_b[0]) / (norm_a * norm_b))
        return max(0.0, cosine_sim)

    def _extract_insight(
        self, trade_result: str, features: Dict[str, Any]
    ) -> List[PatternInsight]:
        insights: List[PatternInsight] = []

        feature_descriptions = {
            "rs_rank": ("RS Rank", "Relative Strength"),
            "volume_ratio": ("Volume Ratio", "Volume"),
            "vwap_distance_pct": ("VWAP Distance %", "VWAP"),
            "sector_rank": ("Sector Rank", "Sector Strength"),
            "delivery_pct": ("Delivery %", "Delivery Quality"),
            "oi_change_pct": ("OI Change %", "Options Activity"),
            "breakout_strength": ("Breakout Strength", "Breakout"),
            "momentum_score": ("Momentum Score", "Momentum"),
            "trend_score": ("Trend Score", "Trend"),
            "adx_value": ("ADX Value", "Trend Strength"),
            "atr_pct": ("ATR %", "Volatility"),
            "bb_position": ("BB Position", "Bollinger Band"),
            "macd_signal": ("MACD Signal", "MACD"),
            "rsi_value": ("RSI Value", "RSI"),
        }

        thresholds = {
            "rs_rank": (70, 90),
            "volume_ratio": (1.5, 3.0),
            "delivery_pct": (50, 70),
            "breakout_strength": (60, 85),
            "momentum_score": (60, 80),
            "trend_score": (60, 80),
            "oi_change_pct": (10, 30),
            "adx_value": (20, 35),
        }

        for fname, fval in features.items():
            if fval is None or not isinstance(fval, (int, float)):
                continue

            display_name, category = feature_descriptions.get(
                fname, (fname.replace("_", " ").title(), "Other")
            )
            low, high = thresholds.get(fname, (50, 80))

            if fval >= high:
                strength, direction = "strong", "bullish"
            elif fval >= low:
                strength, direction = "moderate", "bullish"
            elif fval <= (100 - high):
                strength, direction = "strong", "bearish"
            elif fval <= (100 - low):
                strength, direction = "moderate", "bearish"
            else:
                continue

            if trade_result == "win":
                impact = float(fval) if direction == "bullish" else float(100 - fval)
            else:
                impact = -float(fval) if direction == "bullish" else -float(100 - fval)

            insights.append(
                PatternInsight(
                    feature_name=display_name,
                    direction=direction,
                    condition=f"{strength} {direction}",
                    value=float(fval),
                    impact_pct=round(impact, 2),
                    description=(
                        f"{display_name}={fval:.1f} ({strength} {direction}) → {trade_result}"
                    ),
                )
            )

        insights.sort(key=lambda x: abs(x.impact_pct), reverse=True)
        return insights[:15]

    @staticmethod
    def _features_to_score(features: Dict[str, Any]) -> float:
        weighted_keys = [
            "rs_rank", "momentum_score", "trend_score", "volume_ratio",
            "delivery_pct", "breakout_strength", "sector_rank",
        ]
        weights = [0.20, 0.18, 0.18, 0.15, 0.12, 0.10, 0.07]
        total = 0.0
        w_sum = 0.0

        for key, w in zip(weighted_keys, weights):
            val = features.get(key)
            if val is not None and isinstance(val, (int, float)):
                total += float(val) * w
                w_sum += w

        return round(total / w_sum, 2) if w_sum > 0 else 50.0

    @staticmethod
    def _compute_weight_deltas(
        features: Dict[str, Any], result: str, pnl_pct: float
    ) -> Dict[str, float]:
        deltas: Dict[str, float] = {}
        category_features = {
            "trend": ["trend_score", "adx_value", "breakout_strength"],
            "momentum": ["rs_rank", "momentum_score"],
            "volume": ["volume_ratio", "delivery_pct", "vwap_distance_pct"],
            "options": ["oi_change_pct", "put_call_ratio"],
            "sector": ["sector_rank"],
            "market": ["market_breadth", "fii_flow"],
            "volatility": ["vix_level", "atr_pct", "bb_position"],
        }

        sign = 1.0 if result == "win" else -1.0
        magnitude = min(abs(pnl_pct) / 5.0, 1.0)

        for category, keys in category_features.items():
            relevant_vals = [float(features[k]) for k in keys if k in features and features[k] is not None]
            if relevant_vals:
                avg_val = np.mean(relevant_vals)
                delta = sign * magnitude * (avg_val / 100.0) * 2.0
                deltas[category] = round(float(delta), 4)

        return deltas

    @staticmethod
    def _quick_feature_importance(
        features: Dict[str, Any], pnl_pct: float
    ) -> Dict[str, float]:
        importance: Dict[str, float] = {}
        for fname, fval in features.items():
            if fval is not None and isinstance(fval, (int, float)):
                importance[fname] = round(abs(float(fval) - 50) / 50.0, 4)
        return importance

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

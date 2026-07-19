import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from trademind.config.settings import settings as cfg
from trademind.database.models import (
    IndexData,
    InstitutionalFlowData,
    MarketBreadthData,
    OptionChainData,
    OptionStrike,
    PriceData,
    VIXData,
)
from trademind.engines.nse_client import NSEClient

logger = logging.getLogger(__name__)


class MarketDataCollector:
    """Collects NSE market data via the shared NSEClient singleton.

    All HTTP (cookies, rate limiting, retries) is handled by NSEClient.
    This class is responsible only for NSE API URL construction and
    JSON → model parsing.
    """

    def __init__(self) -> None:
        self._nse: Optional[NSEClient] = None

    async def _ensure_nse(self) -> NSEClient:
        if self._nse is None:
            self._nse = await NSEClient.get()
        return self._nse

    async def _get(
        self, url: str, params: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Authenticated GET via the shared NSE client."""
        client = await self._ensure_nse()
        return await client.get_json(url, params=params)

    # ── Decimal helpers ─────────────────────────────────────────

    @staticmethod
    def _to_decimal_sync(value) -> Optional[Decimal]:
        if value is None:
            return None
        try:
            return Decimal(str(value))
        except (ValueError, TypeError):
            return None

    async def _to_decimal(self, value) -> Optional[Decimal]:
        return self._to_decimal_sync(value)

    # ── Price Data ──────────────────────────────────────────────

    async def collect_price_data(self, symbol: str) -> Optional[PriceData]:
        url = f"{cfg.nse_quote_url}/equity/{symbol}"
        data = await self._get(url)
        if not data:
            return None

        try:
            price_info = data.get("priceInfo", {})
            ohlc = price_info.get("ohlc", {})
            delivery_info = data.get("securityWiseDP", {}) or {}

            return PriceData(
                symbol=symbol,
                timestamp=datetime.now(),
                open=await self._to_decimal(ohlc.get("open")) or Decimal("0"),
                high=(
                    await self._to_decimal(
                        price_info.get("intraDayHighHigh", {}).get("value")
                    )
                    or await self._to_decimal(ohlc.get("high"))
                    or Decimal("0")
                ),
                low=(
                    await self._to_decimal(
                        price_info.get("intraDayHighLow", {})
                        .get("low", {})
                        .get("value")
                    )
                    or await self._to_decimal(ohlc.get("low"))
                    or Decimal("0")
                ),
                close=await self._to_decimal(price_info.get("lastPrice"))
                or Decimal("0"),
                volume=int(
                    price_info.get("totalTradedVolume")
                    or price_info.get("quantity", 0)
                ),
                vwap=await self._to_decimal(price_info.get("averagePrice", 0)),
                delivery_volume=self._extract_delivery_volume(delivery_info),
                delivery_percent=self._extract_delivery_percent(delivery_info),
                prev_close=await self._to_decimal(ohlc.get("previousClose")),
                change_percent=await self._to_decimal(
                    price_info.get("pChange")
                ),
            )
        except Exception as exc:
            logger.error("Error parsing price data for %s: %s", symbol, exc)
            return None

    def _extract_delivery_volume(self, delivery_info: Dict) -> Optional[int]:
        if isinstance(delivery_info, dict):
            return delivery_info.get("deliveryQuantity")
        return None

    def _extract_delivery_percent(
        self, delivery_info: Dict
    ) -> Optional[Decimal]:
        if isinstance(delivery_info, dict):
            val = delivery_info.get("deliveryToTradedQuantity")
            if val is not None:
                return Decimal(str(val))
        return None

    # ── Option Chain ────────────────────────────────────────────

    async def collect_option_chain(
        self, symbol: str
    ) -> Optional[OptionChainData]:
        data = await self._get(cfg.nse_option_chain_url, params={"symbol": symbol})
        if not data:
            return None

        try:
            records = data.get("records", {})
            expiry_dates = records.get("expiryDates", [])
            current_expiry = expiry_dates[0] if expiry_dates else None
            strike_data = records.get("data", [])

            strikes: List[OptionStrike] = []
            total_ce_oi = 0
            total_pe_oi = 0

            for item in strike_data:
                ce = item.get("CE")
                pe = item.get("PE")
                strike_price = Decimal(str(item.get("strikePrice", 0)))

                if ce:
                    strikes.append(
                        self._parse_option_strike(ce, strike_price, "CE")
                    )
                    total_ce_oi += ce.get("openInterest", 0) or 0

                if pe:
                    strikes.append(
                        self._parse_option_strike(pe, strike_price, "PE")
                    )
                    total_pe_oi += pe.get("openInterest", 0) or 0

            pcr_val = None
            if total_ce_oi > 0:
                pcr_val = Decimal(str(total_pe_oi / total_ce_oi))

            iv_list = [
                s.iv for s in strikes if s.iv is not None and s.iv > 0
            ]

            return OptionChainData(
                symbol=symbol,
                timestamp=datetime.now(),
                expiry_date=current_expiry,
                strikes=[s.__dict__ for s in strikes],
                pcr=pcr_val,
                total_ce_oi=total_ce_oi,
                total_pe_oi=total_pe_oi,
                max_pain=self._calculate_max_pain(strikes),
                iv_rank=self._calculate_iv_rank(iv_list),
                iv_percentile=self._calculate_iv_percentile(iv_list),
            )
        except Exception as exc:
            logger.error("Error parsing option chain for %s: %s", symbol, exc)
            return None

    def _parse_option_strike(
        self, data: Dict, strike_price: Decimal, option_type: str
    ) -> OptionStrike:
        greeks = data.get("greeks", {}) or {}
        return OptionStrike(
            strike_price=strike_price,
            expiry_date=data.get("expiryDate", ""),
            option_type=option_type,
            open_interest=data.get("openInterest", 0) or 0,
            change_in_oi=data.get("changeinOpenInterest", 0) or 0,
            volume=data.get("totalTradedVolume", 0) or 0,
            iv=Decimal(str(data.get("impliedVolatility", 0) or 0)),
            last_price=Decimal(str(data.get("lastPrice", 0) or 0)),
            delta=self._to_decimal_sync(greeks.get("delta")),
            gamma=self._to_decimal_sync(greeks.get("gamma")),
            theta=self._to_decimal_sync(greeks.get("theta")),
            vega=self._to_decimal_sync(greeks.get("vega")),
        )

    def _calculate_iv_rank(self, iv_list: List[Decimal]) -> Optional[Decimal]:
        if len(iv_list) < 2:
            return None
        vals = sorted(iv_list)
        min_iv, max_iv = vals[0], vals[-1]
        if max_iv == min_iv:
            return Decimal("50")
        rank = (vals[-1] - min_iv) / (max_iv - min_iv) * 100
        return Decimal(str(round(rank, 2)))

    def _calculate_iv_percentile(
        self, iv_list: List[Decimal]
    ) -> Optional[Decimal]:
        if not iv_list:
            return None
        vals = sorted(iv_list)
        n = len(vals)
        current = vals[-1]
        count_less = sum(1 for v in vals if v <= current)
        return Decimal(str(round((count_less / n) * 100, 2)))

    def _calculate_max_pain(
        self, strikes: List[OptionStrike]
    ) -> Optional[Decimal]:
        if not strikes:
            return None
        pain_by_strike: Dict[Decimal, Decimal] = {}
        for s in strikes:
            sp = s.strike_price
            if sp not in pain_by_strike:
                pain_by_strike[sp] = Decimal("0")
            otm_ce = [
                x
                for x in strikes
                if x.option_type == "CE" and x.strike_price >= sp
            ]
            otm_pe = [
                x
                for x in strikes
                if x.option_type == "PE" and x.strike_price <= sp
            ]
            ce_pain = sum(
                (x.strike_price - sp) * Decimal(str(x.volume))
                for x in otm_ce
                if x.volume > 0
            )
            pe_pain = sum(
                (sp - x.strike_price) * Decimal(str(x.volume))
                for x in otm_pe
                if x.volume > 0
            )
            pain_by_strike[sp] = abs(ce_pain) + abs(pe_pain)
        if pain_by_strike:
            return min(pain_by_strike, key=pain_by_strike.get)
        return None

    # ── Market Breadth ──────────────────────────────────────────

    async def collect_market_breadth(
        self,
    ) -> Optional[MarketBreadthData]:
        url = f"{cfg.nse_base_url}/api/marketStatus"
        data = await self._get(url)
        if not data:
            return None

        try:
            market_state = data.get("marketState", [{}])
            breadth_info = {}
            for state in market_state:
                if "advance" in state:
                    breadth_info = state
                    break

            advances = int(breadth_info.get("advance", data.get("advances", 0)))
            declines = int(
                breadth_info.get("decline", data.get("declines", 0))
            )
            unchanged = int(
                breadth_info.get("unchanged", data.get("unchanged", 0))
            )

            ad_ratio = None
            if declines > 0:
                ad_ratio = Decimal(str(round(advances / declines, 4)))

            return MarketBreadthData(
                timestamp=datetime.now(),
                advances=advances,
                declines=declines,
                unchanged=unchanged,
                advance_decline_ratio=ad_ratio,
                total_traded=advances + declines + unchanged,
            )
        except Exception as exc:
            logger.error("Error parsing market breadth: %s", exc)
            return None

    # ── FII / DII Flow ─────────────────────────────────────────

    async def collect_fii_dii_flow(
        self,
    ) -> Optional[InstitutionalFlowData]:
        data = await self._get("https://www.nseindia.com/api/fiidii")
        if not data:
            return None

        try:
            rows = data if isinstance(data, list) else data.get("data", [])
            if rows:
                latest = rows[0]
                return InstitutionalFlowData(
                    timestamp=datetime.now(),
                    fii_cash=await self._to_decimal(
                        latest.get("fiiBuy") or latest.get("fiiCash")
                    ),
                    fii_fno=await self._to_decimal(
                        latest.get("fiiSell") or latest.get("fiiFno")
                    ),
                    dii_cash=await self._to_decimal(
                        latest.get("diiBuy") or latest.get("diiCash")
                    ),
                )
            return None
        except Exception as exc:
            logger.error("Error parsing FII/DII data: %s", exc)
            return None

    # ── Index Data ──────────────────────────────────────────────

    async def collect_index_data(self) -> Dict[str, IndexData]:
        data = await self._get(cfg.nse_index_url)
        result: Dict[str, IndexData] = {}
        if not data:
            return result

        try:
            for idx_data in data.get("data", []):
                idx_name = idx_data.get("index", "")
                if idx_name not in cfg.index_symbols:
                    continue

                ohlc = idx_data.get("ohlc", {})
                result[idx_name] = IndexData(
                    symbol=idx_name,
                    timestamp=datetime.now(),
                    open=await self._to_decimal(ohlc.get("open"))
                    or Decimal("0"),
                    high=await self._to_decimal(ohlc.get("high"))
                    or Decimal("0"),
                    low=await self._to_decimal(ohlc.get("low"))
                    or Decimal("0"),
                    close=await self._to_decimal(idx_data.get("last"))
                    or Decimal("0"),
                    change_percent=await self._to_decimal(
                        idx_data.get("percentChange")
                    ),
                )
        except Exception as exc:
            logger.error("Error parsing index data: %s", exc)

        return result

    # ── VIX ─────────────────────────────────────────────────────

    async def collect_vix(self) -> Optional[VIXData]:
        data = await self._get(
            f"{cfg.nse_base_url}/api/option-chain-indices",
            params={"symbol": "INDIAVIX"},
        )
        if not data:
            return None

        try:
            records = data.get("records", {})
            underlying = records.get("underlying", {})
            val = await self._to_decimal(
                underlying.get("value") or data.get("last")
            )
            prev = await self._to_decimal(
                underlying.get("previousClose") or data.get("prevClose")
            )

            change = None
            if val is not None and prev is not None and prev > 0:
                change = Decimal(
                    str(round((val - prev) / prev * 100, 2))
                )

            return VIXData(
                timestamp=datetime.now(),
                value=val or Decimal("0"),
                change_percent=change,
                prev_close=prev,
            )
        except Exception as exc:
            logger.error("Error parsing VIX data: %s", exc)
            return None

    # ── Convenience ─────────────────────────────────────────────

    async def collect_all_for_symbol(
        self, symbol: str
    ) -> Tuple[Optional[PriceData], Optional[OptionChainData]]:
        price_task = self.collect_price_data(symbol)
        option_task = self.collect_option_chain(symbol)
        results = await asyncio.gather(
            price_task, option_task, return_exceptions=True
        )
        price_data = (
            results[0] if not isinstance(results[0], Exception) else None
        )
        option_data = (
            results[1] if not isinstance(results[1], Exception) else None
        )
        return price_data, option_data

    async def collect_market_snapshot(self) -> Dict[str, Dict]:
        snapshot: Dict[str, Dict] = {
            "symbols": {},
            "indices": {},
            "breadth": None,
            "vix": None,
            "institutional": None,
            "timestamp": datetime.now().isoformat(),
        }

        breadth, vix, inst = await asyncio.gather(
            self.collect_market_breadth(),
            self.collect_vix(),
            self.collect_fii_dii_flow(),
            return_exceptions=True,
        )
        if not isinstance(breadth, Exception):
            snapshot["breadth"] = breadth
        if not isinstance(vix, Exception):
            snapshot["vix"] = vix
        if not isinstance(inst, Exception):
            snapshot["institutional"] = inst

        indices = await self.collect_index_data()
        if not isinstance(indices, Exception):
            snapshot["indices"] = indices

        symbol_results = await asyncio.gather(
            *[self.collect_all_for_symbol(s) for s in cfg.fno_symbols],
            return_exceptions=True,
        )
        for i, result in enumerate(symbol_results):
            if isinstance(result, Exception):
                continue
            price_data, option_data = result
            snapshot["symbols"][cfg.fno_symbols[i]] = {
                "price": price_data,
                "options": option_data,
            }

        return snapshot

    # ── Context manager (no-op close — NSEClient owns the httpx session) ──

    async def close(self) -> None:
        pass

    async def __aenter__(self) -> "MarketDataCollector":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

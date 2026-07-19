"""Bhavcopy data engine — downloads daily F&O trading data from NSE.

Downloads the F&O Bhavcopy ZIP from nsearchives.nseindia.com which contains:
  - fo*.csv: F&O futures + stock data (200+ stocks with close, change, OI, volume)
  - op*.csv: Options data (37,000+ option contracts)

Falls back to allIndices API (139 indices with PE/PB/DY) if ZIP download fails.

Provides individual stock-level data for AI scoring — the real Bhavcopy.
"""

from __future__ import annotations

import csv
import io
import logging
import re
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class StockData:
    """Individual stock data from Bhavcopy."""
    symbol: str
    close: float = 0.0
    prev_close: float = 0.0
    change: float = 0.0
    change_pct: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    oi: float = 0.0
    volume: float = 0.0
    turnover: float = 0.0
    settlement: float = 0.0


@dataclass
class IndexData:
    """Index data from allIndices API."""
    symbol: str
    last: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    prev_close: float = 0.0
    change_pct: float = 0.0
    pe: float = 0.0
    pb: float = 0.0
    dy: float = 0.0
    advances: int = 0
    declines: int = 0
    unchanged: int = 0
    per_change_30d: float = 0.0
    per_change_365d: float = 0.0
    year_high: float = 0.0
    year_low: float = 0.0
    traded_volume: float = 0.0


@dataclass
class OptionData:
    """Option contract data from op*.csv."""
    symbol: str
    contract: str
    expiry: str
    option_type: str
    strike: float = 0.0
    close: float = 0.0
    settlement: float = 0.0
    oi: float = 0.0
    volume: float = 0.0
    underlying: float = 0.0
    notional: float = 0.0
    premium_traded: float = 0.0


@dataclass
class BhavcopyData:
    """Complete Bhavcopy dataset."""
    timestamp: str = ""
    stocks: Dict[str, StockData] = field(default_factory=dict)
    indices: Dict[str, IndexData] = field(default_factory=dict)
    options: Dict[str, List[OptionData]] = field(default_factory=dict)
    source: str = "none"
    fo_count: int = 0
    option_count: int = 0
    advances: int = 0
    declines: int = 0


class BhavcopyEngine:
    """Downloads and caches daily F&O Bhavcopy data from NSE."""

    def __init__(self) -> None:
        self._cache: Optional[BhavcopyData] = None
        self._cache_time: float = 0.0
        self._cache_ttl: float = 86400.0  # 24h — EOD data, only refresh once daily

    async def get_bhavcopy(self) -> BhavcopyData:
        """Get cached or fresh bhavcopy data.

        Strategy:
        1. Use cached data if fresh (< 24h for EOD, < 5 min for Yahoo)
        2. Try F&O Bhavcopy ZIP (EOD — available after market close)
        3. Supplement with Yahoo Finance during market hours (delayed 15 min)
        4. Merge: Yahoo prices override stale Bhavcopy prices, Bhavcopy OI preserved
        """
        # Return cache if still fresh
        if self._cache and (time.time() - self._cache_time) < self._cache_ttl:
            return self._cache

        data = BhavcopyData(source="none")

        # 1. Try F&O Bhavcopy ZIP (individual stock data — the gold)
        fo_data = await self._download_fo_zip()
        if fo_data:
            data.stocks.update(fo_data.get("stocks", {}))
            data.options.update(fo_data.get("options", {}))
            data.fo_count = len(data.stocks)
            data.option_count = sum(len(v) for v in data.options.values())
            data.source = "fo_zip"
            data.timestamp = datetime.now().isoformat()

        # 2. Get allIndices for PE/PB/DY (index-level)
        index_data = await self._fetch_all_indices()
        if index_data:
            data.indices = index_data
            if not data.source:
                data.source = "allIndices"
                data.timestamp = datetime.now().isoformat()

        # 3. Yahoo Finance: live prices during market hours, fallback if no bhavcopy
        from trademind.engines.bhavcopy.yahoo import yahoo_provider
        from trademind.config.settings import settings as cfg
        from trademind.config.constants import MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE, MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE
        from datetime import timezone, timedelta

        IST = timezone(timedelta(hours=5, minutes=30))
        now_ist = datetime.now(IST)
        market_open = now_ist.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0, microsecond=0)
        market_close = now_ist.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0, microsecond=0)
        is_market_hours = market_open <= now_ist <= market_close

        if is_market_hours or not data.stocks:
            # During market hours: Yahoo gives live(ish) prices
            # No bhavcopy yet: Yahoo is the only source
            try:
                # Use Bhavcopy stock list (208+) if available, else fallback to fno_symbols
                stock_symbols = list(data.stocks.keys()) if data.stocks else cfg.fno_symbols
                index_symbols = cfg.index_symbols
                yahoo_stocks, yahoo_indices = await yahoo_provider.fetch_all(
                    stock_symbols, index_symbols,
                )
                if yahoo_stocks:
                    # Merge: Yahoo prices override, Bhavcopy OI preserved
                    for sym, y_stock in yahoo_stocks.items():
                        if sym in data.stocks:
                            # Keep Bhavcopy OI/options, update prices from Yahoo
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
                            # New stock from Yahoo (not in Bhavcopy)
                            data.stocks[sym] = y_stock
                    data.fo_count = len(data.stocks)
                    if not data.source or data.source == "none":
                        data.source = "yahoo_delayed"
                    logger.info("Yahoo supplemented: %d stocks merged", len(yahoo_stocks))

                if yahoo_indices:
                    for name, y_idx in yahoo_indices.items():
                        if name in data.indices:
                            # Update prices from Yahoo, keep PE/PB from allIndices
                            bh = data.indices[name]
                            bh.last = y_idx.last
                            bh.open = y_idx.open
                            bh.high = y_idx.high
                            bh.low = y_idx.low
                            bh.change_pct = y_idx.change_pct
                            bh.prev_close = y_idx.prev_close
                            bh.traded_volume = y_idx.traded_volume
                        else:
                            data.indices[name] = y_idx
            except Exception as exc:
                logger.warning("Yahoo fallback failed: %s", exc)

        # 4. Compute advances/declines from stock data
        for stock in data.stocks.values():
            if stock.change_pct > 0:
                data.advances += 1
            elif stock.change_pct < 0:
                data.declines += 1

        self._cache = data
        self._cache_time = time.time()

        logger.info(
            "Bhavcopy loaded: %d stocks, %d indices, %d option contracts (source=%s)",
            data.fo_count, len(data.indices), data.option_count, data.source,
        )
        return data

    async def _download_fo_zip(self) -> Optional[dict]:
        """Download and parse F&O Bhavcopy ZIP from nsearchives.nseindia.com."""
        try:
            async with httpx.AsyncClient(
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "*/*",
                },
                follow_redirects=True,
                timeout=30,
            ) as client:
                # Step 1: Get cookies from homepage
                resp = await client.get("https://www.nseindia.com", timeout=15)
                if resp.status_code != 200:
                    logger.warning("NSE homepage returned %d", resp.status_code)
                    return None

                # Step 2: Download FO bhavcopy ZIP
                resp2 = await client.get(
                    "https://nsearchives.nseindia.com/content/fo/fo.zip",
                    timeout=30,
                )
                if resp2.status_code != 200 or len(resp2.content) < 500:
                    logger.warning("FO ZIP download failed: status=%d size=%d", resp2.status_code, len(resp2.content))
                    return None

                return self._parse_fo_zip(resp2.content)

        except Exception as exc:
            logger.warning("FO Bhavcopy download failed: %s", exc)
            return None

    def _parse_fo_zip(self, content: bytes) -> dict:
        """Parse the F&O Bhavcopy ZIP into stocks and options."""
        stocks: Dict[str, StockData] = {}
        options: Dict[str, List[OptionData]] = {}

        try:
            z = zipfile.ZipFile(io.BytesIO(content))
        except zipfile.BadZipFile:
            logger.warning("Invalid ZIP content from NSE")
            return {}

        for name in z.namelist():
            try:
                file_content = z.read(name).decode("utf-8", errors="replace")
                reader = csv.DictReader(io.StringIO(file_content))

                if name.startswith("fo"):
                    stocks = self._parse_fo_csv(reader)
                elif name.startswith("op"):
                    options = self._parse_op_csv(reader)
            except Exception as exc:
                logger.warning("Failed to parse %s: %s", name, exc)

        return {"stocks": stocks, "options": options}

    def _parse_fo_csv(self, reader: csv.DictReader) -> Dict[str, StockData]:
        """Parse fo*.csv — F&O futures and stock data."""
        stocks: Dict[str, StockData] = {}

        for row in reader:
            contract = row.get("CONTRACT_D", "")
            close = self._safe_float(row.get("CLOSE_PRIC"))
            prev = self._safe_float(row.get("PREVIOUS_S"))
            settlement = self._safe_float(row.get("SETTLEMENT"))

            if close <= 0:
                continue

            # Extract symbol from contract name
            sym = self._extract_symbol(contract)
            if not sym:
                continue

            # Use settlement price if close is 0, or keep close
            effective_close = settlement if close == 0 and settlement > 0 else close
            if effective_close <= 0:
                continue

            # Prefer the contract with highest OI or volume for each symbol
            if sym in stocks:
                existing = stocks[sym]
                new_oi = self._safe_float(row.get("OI_NO_CON"))
                if new_oi <= existing.oi:
                    continue

            change = self._safe_float(row.get("NET_CHANGE"))
            change_pct = (change / prev * 100) if prev > 0 else 0.0

            stocks[sym] = StockData(
                symbol=sym,
                close=effective_close,
                prev_close=prev,
                change=change,
                change_pct=round(change_pct, 2),
                open=self._safe_float(row.get("OPEN_PRICE")),
                high=self._safe_float(row.get("HIGH_PRICE")),
                low=self._safe_float(row.get("LOW_PRICE")),
                oi=self._safe_float(row.get("OI_NO_CON")),
                volume=self._safe_float(row.get("TRADED_QUA")),
                turnover=self._safe_float(row.get("TRADED_VAL")),
                settlement=settlement,
            )

        return stocks

    def _parse_op_csv(self, reader: csv.DictReader) -> Dict[str, List[OptionData]]:
        """Parse op*.csv — options data grouped by underlying symbol."""
        options: Dict[str, List[OptionData]] = {}
        count = 0

        for row in reader:
            contract = row.get("CONTRACT_D", "")
            m = re.search(r"OPTSTK([A-Z0-9&]+?)(?:\d{2})", contract)
            if not m:
                continue

            sym = m.group(1)
            close = self._safe_float(row.get("CLOSE_PRIC"))
            oi = self._safe_float(row.get("OI_NO_CON"))
            vol = self._safe_float(row.get("TRADED_QUA"))

            if close <= 0 and oi <= 0:
                continue

            # Parse option type and strike from contract
            opt_type = "CE" if "CE" in contract else "PE" if "PE" in contract else "?"
            strike_match = re.search(r"(?:CE|PE)(\d+(?:\.\d+)?)", contract)
            strike = float(strike_match.group(1)) if strike_match else 0.0

            # Parse expiry
            expiry_match = re.search(r"(\d{2}-[A-Z]{3}-\d{4})", contract)
            expiry = expiry_match.group(1) if expiry_match else ""

            od = OptionData(
                symbol=sym,
                contract=contract,
                expiry=expiry,
                option_type=opt_type,
                strike=strike,
                close=close,
                settlement=self._safe_float(row.get("SETTLEMENT")),
                oi=oi,
                volume=vol,
                underlying=self._safe_float(row.get("UNDRLNG_ST")),
                notional=self._safe_float(row.get("NOTIONAL_V")),
                premium_traded=self._safe_float(row.get("PREMIUM_TR")),
            )

            if sym not in options:
                options[sym] = []
            options[sym].append(od)
            count += 1
            if count > 50000:
                break  # Cap to avoid memory issues

        return options

    def _extract_symbol(self, contract: str) -> Optional[str]:
        """Extract stock symbol from F&O contract name.

        FUTSTKASHOKLEY29-SEP-2026 -> ASHOKLEY
        OPTSTKRELIANCE28-JUL-2026CE135 -> RELIANCE
        FUTIDXNIFTY28-JUL-2026 -> NIFTY (index, skip)
        """
        if "FUTIDX" in contract or "OPTIDX" in contract:
            return None  # Index futures/options — skip

        m = re.search(r"(?:FUTSTK|OPTSTK)([A-Z0-9&]+?)(?:\d{2}[A-Z])", contract)
        if m:
            return m.group(1)

        # Fallback
        m = re.search(r"(?:FUTSTK|OPTSTK)([A-Z0-9&]+?)(?:\d)", contract)
        if m:
            return m.group(1)

        return None

    async def _fetch_all_indices(self) -> Dict[str, IndexData]:
        """Fetch allIndices API for PE/PB/DY data."""
        try:
            async with httpx.AsyncClient(
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "*/*",
                    "Referer": "https://www.nseindia.com/",
                },
                follow_redirects=True,
            ) as client:
                # Warm up session
                await client.get("https://www.nseindia.com", timeout=15)
                resp = await client.get(
                    "https://www.nseindia.com/api/allIndices",
                    timeout=15,
                )
                if resp.status_code != 200:
                    return {}

                raw = resp.json()
                indices: Dict[str, IndexData] = {}

                for item in raw.get("data", []):
                    name = item.get("index", "")
                    if not name:
                        continue

                    indices[name] = IndexData(
                        symbol=name,
                        last=float(item.get("last", 0) or 0),
                        open=float(item.get("open", 0) or 0),
                        high=float(item.get("high", 0) or 0),
                        low=float(item.get("low", 0) or 0),
                        prev_close=float(item.get("previousClose", 0) or 0),
                        change_pct=float(item.get("percentChange", 0) or 0),
                        pe=float(item.get("pe", 0) or 0),
                        pb=float(item.get("pb", 0) or 0),
                        dy=float(item.get("dy", 0) or 0),
                        advances=int(item.get("advances", 0) or 0),
                        declines=int(item.get("declines", 0) or 0),
                        unchanged=int(item.get("unchanged", 0) or 0),
                        per_change_30d=float(item.get("perChange30d", 0) or 0),
                        per_change_365d=float(item.get("perChange365d", 0) or 0),
                        year_high=float(item.get("yearHigh", 0) or 0),
                        year_low=float(item.get("yearLow", 0) or 0),
                        traded_volume=float(item.get("tradedVolume", 0) or 0),
                    )

                return indices
        except Exception as exc:
            logger.warning("allIndices fetch failed: %s", exc)
            return {}

    def get_option_pcr(self, symbol: str) -> Optional[float]:
        """Get Put-Call Ratio from options data for a symbol."""
        opts = self._cache.options.get(symbol, []) if self._cache else []
        if not opts:
            return None

        ce_oi = sum(o.oi for o in opts if o.option_type == "CE")
        pe_oi = sum(o.oi for o in opts if o.option_type == "PE")

        return round(pe_oi / ce_oi, 3) if ce_oi > 0 else None

    def get_option_max_pain(self, symbol: str) -> Optional[float]:
        """Get max pain strike for a symbol."""
        opts = self._cache.options.get(symbol, []) if self._cache else []
        if not opts:
            return None

        strikes = sorted(set(o.strike for o in opts if o.strike > 0))
        if not strikes:
            return None

        min_pain = float("inf")
        max_pain_strike = strikes[0]

        for strike in strikes:
            pain = 0
            for o in opts:
                if o.strike <= 0:
                    continue
                if o.option_type == "CE" and o.strike < strike:
                    pain += (strike - o.strike) * o.oi
                elif o.option_type == "PE" and o.strike > strike:
                    pain += (o.strike - strike) * o.oi

            if pain < min_pain:
                min_pain = pain
                max_pain_strike = strike

        return max_pain_strike

    @staticmethod
    def _safe_float(val) -> float:
        if val is None:
            return 0.0
        try:
            s = str(val).strip()
            if not s or s == "":
                return 0.0
            return float(s)
        except (ValueError, TypeError):
            return 0.0

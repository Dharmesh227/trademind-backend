"""NSE cookie-based session manager.

Handles authentication with NSE India by:
1. Fetching cookies from the NSE homepage (which sets cf_clearance + nse_* cookies)
2. Caching cookies with a TTL (NSE cookies typically expire in ~5 min)
3. Providing authenticated async GET requests via httpx
4. Rate limiting to respect NSE's ~30 req/min threshold
5. Auto-refreshing on 403 / cookie expiry

Usage:
    async with NSEClient.get() as client:
        data = await client.get_json("https://www.nseindia.com/api/allIndices")
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import httpx

from trademind.config.settings import settings as cfg

logger = logging.getLogger(__name__)

# NSE cookies typically expire in ~5 minutes; refresh at 4 min to be safe.
_COOKIE_TTL_SECONDS: float = 240.0
_RATE_LIMIT_WINDOW: float = 60.0


class NSEClient:
    """Async NSE client with cookie-based auth, rate limiting, and retry."""

    _instance: Optional["NSEClient"] = None
    _init_lock = asyncio.Lock()

    def __init__(self) -> None:
        self._client: Optional[httpx.AsyncClient] = None
        self._cookies: Dict[str, str] = {}
        self._cookie_obtained_at: float = 0.0
        self._request_timestamps: List[float] = []
        self._lock = asyncio.Lock()

    @classmethod
    async def get(cls) -> "NSEClient":
        """Return the singleton NSEClient, creating it on first call."""
        if cls._instance is None:
            async with cls._init_lock:
                if cls._instance is None:
                    instance = NSEClient()
                    await instance._ensure_session()
                    cls._instance = instance
        return cls._instance

    @classmethod
    async def reset(cls) -> None:
        """Reset the singleton (for testing or forced re-auth)."""
        async with cls._init_lock:
            if cls._instance and cls._instance._client:
                await cls._instance._client.aclose()
            cls._instance = None

    # ── Session / Cookie Management ────────────────────────────

    async def _ensure_session(self) -> None:
        """Create the httpx client if needed."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers=cfg.headers,
                timeout=cfg.request_timeout_seconds,
                follow_redirects=True,
            )

    async def _ensure_cookies(self) -> None:
        """Fetch fresh cookies from NSE homepage if stale or missing."""
        if not self._cookies or (time.time() - self._cookie_obtained_at) > _COOKIE_TTL_SECONDS:
            await self._refresh_cookies()

    async def _refresh_cookies(self) -> bool:
        """Hit the NSE homepage to obtain a fresh set of cookies."""
        await self._ensure_session()
        try:
            resp = await self._client.get(
                cfg.nse_base_url,
                headers=cfg.headers,
                timeout=15,
            )
            if resp.status_code == 200:
                self._cookies.update(dict(resp.cookies))
                self._cookie_obtained_at = time.time()
                logger.info(
                    "NSE cookies refreshed — %d cookies obtained", len(self._cookies)
                )
                return True
            logger.warning("NSE cookie refresh returned status %d", resp.status_code)
            return False
        except Exception as exc:
            logger.warning("NSE cookie refresh failed: %s", exc)
            return False

    # ── Rate Limiting ──────────────────────────────────────────

    async def _rate_limit(self) -> None:
        """Sleep if we'd exceed 30 requests in the last 60 seconds."""
        now = time.time()
        self._request_timestamps = [
            t for t in self._request_timestamps if now - t < _RATE_LIMIT_WINDOW
        ]
        if len(self._request_timestamps) >= cfg.requests_per_minute:
            sleep_time = self._request_timestamps[0] + _RATE_LIMIT_WINDOW - now
            if sleep_time > 0:
                logger.debug("NSE rate limit: sleeping %.2fs", sleep_time)
                await asyncio.sleep(sleep_time)
        self._request_timestamps.append(time.time())

    # ── Public API ─────────────────────────────────────────────

    async def get_json(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        retries: int = 3,
        delay: float = 1.5,
    ) -> Optional[Dict]:
        """Authenticated GET that returns parsed JSON.  Retries on 403."""
        async with self._lock:
            await self._ensure_cookies()

            last_error: Optional[Exception] = None
            for attempt in range(retries):
                try:
                    await self._rate_limit()
                    await self._ensure_session()

                    resp = await self._client.get(
                        url,
                        params=params,
                        cookies=self._cookies,
                    )

                    if resp.status_code == 200:
                        self._cookies.update(dict(resp.cookies))
                        return resp.json()

                    if resp.status_code == 403 and attempt < retries - 1:
                        logger.warning(
                            "NSE 403 on %s — refreshing cookies (attempt %d)",
                            url, attempt + 1,
                        )
                        await self._refresh_cookies()
                        await asyncio.sleep(delay * (attempt + 1))
                        continue

                    logger.error(
                        "NSE request failed: %s status=%d", url, resp.status_code
                    )
                    return None

                except httpx.TimeoutException as exc:
                    last_error = exc
                    logger.warning(
                        "NSE timeout on %s (attempt %d)", url, attempt + 1
                    )
                    await asyncio.sleep(delay * (attempt + 1))
                except httpx.HTTPError as exc:
                    last_error = exc
                    logger.warning(
                        "NSE HTTP error on %s (attempt %d): %s",
                        url, attempt + 1, exc,
                    )
                    await asyncio.sleep(delay * (attempt + 1))
                except Exception as exc:
                    last_error = exc
                    logger.error("NSE unexpected error on %s: %s", url, exc)
                    return None

            logger.error("All retries failed for %s: %s", url, last_error)
            return None

    async def close(self) -> None:
        """Shut down the underlying httpx client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            logger.info("NSE client closed")

    async def __aenter__(self) -> "NSEClient":
        await self._ensure_session()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

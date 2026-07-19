"""Global NSE connection status tracker.

Provides a shared state that the health endpoint and Android app can query
to know whether live market data is available.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class NSEStatus:
    """Tracks whether NSE is reachable and when it was last checked."""

    def __init__(self) -> None:
        self.is_live: bool = False
        self.last_checked: Optional[datetime] = None
        self.last_error: Optional[str] = None
        self.indices_count: int = 0
        self.message: str = "Not initialized"

    def update(
        self,
        *,
        is_live: bool,
        last_error: Optional[str] = None,
        indices_count: int = 0,
        message: Optional[str] = None,
    ) -> None:
        self.is_live = is_live
        self.last_checked = datetime.now()
        self.last_error = last_error
        self.indices_count = indices_count
        self.message = message or (
            "Live NSE data available" if is_live else "NSE unreachable — no live data"
        )
        logger.info("NSE status: %s", self.message)

    def to_dict(self) -> dict:
        return {
            "is_live": self.is_live,
            "last_checked": self.last_checked.isoformat() if self.last_checked else None,
            "last_error": self.last_error,
            "indices_count": self.indices_count,
            "message": self.message,
        }


# Module-level singleton
nse_status = NSEStatus()

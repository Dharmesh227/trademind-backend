"""TradeMind AI — Uvicorn entry point.

Usage:
    python main.py
    # or
    uvicorn trademind.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run(
        "trademind.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()

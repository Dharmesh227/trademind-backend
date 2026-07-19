from typing import Dict, List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    database_url: str = "sqlite+aiosqlite:///./trademind.db"
    debug: bool = False

    # NSE API endpoints
    nse_base_url: str = "https://www.nseindia.com"
    nse_quote_url: str = "https://www.nseindia.com/api/quote-equity"
    nse_option_chain_url: str = "https://www.nseindia.com/api/option-chain-equities"
    nse_index_url: str = "https://www.nseindia.com/api/allIndices"
    nse_vix_url: str = "https://www.nseindia.com/api/option-chain-indices?symbol=INDIAVIX"
    nse_market_status_url: str = "https://www.nseindia.com/api/marketStatus"

    # Rate limiting
    requests_per_minute: int = 30
    max_retries: int = 3
    retry_delay_seconds: float = 1.5
    request_timeout_seconds: int = 30

    # Headers to mimic browser
    headers: Dict[str, str] = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
        "DNT": "1",
    }

    # Collection intervals (seconds)
    collection_interval: int = 300
    option_chain_interval: int = 300
    breadth_interval: int = 300
    institutional_interval: int = 3600

    # Symbols to track
    fno_symbols: List[str] = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
        "BAJFINANCE", "LT", "WIPRO", "AXISBANK", "TITAN",
        "ASIANPAINT", "MARUTI", "SUNPHARMA", "NTPC", "ONGC",
        "POWERGRID", "NESTLEIND", "M&M", "TRENT", "JSWSTEEL",
        "TECHM", "HCLTECH", "BAJAJFINSV", "ULTRACEMCO", "ADANIPORTS",
        "HDFCLIFE", "SBILIFE", "DRREDDY", "CIPLA", "BRITANNIA",
        "DIVISLAB", "GRASIM", "HINDALCO", "EICHERMOT", "COALINDIA",
        "BPCL", "IOC", "HEROMOTOCO", "BAJAJ-AUTO", "TATASTEEL",
        "BEL", "HAL", "VOLTAS", "ICICIPRULI", "MUTHOOTFIN",
    ]

    # Index symbols
    index_symbols: List[str] = [
        "NIFTY 50", "NIFTY BANK", "NIFTY IT", "NIFTY PHARMA",
        "NIFTY AUTO", "NIFTY FMCG", "NIFTY METAL", "NIFTY MEDIA",
        "NIFTY OIL & GAS", "NIFTY REALTY", "NIFTY CONSUMER DURABLES",
        "NIFTY MIDCAP 100", "NIFTY SMALLCAP 100",
    ]

    # Feature category weights
    category_weights: Dict[str, float] = {
        "trend": 0.28,
        "momentum": 0.22,
        "volume": 0.14,
        "options": 0.06,
        "sector": 0.12,
        "market": 0.04,
        "volatility": 0.14,
    }

    # Risk management
    default_risk_reward_ratio: float = 2.5
    default_atr_multiplier_sl: float = 1.5
    default_atr_multiplier_target: float = 3.75
    trailing_stop_activate_percent: float = 1.5
    trailing_stop_gap_atr: float = 1.0
    max_holding_days: int = 10
    min_holding_hours: int = 1

    # Paper trading defaults
    default_capital: float = 100000.0
    max_position_size_percent: float = 10.0
    max_open_positions: int = 10

    # Learning
    min_trades_for_learning: int = 30
    adaptive_weight_update_interval: int = 7
    pattern_min_samples: int = 20

    # Scoring thresholds
    confidence_high_threshold: float = 0.75
    confidence_medium_threshold: float = 0.50
    feature_quality_high_threshold: float = 0.80
    feature_quality_medium_threshold: float = 0.60

    model_config = {"env_prefix": "TRADEMIND_", "extra": "ignore"}


settings = Settings()


def get_settings() -> Settings:
    """Return the singleton Settings instance."""
    return settings

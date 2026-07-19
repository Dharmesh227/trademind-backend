"""Constants used across the TradeMind AI platform."""

from __future__ import annotations

from enum import Enum

# ── NSE Top 50 F&O Stocks ─────────────────────────────────
NSE_FO_SYMBOLS: list[str] = [
    "ACC", "ADANIENT", "ADANIPORTS", "AMBUJACEM", "APOLLOHOSP",
    "ASHOKLEY", "ASIANPAINT", "AUROPHARMA", "AXISBANK", "BAJAJ-AUTO",
    "BAJFINANCE", "BAJAJFINSV", "BPCL", "BRITANNIA", "CANBK",
    "CHAMBLFERT", "CIPLA", "COALINDIA", "DLF", "DRREDDY",
    "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE",
    "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDIGO",
    "INFY", "IOC", "ITC", "JSWSTEEL", "KOTAKBANK",
    "LT", "M&M", "MARUTI", "NESTLEIND", "NTPC",
    "ONGC", "POWERGRID", "SBIN", "SBILIFE", "SUNPHARMA",
    "TATAMOTORS", "TATASTEEL", "TCS", "TECHM", "TRENT",
    "ULTRACEMCO", "WIPRO", "ZEEL",
]

# ── Sector Mappings ────────────────────────────────────────
SECTOR_MAP: dict[str, str] = {
    "ACC": "Cement", "ADANIENT": "Diversified", "ADANIPORTS": "Infrastructure",
    "AMBUJACEM": "Cement", "APOLLOHOSP": "Healthcare", "ASHOKLEY": "Automobile",
    "ASIANPAINT": "Consumer", "AUROPHARMA": "Pharma", "AXISBANK": "Banking",
    "BAJAJ-AUTO": "Automobile", "BAJFINANCE": "NBFC", "BAJAJFINSV": "NBFC",
    "BPCL": "Oil & Gas", "BRITANNIA": "FMCG", "CANBK": "Banking",
    "CHAMBLFERT": "Chemicals", "CIPLA": "Pharma", "COALINDIA": "Mining",
    "DLF": "Real Estate", "DRREDDY": "Pharma", "EICHERMOT": "Automobile",
    "GRASIM": "Diversified", "HCLTECH": "IT", "HDFCBANK": "Banking",
    "HDFCLIFE": "Insurance", "HEROMOTOCO": "Automobile", "HINDALCO": "Metals",
    "HINDUNILVR": "FMCG", "ICICIBANK": "Banking", "INDIGO": "Aviation",
    "INFY": "IT", "IOC": "Oil & Gas", "ITC": "FMCG", "JSWSTEEL": "Metals",
    "KOTAKBANK": "Banking", "LT": "Infrastructure", "M&M": "Automobile",
    "MARUTI": "Automobile", "NESTLEIND": "FMCG", "NTPC": "Power",
    "ONGC": "Oil & Gas", "POWERGRID": "Power", "SBIN": "Banking",
    "SBILIFE": "Insurance", "SUNPHARMA": "Pharma", "TATAMOTORS": "Automobile",
    "TATASTEEL": "Metals", "TCS": "IT", "TECHM": "IT", "TRENT": "Retail",
    "ULTRACEMCO": "Cement", "WIPRO": "IT", "ZEEL": "Media",
}

# ── Feature Categories ─────────────────────────────────────
class FeatureCategory(str, Enum):
    MOMENTUM = "momentum"
    VOLUME = "volume"
    OPTION_CHAIN = "option_chain"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    VOLATILITY = "volatility"
    SENTIMENT = "sentiment"


# ── Feature Names by Category ──────────────────────────────
FEATURE_NAMES: dict[str, list[str]] = {
    FeatureCategory.MOMENTUM: [
        "rsi_14", "rsi_7", "rsi_21", "macd", "macd_signal", "macd_hist",
        "stoch_k", "stoch_d", "williams_r", "roc_10", "roc_20",
        "momentum_5d", "momentum_10d", "momentum_20d",
        "cci_20", "adx_14", "plus_di", "minus_di",
    ],
    FeatureCategory.VOLUME: [
        "volume_sma_20", "volume_ratio", "obv", "mfi_14",
        "ad_line", "vwap_deviation", "volume_price_trend",
        "accumulation_distribution", "chaikin_money_flow",
        "volume_weighted_rsi",
    ],
    FeatureCategory.OPTION_CHAIN: [
        "pcr_oi", "pcr_volume", "max_pain", "put_call_ratio_trend",
        "oi_change_bullish", "oi_change_bearish", "iv_percentile",
        "iv_rank", "iv_skew", "gamma_exposure",
        "delta_exposure", "net_theta", "total_vega",
        "put_oi_concentration", "call_oi_concentration",
    ],
    FeatureCategory.TECHNICAL: [
        "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
        "ema_5", "ema_10", "ema_20", "ema_50",
        "bollinger_upper", "bollinger_lower", "bollinger_width",
        "atr_14", "atr_20", "keltner_upper", "keltner_lower",
        "ichimoku_tenkan", "ichimoku_kijun", "ichimoku_senkou_a",
        "ichimoku_senkou_b", "pivot_point", "support_1", "resistance_1",
        "support_2", "resistance_2", "fib_382", "fib_500", "fib_618",
    ],
    FeatureCategory.FUNDAMENTAL: [
        "pe_ratio", "pb_ratio", "roe", "roa", "profit_margin",
        "revenue_growth", "earnings_growth", "debt_to_equity",
        "current_ratio", "dividend_yield", "market_cap_rank",
        "sector_momentum", "sector_relative_strength",
    ],
    FeatureCategory.VOLATILITY: [
        "realized_vol_10", "realized_vol_20", "realized_vol_60",
        "garman_klass_vol", "parkinson_vol", "yang_zhang_vol",
        "atr_ratio", "bollinger_squeeze", "keltner_squeeze",
        "historical_vol_ratio", "vix_nifty", "term_structure_slope",
    ],
    FeatureCategory.SENTIMENT: [
        "news_sentiment_score", "news_volume_7d", "social_buzz_score",
        "fii_flow_signal", "dii_flow_signal", "block_deal_signal",
        "insider_transaction_signal", "market_breadth",
        "advance_decline_ratio", "put_call_panic_index",
    ],
}

# ── Default Feature Weights ────────────────────────────────
DEFAULT_WEIGHTS: dict[str, float] = {
    FeatureCategory.MOMENTUM: 0.20,
    FeatureCategory.VOLUME: 0.15,
    FeatureCategory.OPTION_CHAIN: 0.20,
    FeatureCategory.TECHNICAL: 0.15,
    FeatureCategory.FUNDAMENTAL: 0.10,
    FeatureCategory.VOLATILITY: 0.10,
    FeatureCategory.SENTIMENT: 0.10,
}

# ── Scoring Thresholds ─────────────────────────────────────
class ScoreCategory(str, Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


SCORE_THRESHOLDS: dict[str, tuple[float, float]] = {
    ScoreCategory.STRONG_BUY: (85.0, 100.0),
    ScoreCategory.BUY: (70.0, 85.0),
    ScoreCategory.NEUTRAL: (40.0, 70.0),
    ScoreCategory.SELL: (15.0, 40.0),
    ScoreCategory.STRONG_SELL: (0.0, 15.0),
}

# ── Risk Parameters ────────────────────────────────────────
class TradeAction(str, Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class TradeStatus(str, Enum):
    ACTIVE = "ACTIVE"
    CLOSED = "CLOSED"
    STOPPED_OUT = "STOPPED_OUT"
    TARGET_HIT = "TARGET_HIT"
    EXPIRED = "EXPIRED"


class ExitReason(str, Enum):
    STOP_LOSS = "stop_loss"
    TARGET = "target"
    TRAILING_STOP = "trailing_stop"
    TIME_EXIT = "time_exit"
    MANUAL = "manual"
    RISK_OVERRIDE = "risk_override"
    END_OF_DAY = "end_of_day"


DEFAULT_RISK_PARAMS: dict[str, float] = {
    "max_portfolio_risk_percent": 15.0,
    "max_single_trade_risk_percent": 2.0,
    "stop_loss_atr_multiplier": 2.0,
    "target_risk_reward_ratio": 2.5,
    "max_drawdown_percent": 10.0,
    "max_concurrent_trades": 10,
    "trailing_stop_activation_percent": 3.0,
    "trailing_stop_distance_percent": 1.5,
    "max_holding_period_days": 15,
    "min_holding_period_days": 1,
}

# ── Technical Indicator Periods ─────────────────────────────
INDICATOR_PERIODS: dict[str, int] = {
    "rsi_short": 7,
    "rsi_medium": 14,
    "rsi_long": 21,
    "sma_very_short": 5,
    "sma_short": 10,
    "sma_medium": 20,
    "sma_long": 50,
    "sma_very_long": 200,
    "ema_short": 5,
    "ema_medium": 10,
    "ema_long": 20,
    "ema_very_long": 50,
    "atr_short": 14,
    "atr_long": 20,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bollinger_period": 20,
    "bollinger_std": 2,
    "adx_period": 14,
    "stoch_k": 14,
    "stoch_d": 3,
    "cci_period": 20,
    "mfi_period": 14,
    "williams_period": 14,
    "obv_sma": 20,
    "volume_sma": 20,
}

# ── NSE Index Symbols ──────────────────────────────────────
NSE_INDICES: dict[str, str] = {
    "NIFTY 50": "^NSEI",
    "NIFTY Bank": "^NSEBANK",
    "NIFTY IT": "^CNXIT",
    "NIFTY Midcap 100": "NIFTY_MIDCAP_100",
    "NIFTY Next 50": "NIFTY_NEXT_50",
    "NIFTY Financial Services": "NIFTY_FIN_SERVICE",
    "NIFTY Pharma": "NIFTY_PHARMA",
    "NIFTY FMCG": "NIFTY_FMCG",
    "NIFTY Metal": "NIFTY_METAL",
    "NIFTY Auto": "NIFTY_AUTO",
    "India VIX": "INDIA_VIX",
}

# ── Market Session Timings (IST) ───────────────────────────
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30
PRE_MARKET_OPEN_HOUR = 9
PRE_MARKET_OPEN_MINUTE = 0

# ── Data Freshness Thresholds ──────────────────────────────
MAX_STALENESS_MINUTES: dict[str, int] = {
    "market_data": 5,
    "option_chain": 15,
    "fundamental": 1440,
    "sentiment": 60,
}

# ── File & Path Constants ──────────────────────────────────
MODEL_EXTENSIONS: dict[str, str] = {
    "sklearn": ".pkl",
    "numpy": ".npy",
    "json": ".json",
}

LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name}:{function}:{line} | "
    "{message}"
)

# ── Database Table Names ───────────────────────────────────
TABLE_PREFIX = "tm_"

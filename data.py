"""
OHLCV fetching, split by source since Deriv doesn't offer individual
equities:

  - Deriv public WebSocket (no API token required for market data):
    forex, crypto, and indices/synthetics. Native candle granularities
    mean no resampling hack is needed for 4H like the old yfinance-only
    version required.
  - yfinance: individual stocks only (and as a general fallback).

Either way, the most recent candle is always dropped before returning
-- it may still be forming. Pattern detection only ever evaluates
fully closed candles, so a late-running scheduled job never causes a
false read.
"""

import json
import time
import pandas as pd

DERIV_WS_URL = "wss://api.derivws.com/trading/v1/options/ws/public"

GRANULARITY_SECONDS = {
    "1D": 86400,
    "4H": 14400,
    "1H": 3600,
    "15m": 900,
    "5m": 300,
}

# Deriv has no real traded-volume concept for forex/synthetics (it's a
# decentralized/synthetic market, same reason most retail forex
# platforms show "tick volume" instead). Tick COUNT per candle is a
# legitimate, standard proxy -- but fetching raw ticks to build it is
# only worth the extra round-trip on the faster timeframes, where the
# tick volume is manageable and actually meaningful; 1D/4H candles
# would require fetching an unreasonable number of ticks for little
# benefit, so those are left at Volume=0.
TICK_VOLUME_TIMEFRAMES = {"5m", "15m", "1H"}


def _ws_request(request: dict, timeout: int = 20) -> dict:
    import websocket  # websocket-client
    ws = websocket.create_connection(DERIV_WS_URL, timeout=timeout)
    try:
        ws.send(json.dumps(request))
        raw = ws.recv()
    finally:
        ws.close()
    return json.loads(raw)


def _fetch_tick_counts(symbol: str, start_epoch: int, end_epoch: int,
                        granularity: int) -> dict:
    """Best-effort: fetches raw ticks for [start_epoch, end_epoch] and
    buckets them into `granularity`-second candle periods, returning
    {bucket_start_epoch: tick_count}. Returns {} on any failure --
    callers should treat this as optional, not required."""
    try:
        payload = _ws_request({
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "start": start_epoch,
            "end": end_epoch,
            "style": "ticks",
        })
        if payload.get("error"):
            return {}
        times = payload.get("history", {}).get("times", [])
        counts = {}
        for t in times:
            bucket = (int(t) // granularity) * granularity
            counts[bucket] = counts.get(bucket, 0) + 1
        return counts
    except Exception:
        return {}  # tick volume is a nice-to-have, never block the scan on it


def fetch_ohlcv_deriv(symbol: str, timeframe: str, count: int = 200) -> pd.DataFrame:
    """Fetches candles from Deriv's public endpoint. No auth needed."""
    try:
        import websocket  # noqa: F401  (import check only)
    except ImportError as e:
        raise RuntimeError(
            "websocket-client is required for Deriv data -- add it to requirements.txt"
        ) from e

    granularity = GRANULARITY_SECONDS.get(timeframe)
    if granularity is None:
        raise ValueError(f"Unknown timeframe for Deriv granularity: {timeframe}")

    payload = _ws_request({
        "ticks_history": symbol,
        "adjust_start_time": 1,
        "count": count,
        "end": "latest",
        "start": 1,
        "style": "candles",
        "granularity": granularity,
    })
    if payload.get("error"):
        raise RuntimeError(f"Deriv API error for {symbol}: {payload['error'].get('message')}")

    candles = payload.get("candles", [])
    if not candles:
        return pd.DataFrame()

    df = pd.DataFrame(candles)
    epochs = df["epoch"].astype(int).tolist()
    df["Datetime"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
    df = df.set_index("Datetime")
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    df = df[["Open", "High", "Low", "Close"]].astype(float)
    df["Volume"] = 0  # default; replaced with tick-count proxy below where feasible

    if timeframe in TICK_VOLUME_TIMEFRAMES and len(candles) > 1:
        start_epoch = epochs[0]
        end_epoch = epochs[-1] + granularity
        tick_counts = _fetch_tick_counts(symbol, start_epoch, end_epoch, granularity)
        if tick_counts:
            df["Volume"] = [tick_counts.get(e, 0) for e in epochs]

    if len(df) > 1:
        df = df.iloc[:-1]  # drop the still-forming candle
    return df


_YF_INTERVAL_MAP = {
    "1D": ("1d", "1y"),
    "4H": ("1h", "3mo"),   # resampled below -- yfinance has no native 4h
    "1H": ("1h", "1mo"),
    "15m": ("15m", "5d"),
    "5m": ("5m", "5d"),
}


def fetch_ohlcv_yfinance(symbol: str, timeframe: str) -> pd.DataFrame:
    import yfinance as yf

    interval, period = _YF_INTERVAL_MAP[timeframe]
    df = yf.download(tickers=symbol, interval=interval, period=period,
                      progress=False, auto_adjust=False, multi_level_index=False)
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.rename(columns=str.title)
    df.index.name = "Datetime"

    if timeframe == "4H":
        agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
        df = df.resample("4h").agg(agg).dropna(how="any")

    if len(df) > 1:
        df = df.iloc[:-1]
    return df


def fetch_ohlcv(symbol: str, timeframe: str, source: str = "deriv") -> pd.DataFrame:
    """
    source: "deriv" (forex/crypto/indices, no token needed) or
    "yfinance" (individual stocks, or manual fallback).
    """
    if source == "deriv":
        return fetch_ohlcv_deriv(symbol, timeframe)
    elif source == "yfinance":
        return fetch_ohlcv_yfinance(symbol, timeframe)
    raise ValueError(f"Unknown data source: {source}")

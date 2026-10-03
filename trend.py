"""
Two upstream filters, applied before a pattern is scored:

1. Trend alignment (Dow Theory tenet 3 + Vol.3's RSI-50 regime read):
   classifies the higher timeframe as bullish/bearish/no-clear-trend
   from swing structure (higher-highs/higher-lows etc.), cross-checked
   against the RSI-50 regime read. This is a TAG, not a hard block --
   a pattern trading WITH this trend gets a confidence boost; a pattern
   AGAINST it still fires, just flagged and held to a stricter bar.

2. Regime/volatility filter: patterns detected during abnormally low
   volatility are noisy zigzag artifacts, not real structure. ATR-based
   floor relative to the symbol's own recent history.
"""

from dataclasses import dataclass
from typing import List
import pandas as pd
import numpy as np

from patterns import Pivot, zigzag_pivots
from indicators import wilder_rsi, trend_regime_from_rsi


@dataclass
class TrendState:
    price_structure: str   # "bullish", "bearish", "ranging"
    rsi_regime: str         # "bullish", "bearish", "neutral"
    agreement: bool         # do the two independent reads agree?


def classify_price_structure(pivots: List[Pivot], lookback: int = 6) -> str:
    """Dow Theory tenet 3: successive higher-highs+higher-lows = bullish,
    successive lower-highs+lower-lows = bearish, otherwise ranging."""
    recent = pivots[-lookback:] if len(pivots) >= lookback else pivots
    highs = [p.price for p in recent if p.kind == "H"]
    lows = [p.price for p in recent if p.kind == "L"]
    if len(highs) < 2 or len(lows) < 2:
        return "ranging"

    higher_highs = all(highs[i] < highs[i + 1] for i in range(len(highs) - 1))
    higher_lows = all(lows[i] < lows[i + 1] for i in range(len(lows) - 1))
    lower_highs = all(highs[i] > highs[i + 1] for i in range(len(highs) - 1))
    lower_lows = all(lows[i] > lows[i + 1] for i in range(len(lows) - 1))

    if higher_highs and higher_lows:
        return "bullish"
    if lower_highs and lower_lows:
        return "bearish"
    return "ranging"


def get_trend_state(df: pd.DataFrame, zigzag_threshold_pct: float) -> TrendState:
    pivots = zigzag_pivots(df, zigzag_threshold_pct)
    price_structure = classify_price_structure(pivots)
    rsi = wilder_rsi(df["Close"])
    rsi_regime = trend_regime_from_rsi(rsi)

    agreement = (
        (price_structure == "bullish" and rsi_regime == "bullish") or
        (price_structure == "bearish" and rsi_regime == "bearish") or
        (price_structure == "ranging" and rsi_regime == "neutral")
    )
    return TrendState(price_structure, rsi_regime, agreement)


def pattern_alignment_tag(pattern_direction: str, htf_state: TrendState) -> str:
    """Returns 'with-trend', 'counter-trend', or 'neutral' -- a TAG for
    scoring/confluence, never a hard reject (see module docstring)."""
    htf_trend = htf_state.price_structure
    if htf_trend == "ranging":
        return "neutral"
    if pattern_direction == htf_trend:
        return "with-trend"
    return "counter-trend"


# ------------------------------------------------------------------ #
# Regime / volatility filter
# ------------------------------------------------------------------ #

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def volatility_regime_ok(df: pd.DataFrame, lookback: int = 100,
                          floor_percentile: float = 0.25) -> bool:
    """Rejects scanning when current ATR sits in the bottom quartile of
    its own recent history for this symbol/timeframe -- a dead, choppy
    market throws off noisy zigzag pivots that technically satisfy
    ratio checks without being real structure."""
    series = atr(df).dropna()
    if len(series) < 20:
        return True  # not enough history to judge -- don't block
    recent = series.tail(lookback)
    current = recent.iloc[-1]
    threshold = recent.quantile(floor_percentile)
    return current >= threshold

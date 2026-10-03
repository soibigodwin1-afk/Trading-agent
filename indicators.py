"""
RSI (Wilder) and RSI BAMM divergence -- the only indicator with real
documented backing across the Carney books (Vol. 2 Ch. 6, Vol. 3 Ch. 5).

Key rules pulled directly from the source material, not approximated:
  - Standard thresholds: oversold <31 (i.e. effectively <=30), overbought
    >=70. A secondary "deep extreme" tier exists at 80/20.
  - The 50-level is the trend-regime gauge: RSI holding above 50 on
    pullbacks implies a bullish regime; holding below 50 implies bearish.
  - A valid RSI BAMM (M or W divergence shape) requires RSI to cross
    back through 50 BETWEEN the two extreme readings. Two extremes
    without a 50-crossing between them do not count.
  - Per Vol. 3 Ch. 5: harmonic completions should have SOME indicator
    confirmation in every instance -- this is treated as a hard gate,
    not an optional confidence boost.
  - Per Vol. 3 Ch. 8: confirmation is meant to be READ ON THE PROXIMATE
    timeframe (the next-shorter one), timed around the Primary
    pattern's completion bar -- not on the Primary timeframe itself.
    Callers should pass the proximate-timeframe dataframe here, not
    the primary one the pattern was identified on.

Two distinct BAMM shapes, checked by two different functions, because
they are genuinely different structures with different costs:
  - Type-I: the pattern's own B-to-D swing IS the divergence (RSI
    extreme near B, RSI crosses back through 50 as price runs B->C,
    comes back down into D). Confirmation completes at the same
    moment the pattern completes -- no time cost, nothing has been
    "missed" by the time this confirms.
  - Type-II: a genuine RETEST of the same PRZ after an initial
    reaction didn't fully commit -- two separate tests of the same
    zone, close together in time. This DOES cost real time/price
    (the retest has to happen), which is exactly why Type-II trades
    are managed differently (bigger objective, more discretionary)
    in the books -- it isn't a flaw, it's priced into how it's used.
"""

from dataclasses import dataclass
from typing import Optional, List
import pandas as pd
import numpy as np

RSI_OVERSOLD = 30.0
RSI_OVERBOUGHT = 70.0
RSI_DEEP_OVERSOLD = 20.0
RSI_DEEP_OVERBOUGHT = 80.0
RSI_MID = 50.0


def wilder_rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(100)  # avg_loss == 0 means pure uptrend -> RSI 100
    return rsi


@dataclass
class RsiConfirmation:
    confirmed: bool
    reason: str
    regime: str              # "bullish", "bearish", "neutral"
    divergence: bool
    deep_extreme: bool
    rsi_at_completion: float
    bamm_type: str = ""      # "type1", "type2", or "" if not confirmed


def trend_regime_from_rsi(rsi: pd.Series, lookback: int = 20) -> str:
    """Per Vol. 3 Ch. 5: RSI holding above 50 on pullbacks = bullish
    regime; holding below 50 = bearish. Used as a second, independent
    trend read alongside the price-structure (HH/HL) method."""
    recent = rsi.tail(lookback)
    if recent.empty:
        return "neutral"
    above_50_frac = (recent > RSI_MID).mean()
    if above_50_frac >= 0.65:
        return "bullish"
    if above_50_frac <= 0.35:
        return "bearish"
    return "neutral"


def _find_extreme_indices(rsi: pd.Series, kind: str) -> List[int]:
    """Indices where RSI is at/beyond the standard extreme for `kind`
    ('high' -> >=70, 'low' -> <=30)."""
    if kind == "high":
        mask = rsi >= RSI_OVERBOUGHT
    else:
        mask = rsi <= RSI_OVERSOLD
    return list(np.where(mask.values)[0])


def _divergence_in_window(rsi: pd.Series, closes: pd.Series, direction: str,
                           start_idx: int, end_idx: int) -> dict:
    """Shared core: given a window of (rsi, closes), find the first and
    last extreme RSI reading in it, check for a 50-crossing between
    them, and check for genuine price/RSI divergence. Returns a dict
    of the raw findings -- callers decide what 'confirmed' means for
    their specific shape (Type-I vs Type-II)."""
    window_rsi = rsi.iloc[max(0, start_idx):end_idx + 1].reset_index(drop=True)
    window_close = closes.iloc[max(0, start_idx):end_idx + 1].reset_index(drop=True)

    if window_rsi.empty or window_rsi.isna().all():
        return {"ok": False, "reason": "insufficient data for RSI"}

    kind = "low" if direction == "bullish" else "high"
    extreme_positions = _find_extreme_indices(window_rsi, kind)
    rsi_at_end = float(window_rsi.iloc[-1])
    deep_extreme = (rsi_at_end <= RSI_DEEP_OVERSOLD if direction == "bullish"
                     else rsi_at_end >= RSI_DEEP_OVERBOUGHT)

    if len(extreme_positions) < 2:
        return {
            "ok": False, "divergence": False, "crossed_50": False,
            "single_extreme": len(extreme_positions) == 1,
            "rsi_at_end": rsi_at_end, "deep_extreme": deep_extreme,
        }

    first_i, last_i = extreme_positions[0], extreme_positions[-1]
    between_rsi = window_rsi.iloc[first_i:last_i + 1]
    crossed_50 = (between_rsi < RSI_MID).any() and (between_rsi > RSI_MID).any()

    price_between = window_close.iloc[first_i:last_i + 1]
    if direction == "bullish":
        price_extended = price_between.iloc[-1] < price_between.iloc[0]
        rsi_weaker = window_rsi.iloc[last_i] > window_rsi.iloc[first_i]
    else:
        price_extended = price_between.iloc[-1] > price_between.iloc[0]
        rsi_weaker = window_rsi.iloc[last_i] < window_rsi.iloc[first_i]
    divergence = price_extended and rsi_weaker

    return {
        "ok": True, "divergence": divergence, "crossed_50": crossed_50,
        "single_extreme": False, "rsi_at_end": rsi_at_end, "deep_extreme": deep_extreme,
    }


def check_type1_bamm(df: pd.DataFrame, direction: str, b_idx: int, d_idx: int,
                      period: int = 14) -> RsiConfirmation:
    """
    Type-I: the pattern's OWN B-to-D swing is the divergence window.
    RSI extreme near B, crosses back through 50 on the BC leg, comes
    back down/up into D. Confirms at the same moment the pattern
    completes -- nothing has been "missed" waiting for this.

    `df` should be the PROXIMATE timeframe's dataframe (Vol.3 Ch.8),
    with b_idx/d_idx expressed as integer positions in that same
    dataframe (the caller is responsible for mapping the Primary
    pattern's B/D timestamps onto Proximate-timeframe bar indices).
    """
    rsi = wilder_rsi(df["Close"], period=period)
    result = _divergence_in_window(rsi, df["Close"], direction, b_idx, d_idx)
    regime = trend_regime_from_rsi(rsi)

    if not result.get("ok"):
        if result.get("single_extreme"):
            return RsiConfirmation(True, "Type-I: single RSI extreme near B/D, no divergence shape",
                                    regime, False, result.get("deep_extreme", False),
                                    result.get("rsi_at_end", float("nan")), "type1")
        return RsiConfirmation(False, result.get("reason", "no RSI extreme found in B-D window"),
                                regime, False, False, float("nan"), "")

    confirmed = result["divergence"] and result["crossed_50"]
    if confirmed:
        reason = "Type-I BAMM confirmed: B-to-D divergence with a 50-level crossing on the BC leg"
    elif result["divergence"]:
        reason = "RSI divergence present across B-D but no 50-crossing on the BC leg -- not valid Type-I"
    else:
        reason = "two RSI extremes near B/D but no genuine divergence"

    return RsiConfirmation(confirmed, reason, regime, result["divergence"],
                            result["deep_extreme"], result["rsi_at_end"], "type1" if confirmed else "")


def check_type2_bamm(df: pd.DataFrame, direction: str, prz_start_idx: int,
                      prz_end_idx: int, period: int = 14) -> RsiConfirmation:
    """
    Type-II: a genuine RETEST of the same PRZ after an initial
    reaction didn't fully commit -- two extreme tests close together
    near the zone itself, not spread across the whole pattern. This
    costs real time/price by definition (the retest has to happen),
    which is why Type-II trades carry a bigger objective and more
    discretionary (trendline-based) management in the source material.

    `df` should be the PROXIMATE timeframe's dataframe, same as
    check_type1_bamm.
    """
    rsi = wilder_rsi(df["Close"], period=period)
    result = _divergence_in_window(rsi, df["Close"], direction,
                                    max(0, prz_start_idx - 5), prz_end_idx)
    regime = trend_regime_from_rsi(rsi)

    if not result.get("ok"):
        return RsiConfirmation(False, result.get("reason", "no retest of the PRZ found"),
                                regime, False, result.get("deep_extreme", False),
                                result.get("rsi_at_end", float("nan")), "")

    confirmed = result["divergence"] and result["crossed_50"]
    reason = ("Type-II BAMM confirmed: PRZ retest with divergence and a 50-crossing between tests"
              if confirmed else
              "PRZ retest found but divergence/50-crossing conditions not met -- not valid Type-II")

    return RsiConfirmation(confirmed, reason, regime, result["divergence"],
                            result["deep_extreme"], result["rsi_at_end"], "type2" if confirmed else "")


def check_rsi_bamm(proximate_df: pd.DataFrame, direction: str, b_idx: int, d_idx: int,
                    period: int = 14) -> RsiConfirmation:
    """
    Convenience dispatcher: tries Type-I first (the pattern's own B-D
    swing); if that doesn't confirm, falls back to Type-II (a retest
    right around the PRZ/D area). Either confirming is sufficient --
    Type-I is the cheaper, faster-confirming case; Type-II is the
    fallback for patterns that needed a second test to confirm.

    All indices are positions in `proximate_df` -- the caller maps the
    Primary pattern's B/D pivot timestamps onto Proximate-timeframe bar
    positions before calling this (see scan.py).
    """
    type1 = check_type1_bamm(proximate_df, direction, b_idx, d_idx, period)
    if type1.confirmed:
        return type1

    prz_start = max(b_idx, d_idx - 10)
    type2 = check_type2_bamm(proximate_df, direction, prz_start, d_idx, period)
    if type2.confirmed:
        return type2

    # Neither confirmed -- return whichever has more informative detail
    return type1 if type1.reason != "no RSI extreme found in B-D window" else type2

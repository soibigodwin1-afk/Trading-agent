"""
Classic chart patterns, built from Gartley's "Profits in the Stock
Market" (1935) -- his seven cardinal reversal types, plus triangles
and the decisive-penetration breakout filter.

Every detector here works off the same zigzag pivots as the harmonic
detector (patterns.Pivot / patterns.zigzag_pivots), so both families
run over identical swing structure.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd

from patterns import Pivot


@dataclass
class ChartMatch:
    name: str
    direction: str          # "bullish" or "bearish"
    pivots: List[Pivot]
    target: Optional[float]
    stop_loss: Optional[float]
    completion_time: object
    detail: str = ""


def _close_enough(a: float, b: float, tol_pct: float) -> bool:
    ref = (a + b) / 2
    return ref != 0 and abs(a - b) / ref * 100 <= tol_pct


# ------------------------------------------------------------------ #
# Double Top / Bottom
# ------------------------------------------------------------------ #

def find_double_top_bottom(pivots: List[Pivot], tol_pct: float = 1.5) -> List[ChartMatch]:
    matches = []
    for i in range(len(pivots) - 2):
        p1, p2, p3 = pivots[i:i + 3]
        if p1.kind == "H" and p2.kind == "L" and p3.kind == "H" and _close_enough(p1.price, p3.price, tol_pct):
            target = p2.price - (p1.price - p2.price)  # measured move from neckline
            matches.append(ChartMatch(
                "Double Top", "bearish", [p1, p2, p3], target, max(p1.price, p3.price),
                p3.timestamp, f"tops {p1.price:.4f}/{p3.price:.4f}, neckline {p2.price:.4f}",
            ))
        if p1.kind == "L" and p2.kind == "H" and p3.kind == "L" and _close_enough(p1.price, p3.price, tol_pct):
            target = p2.price + (p2.price - p1.price)
            matches.append(ChartMatch(
                "Double Bottom", "bullish", [p1, p2, p3], target, min(p1.price, p3.price),
                p3.timestamp, f"bottoms {p1.price:.4f}/{p3.price:.4f}, neckline {p2.price:.4f}",
            ))
    return matches


# ------------------------------------------------------------------ #
# Head & Shoulders / Inverse -- with Gartley's measuring rule:
# decline (or advance) projected = 2x the SMALLER of the two
# head-to-armpit amplitudes (Ch. VIII, p.223).
# ------------------------------------------------------------------ #

def find_head_shoulders(pivots: List[Pivot], tol_pct: float = 3.0) -> List[ChartMatch]:
    matches = []
    for i in range(len(pivots) - 4):
        p1, p2, p3, p4, p5 = pivots[i:i + 5]
        kinds = [p.kind for p in (p1, p2, p3, p4, p5)]

        if kinds == ["H", "L", "H", "L", "H"]:
            if p3.price > p1.price and p3.price > p5.price \
                    and _close_enough(p1.price, p5.price, tol_pct * 2) \
                    and _close_enough(p2.price, p4.price, tol_pct * 2):
                amp1, amp2 = p3.price - p2.price, p3.price - p4.price
                neckline = (p2.price + p4.price) / 2
                target = neckline - 2 * min(amp1, amp2)
                matches.append(ChartMatch(
                    "Head & Shoulders", "bearish", [p1, p2, p3, p4, p5],
                    target, p3.price, p5.timestamp,
                    f"head {p3.price:.4f}, neckline ~{neckline:.4f}, target {target:.4f}",
                ))

        if kinds == ["L", "H", "L", "H", "L"]:
            if p3.price < p1.price and p3.price < p5.price \
                    and _close_enough(p1.price, p5.price, tol_pct * 2) \
                    and _close_enough(p2.price, p4.price, tol_pct * 2):
                amp1, amp2 = p2.price - p3.price, p4.price - p3.price
                neckline = (p2.price + p4.price) / 2
                target = neckline + 2 * min(amp1, amp2)
                matches.append(ChartMatch(
                    "Inverse Head & Shoulders", "bullish", [p1, p2, p3, p4, p5],
                    target, p3.price, p5.timestamp,
                    f"head {p3.price:.4f}, neckline ~{neckline:.4f}, target {target:.4f}",
                ))
    return matches


# ------------------------------------------------------------------ #
# Broadening Top / Bottom -- 5-point widening structure (Ch. VIII):
# reversals 3 & 5 higher than 1 (top) / lower than 1 (bottom), and
# reversal 4 lower than 2 (top) / higher than 2 (bottom).
# ------------------------------------------------------------------ #

def find_broadening(pivots: List[Pivot]) -> List[ChartMatch]:
    matches = []
    for i in range(len(pivots) - 4):
        p1, p2, p3, p4, p5 = pivots[i:i + 5]
        kinds = [p.kind for p in (p1, p2, p3, p4, p5)]

        if kinds == ["H", "L", "H", "L", "H"]:
            if p3.price > p1.price and p5.price > p3.price and p4.price < p2.price:
                matches.append(ChartMatch(
                    "Broadening Top", "bearish", [p1, p2, p3, p4, p5],
                    target=None, stop_loss=p5.price, completion_time=p5.timestamp,
                    detail="widening highs and lows -- no fixed measuring rule, trade the breakdown",
                ))

        if kinds == ["L", "H", "L", "H", "L"]:
            if p3.price < p1.price and p5.price < p3.price and p4.price > p2.price:
                matches.append(ChartMatch(
                    "Broadening Bottom", "bullish", [p1, p2, p3, p4, p5],
                    target=None, stop_loss=p5.price, completion_time=p5.timestamp,
                    detail="widening highs and lows -- no fixed measuring rule, trade the breakout",
                ))
    return matches


# ------------------------------------------------------------------ #
# Rounding Top / Bottom -- gradual curvature, detected via quadratic
# fit over a window of closing prices rather than discrete pivots.
# ------------------------------------------------------------------ #

def find_rounding(df: pd.DataFrame, window: int = 20, min_r2: float = 0.7) -> List[ChartMatch]:
    matches = []
    closes = df["Close"].values
    if len(closes) < window:
        return matches

    seg = closes[-window:]
    x = np.arange(window)
    coeffs = np.polyfit(x, seg, 2)
    fitted = np.polyval(coeffs, x)
    ss_res = np.sum((seg - fitted) ** 2)
    ss_tot = np.sum((seg - seg.mean()) ** 2) or 1e-9
    r2 = 1 - ss_res / ss_tot

    if r2 < min_r2:
        return matches

    a = coeffs[0]  # curvature: a>0 = rounding bottom (U), a<0 = rounding top (dome)
    end_idx = len(df) - 1
    end_time = df.index[end_idx]
    dummy_pivot_start = Pivot(end_idx - window, df.index[end_idx - window], seg[0], "L" if a > 0 else "H")
    dummy_pivot_end = Pivot(end_idx, end_time, seg[-1], "L" if a > 0 else "H")

    if a > 0:
        matches.append(ChartMatch(
            "Rounding Bottom", "bullish", [dummy_pivot_start, dummy_pivot_end],
            target=None, stop_loss=float(seg.min()), completion_time=end_time,
            detail=f"quadratic fit R^2={r2:.2f}, curving upward",
        ))
    else:
        matches.append(ChartMatch(
            "Rounding Top", "bearish", [dummy_pivot_start, dummy_pivot_end],
            target=None, stop_loss=float(seg.max()), completion_time=end_time,
            detail=f"quadratic fit R^2={r2:.2f}, curving downward",
        ))
    return matches


# ------------------------------------------------------------------ #
# Triangles -- trendline slope through recent swing highs vs lows.
# Measured-move target: the "pole" leading into the triangle projected
# from the breakout point (Ch. IX, p.228 "third side" rule).
# ------------------------------------------------------------------ #

def _fit_slope(xs: List[int], ys: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return float(np.polyfit(xs, ys, 1)[0])


def find_triangles(pivots: List[Pivot], flat_slope_frac: float = 0.15) -> List[ChartMatch]:
    matches = []
    if len(pivots) < 6:
        return matches

    recent = pivots[-6:]
    highs = [p for p in recent if p.kind == "H"]
    lows = [p for p in recent if p.kind == "L"]
    if len(highs) < 2 or len(lows) < 2:
        return matches

    high_slope = _fit_slope([p.index for p in highs], [p.price for p in highs])
    low_slope = _fit_slope([p.index for p in lows], [p.price for p in lows])

    avg_price = np.mean([p.price for p in recent])
    flat_threshold = avg_price * flat_slope_frac / max(1, (recent[-1].index - recent[0].index))

    is_flat_high = abs(high_slope) <= flat_threshold
    is_flat_low = abs(low_slope) <= flat_threshold

    # The "pole": the move leading into the very first pivot of this window.
    pole_start = recent[0]
    pole_size = abs(pole_start.price - pivots[max(0, pivots.index(pole_start) - 1)].price) \
        if pivots.index(pole_start) > 0 else None

    last_pivot = recent[-1]
    kind, direction = None, None
    if is_flat_high and low_slope > 0:
        kind, direction = "Ascending Triangle", "bullish"
    elif is_flat_low and high_slope < 0:
        kind, direction = "Descending Triangle", "bearish"
    elif high_slope < 0 and low_slope > 0:
        kind = "Symmetrical Triangle"
        direction = "bullish" if last_pivot.kind == "L" else "bearish"

    if kind:
        target = None
        if pole_size:
            target = (last_pivot.price + pole_size if direction == "bullish"
                      else last_pivot.price - pole_size)
        matches.append(ChartMatch(
            kind, direction, recent, target, None, last_pivot.timestamp,
            f"high_slope={high_slope:.4g} low_slope={low_slope:.4g} pole={pole_size}",
        ))
    return matches


# ------------------------------------------------------------------ #
# Decisive-penetration breakout confirmation (Ch. VIII, p.222).
# ------------------------------------------------------------------ #

def decisive_penetration(df: pd.DataFrame, level: float, direction: str,
                          min_pct: float = 1.0, window: int = 3,
                          require_volume: bool = True) -> bool:
    """
    Checks the last `window` bars for a "decisive" breakout of `level`:
    exceeds it by at least min_pct%, within a short window, ideally with
    expanding volume. Volume check is skipped gracefully if the symbol
    has no real volume data (typical for forex).
    """
    recent = df.tail(window)
    if recent.empty:
        return False

    if direction == "bullish":
        breached = recent["Close"].max() >= level * (1 + min_pct / 100)
    else:
        breached = recent["Close"].min() <= level * (1 - min_pct / 100)

    if not breached:
        return False

    if require_volume and "Volume" in df.columns and df["Volume"].tail(20).sum() > 0:
        avg_vol = df["Volume"].tail(20).mean()
        recent_vol = recent["Volume"].mean()
        return recent_vol >= avg_vol  # expanding volume on the breakout

    return True  # no usable volume data -- pass on price alone


# ------------------------------------------------------------------ #
# Selling Climax -- volume-based capitulation signature (Ch. XIV).
# Independent of both pattern families; requires real Volume data.
# ------------------------------------------------------------------ #

def find_selling_climax(df: pd.DataFrame, lookback: int = 20) -> List[ChartMatch]:
    matches = []
    if "Volume" not in df.columns or df["Volume"].tail(lookback).sum() == 0:
        return matches
    if len(df) < lookback + 2:
        return matches

    last = df.iloc[-1]
    prev = df.iloc[-2]
    avg_vol = df["Volume"].tail(lookback).mean()

    gapped_down = last["Open"] < prev["Close"]
    high_volume = last["Volume"] > avg_vol * 1.5
    day_range = last["High"] - last["Low"]
    strong_reversal = day_range > 0 and (last["Close"] - last["Low"]) / day_range >= 0.7

    if gapped_down and high_volume and strong_reversal:
        pivot = Pivot(len(df) - 1, df.index[-1], last["Low"], "L")
        matches.append(ChartMatch(
            "Selling Climax", "bullish", [pivot], target=None, stop_loss=last["Low"],
            completion_time=df.index[-1],
            detail=f"gap down + volume {last['Volume']:.0f} vs avg {avg_vol:.0f}, "
                   f"closed {((last['Close']-last['Low'])/day_range*100):.0f}% off the low",
        ))
    return matches


def find_all_chart_patterns(df: pd.DataFrame, pivots: List[Pivot],
                             tol_pct: float = 1.5, min_r2: float = 0.7,
                             flat_slope_frac: float = 0.15) -> List[ChartMatch]:
    return (
        find_double_top_bottom(pivots, tol_pct)
        + find_head_shoulders(pivots, tol_pct * 2)
        + find_broadening(pivots)
        + find_rounding(df, min_r2=min_r2)
        + find_triangles(pivots, flat_slope_frac=flat_slope_frac)
        + find_selling_climax(df)
    )


# Pattern names built on geometric heuristics (quadratic curve fit,
# trendline slope) rather than a precisely sourced ratio/measuring
# rule -- flagged distinctly in alert captions so they aren't weighted
# the same as the sourced patterns, and are the two whose own
# detection thresholds (not just confluence) are eligible for
# calibration proposals in report.py.
HEURISTIC_PATTERN_NAMES = {
    "Rounding Top", "Rounding Bottom",
    "Ascending Triangle", "Descending Triangle", "Symmetrical Triangle",
}

"""
Harmonic pattern detection, built from Carney's standardized spec table
(Harmonic Trading Vol. 3, Ch. 2 "Standardized Harmonic Pattern
Identification") rather than approximated ratio ranges.

Every XABCD pattern is defined by:
  - an exact B-point ratio (retracement of XA), with its own tolerance
  - a BC projection range
  - an exact D-point ratio (retracement/extension of XA) -- the PRZ anchor
  - a stop-loss multiple of XA, measured beyond X

5-0 and Shark have a different skeleton (0-X-A-B-C, completing at C,
not D) and are handled separately from the XABCD family.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
import pandas as pd

# The closed set of true Fibonacci-derived ratios (Vol. 1, Ch. 2).
# Nothing outside this list is ever a valid harmonic number.
FIB_RATIOS = [0.382, 0.5, 0.618, 0.707, 0.786, 0.886, 1.0, 1.13, 1.27,
              1.41, 1.618, 2.0, 2.24, 2.618, 3.14, 3.618]


@dataclass
class Pivot:
    index: int
    timestamp: object
    price: float
    kind: str  # "H" or "L"


@dataclass
class HarmonicMatch:
    name: str
    direction: str          # "bullish" or "bearish"
    pivots: List[Pivot]     # [X, A, B, C, D] (or [0, X, A, B, C] for Shark)
    prz_low: float
    prz_high: float
    converging_ratios: Dict[str, float]   # label -> computed ratio value
    confluence_score: int                  # how many ratios converge
    prz_width_pct: float                   # PRZ width as % of X-D range
    stop_loss: float
    completion_time: object
    detail: str = ""


# ------------------------------------------------------------------ #
# ZigZag pivot extraction
# ------------------------------------------------------------------ #

def zigzag_pivots(df: pd.DataFrame, threshold_pct: float) -> List[Pivot]:
    if df.empty or len(df) < 3:
        return []

    threshold = threshold_pct / 100.0
    highs = df["High"].values
    lows = df["Low"].values
    idx = df.index

    pivots: List[Pivot] = []
    direction: Optional[str] = None
    extreme_high, extreme_high_i = highs[0], 0
    extreme_low, extreme_low_i = lows[0], 0

    for i in range(1, len(df)):
        if highs[i] > extreme_high:
            extreme_high, extreme_high_i = highs[i], i
        if lows[i] < extreme_low:
            extreme_low, extreme_low_i = lows[i], i

        if direction is None:
            if extreme_high_i < i and (extreme_high - lows[i]) / extreme_high >= threshold:
                pivots.append(Pivot(extreme_high_i, idx[extreme_high_i], extreme_high, "H"))
                direction = "down"
                extreme_low, extreme_low_i = lows[i], i
            elif extreme_low_i < i and (highs[i] - extreme_low) / extreme_low >= threshold:
                pivots.append(Pivot(extreme_low_i, idx[extreme_low_i], extreme_low, "L"))
                direction = "up"
                extreme_high, extreme_high_i = highs[i], i

        elif direction == "up":
            if (extreme_high - lows[i]) / extreme_high >= threshold:
                pivots.append(Pivot(extreme_high_i, idx[extreme_high_i], extreme_high, "H"))
                direction = "down"
                extreme_low, extreme_low_i = lows[i], i

        elif direction == "down":
            if (highs[i] - extreme_low) / extreme_low >= threshold:
                pivots.append(Pivot(extreme_low_i, idx[extreme_low_i], extreme_low, "L"))
                direction = "up"
                extreme_high, extreme_high_i = highs[i], i

    cleaned: List[Pivot] = []
    for p in pivots:
        if cleaned and cleaned[-1].kind == p.kind:
            if (p.kind == "H" and p.price > cleaned[-1].price) or (
                p.kind == "L" and p.price < cleaned[-1].price
            ):
                cleaned[-1] = p
        else:
            cleaned.append(p)
    return cleaned


# ------------------------------------------------------------------ #
# Standardized XABCD pattern specs (Vol. 3, Ch. 2)
# tol is a fraction (0.03 = +/-3%). stop_mult is the XA-multiple
# beyond X where the stop sits.
# ------------------------------------------------------------------ #
XABCD_SPECS = {
    "Gartley":        {"b": 0.618, "b_tol": 0.03, "bc": (1.13, 1.618),
                        "d": 0.786, "d_tol": 0.03, "stop_mult": 1.0},
    "Deep Gartley":   {"b": 0.618, "b_tol": 0.03, "bc": (1.618, 2.618),
                        "d": 0.886, "d_tol": 0.03, "stop_mult": 1.13},
    "Bat":            {"b": (0.382, 0.50), "b_tol": 0.05, "bc": (1.618, 2.618),
                        "d": 0.886, "d_tol": 0.05, "stop_mult": 1.13},
    "Alternate Bat":  {"b": 0.382, "b_tol": 0.03, "bc": (2.0, 3.618),
                        "d": 1.13, "d_tol": 0.05, "stop_mult": 1.27},
    "Butterfly":      {"b": 0.786, "b_tol": 0.03, "bc": (1.618, 2.24),
                        "d": 1.27, "d_tol": 0.03, "stop_mult": 1.414},
    "Crab":           {"b": (0.382, 0.618), "b_tol": 0.05, "bc": (2.618, 3.618),
                        "d": 1.618, "d_tol": 0.05, "stop_mult": 2.0},
    "Deep Crab":      {"b": 0.886, "b_tol": 0.05, "bc": (2.0, 3.618),
                        "d": 1.618, "d_tol": 0.05, "stop_mult": 2.0},
}

# C point is always 0.382-0.886 of AB across every XABCD pattern --
# it doesn't discriminate between patterns, just a sanity range.
C_RANGE = (0.382, 0.886)


def _pct_diff(value: float, target: float) -> float:
    if target == 0:
        return float("inf")
    return abs(value - target) / target


def _b_matches(b_ratio: float, spec: dict) -> bool:
    target = spec["b"]
    tol = spec["b_tol"]
    if isinstance(target, tuple):
        lo, hi = target
        return (lo * (1 - tol)) <= b_ratio <= (hi * (1 + tol))
    return _pct_diff(b_ratio, target) <= tol


def find_harmonic_patterns(pivots: List[Pivot], min_bars: int = 30) -> List[HarmonicMatch]:
    matches: List[HarmonicMatch] = []
    if len(pivots) < 5:
        return matches

    for i in range(len(pivots) - 4):
        X, A, B, C, D = pivots[i:i + 5]
        kinds = [p.kind for p in (X, A, B, C, D)]
        if kinds not in (["H", "L", "H", "L", "H"], ["L", "H", "L", "H", "L"]):
            continue

        # Statistical validity floor (Vol. 3, Ch. 8): reject formations
        # spanning fewer than ~30 bars.
        if (D.index - X.index) < min_bars:
            continue

        direction = "bearish" if X.kind == "H" else "bullish"

        xa = abs(X.price - A.price)
        ab = abs(A.price - B.price)
        bc = abs(B.price - C.price)
        cd = abs(C.price - D.price)
        if xa == 0 or ab == 0 or bc == 0:
            continue

        b_ratio = ab / xa
        c_ratio = bc / ab
        if not (C_RANGE[0] * 0.95 <= c_ratio <= C_RANGE[1] * 1.05):
            continue

        bc_ratio = cd / bc
        # D ratio is how far price has retraced/extended back across the
        # XA leg, measured FROM A (not from X) -- e.g. a 0.786 Gartley D
        # means D sits 78.6% of the way back from A toward X.
        d_ratio = abs(A.price - D.price) / xa

        for name, spec in XABCD_SPECS.items():
            if not _b_matches(b_ratio, spec):
                continue
            bc_lo, bc_hi = spec["bc"]
            if not (bc_lo * 0.95 <= bc_ratio <= bc_hi * 1.05):
                continue
            if _pct_diff(d_ratio, spec["d"]) > spec["d_tol"]:
                continue

            # --- Confluence: how many independent numbers converge ---
            converging = {
                "B (XA retr.)": round(b_ratio, 3),
                "BC projection": round(bc_ratio, 3),
                "D (XA)": round(d_ratio, 3),
            }
            ab_cd_ratio = cd / ab
            if (_pct_diff(ab_cd_ratio, 1.0) <= 0.10 or _pct_diff(ab_cd_ratio, 1.27) <= 0.08
                    or _pct_diff(ab_cd_ratio, 1.618) <= 0.08):
                converging["AB=CD"] = round(ab_cd_ratio, 3)
            confluence_score = len(converging)

            prz_center = D.price
            d_tol_price = xa * spec["d"] * spec["d_tol"]
            prz_low, prz_high = prz_center - d_tol_price, prz_center + d_tol_price
            if direction == "bullish":
                stop_loss = X.price - xa * (spec["stop_mult"] - 1.0)
            else:
                stop_loss = X.price + xa * (spec["stop_mult"] - 1.0)

            total_range = abs(X.price - D.price) or 1e-9
            prz_width_pct = abs(prz_high - prz_low) / total_range * 100

            matches.append(HarmonicMatch(
                name=name, direction=direction, pivots=[X, A, B, C, D],
                prz_low=min(prz_low, prz_high), prz_high=max(prz_low, prz_high),
                converging_ratios=converging, confluence_score=confluence_score,
                prz_width_pct=prz_width_pct, stop_loss=stop_loss,
                completion_time=D.timestamp,
                detail=f"B={b_ratio:.3f} BC={bc_ratio:.3f} D={d_ratio:.3f}",
            ))

    return matches


# ------------------------------------------------------------------ #
# 5-0 and Shark: 0-X-A-B-C skeleton, completing at C, not D.
# These do NOT share the XABCD ratio table above.
# ------------------------------------------------------------------ #

def find_5_0_and_shark(pivots: List[Pivot], min_bars: int = 30) -> List[HarmonicMatch]:
    matches: List[HarmonicMatch] = []
    if len(pivots) < 5:
        return matches

    for i in range(len(pivots) - 4):
        O, X, A, B, C = pivots[i:i + 5]
        kinds = [p.kind for p in (O, X, A, B, C)]
        if kinds not in (["H", "L", "H", "L", "H"], ["L", "H", "L", "H", "L"]):
            continue
        if (C.index - O.index) < min_bars:
            continue

        direction = "bearish" if O.kind == "H" else "bullish"

        ox = abs(O.price - X.price)
        xa = abs(X.price - A.price)
        ab = abs(A.price - B.price)
        if ox == 0 or xa == 0 or ab == 0:
            continue

        # "Extreme Harmonic Impulse Wave": B extends 1.618-2.24x beyond
        # the OX-XA structure -- the precursor to both patterns.
        impulse_ratio = ab / xa
        if not (1.5 <= impulse_ratio <= 2.4):
            continue

        c_retrace_of_0x = abs(O.price - C.price) / ox if ox else 0

        # Shark: completes 0.886-1.13 of the 0X leg.
        if 0.85 <= c_retrace_of_0x <= 1.18:
            matches.append(HarmonicMatch(
                name="Shark", direction=direction, pivots=[O, X, A, B, C],
                prz_low=min(O.price, C.price), prz_high=max(O.price, C.price),
                converging_ratios={"C (0X retr.)": round(c_retrace_of_0x, 3),
                                    "Impulse (AB/XA)": round(impulse_ratio, 3)},
                confluence_score=2, prz_width_pct=5.0,
                stop_loss=(C.price + (C.price - O.price) * 0.15 if direction == "bearish"
                           else C.price - (O.price - C.price) * 0.15),
                completion_time=C.timestamp,
                detail=f"C/0X={c_retrace_of_0x:.3f} impulse={impulse_ratio:.3f} "
                       f"(target: lesser of 50% retrace or reciprocal AB=CD)",
            ))

    return matches


def find_all_harmonics(df: pd.DataFrame, zigzag_threshold_pct: float,
                        min_bars: int = 30) -> List[HarmonicMatch]:
    pivots = zigzag_pivots(df, zigzag_threshold_pct)
    return find_harmonic_patterns(pivots, min_bars) + find_5_0_and_shark(pivots, min_bars)

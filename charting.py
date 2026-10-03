"""
Renders a candlestick chart as a full trade plan, not just a shape:
- the pattern's pivot lines
- the PRZ as a shaded zone with converging ratios labeled
- the Terminal Price Bar highlighted distinctly
- entry / stop / target drawn as labeled horizontal lines
Styled so risk:reward is readable at a glance on a small phone preview.
"""

import mplfinance as mpf
import matplotlib.pyplot as plt
import pandas as pd

HARMONIC_LABELS = ["X", "A", "B", "C", "D"]
SHARK_LABELS = ["0", "X", "A", "B", "C"]

# Gartley's own charts were semi-logarithmic (ratio scale), not linear
# (Ch. II) -- on a log axis, equal % moves occupy equal visual space
# regardless of price level, which matches what Fibonacci ratios
# actually measure. Default on for crypto (wide historical price
# ranges) or any symbol whose lookback window itself spans a wide
# range, per the same rationale.
LOG_SCALE_RANGE_RATIO_THRESHOLD = 1.5  # high/low > 1.5x triggers log scale


def should_use_log_scale(df: pd.DataFrame, symbol: str) -> bool:
    if symbol.lower().startswith("cry") or symbol.lower().startswith("cry_"):
        return True
    if df.empty:
        return False
    lo = df["Low"].min()
    hi = df["High"].max()
    if lo <= 0:
        return False
    return (hi / lo) >= LOG_SCALE_RANGE_RATIO_THRESHOLD


def _mpf_plot_kwargs(df: pd.DataFrame, symbol: str, title: str) -> dict:
    kwargs = dict(
        data=df, type="candle", style="yahoo", title=title, ylabel="Price",
        returnfig=True, figsize=(11, 6.5), tight_layout=True,
    )
    if should_use_log_scale(df, symbol):
        kwargs["yscale"] = "log"
    return kwargs


def render_harmonic_chart(df: pd.DataFrame, symbol: str, timeframe: str,
                           match, out_path: str, target: float = None) -> str:
    """`match` is a patterns.HarmonicMatch."""
    fig, axes = mpf.plot(**_mpf_plot_kwargs(
        df, symbol, f"{symbol}  {timeframe}  -  {match.name} ({match.direction})"))
    ax = axes[0]

    xs, ys = [], []
    for p in match.pivots:
        try:
            x = df.index.get_loc(p.timestamp)
        except KeyError:
            continue
        xs.append(x)
        ys.append(p.price)

    if xs:
        ax.plot(xs, ys, color="#1f77b4", linewidth=1.8, marker="o", markersize=5, zorder=5)
        labels = SHARK_LABELS if len(xs) == 5 and match.name == "Shark" else \
            (HARMONIC_LABELS if len(xs) == 5 else [str(i + 1) for i in range(len(xs))])
        for i, (x, y) in enumerate(zip(xs, ys)):
            offset = 10 if match.pivots[i].kind == "H" else -14
            ax.annotate(labels[i], (x, y), textcoords="offset points", xytext=(0, offset),
                        ha="center", fontsize=10, fontweight="bold", color="#1f77b4")

    # PRZ shaded zone with converging ratios labeled
    ax.axhspan(match.prz_low, match.prz_high, color="orange", alpha=0.15, zorder=1)
    ratio_label = "  ".join(f"{k}={v}" for k, v in match.converging_ratios.items())
    ax.text(len(df) - 1, match.prz_high, f"PRZ  {ratio_label}", fontsize=7.5,
            color="#b36b00", ha="right", va="bottom")

    # Terminal Price Bar highlight -- the last pivot's bar (D or C)
    t_bar_time = match.pivots[-1].timestamp
    try:
        t_idx = df.index.get_loc(t_bar_time)
        ax.axvspan(t_idx - 0.4, t_idx + 0.4, color="purple", alpha=0.15, zorder=0)
    except KeyError:
        pass

    # Entry (at PRZ edge nearest current direction) / stop / target lines
    entry = match.prz_high if match.direction == "bullish" else match.prz_low
    _draw_level(ax, entry, "Entry", "#1f77b4", len(df))
    _draw_level(ax, match.stop_loss, "Stop", "#d62728", len(df))
    if target is not None:
        _draw_level(ax, target, "Target", "#2ca02c", len(df))

    fig.text(0.01, 0.01, match.detail, fontsize=8, color="gray")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def render_chart_pattern_chart(df: pd.DataFrame, symbol: str, timeframe: str,
                                match, out_path: str) -> str:
    """`match` is a chart_patterns.ChartMatch."""
    fig, axes = mpf.plot(**_mpf_plot_kwargs(
        df, symbol, f"{symbol}  {timeframe}  -  {match.name} ({match.direction})"))
    ax = axes[0]

    xs, ys = [], []
    for p in match.pivots:
        try:
            x = df.index.get_loc(p.timestamp)
        except KeyError:
            continue
        xs.append(x)
        ys.append(p.price)
    if xs:
        ax.plot(xs, ys, color="#1f77b4", linewidth=1.8, marker="o", markersize=5, zorder=5)

    if match.stop_loss is not None:
        _draw_level(ax, match.stop_loss, "Stop", "#d62728", len(df))
    if match.target is not None:
        _draw_level(ax, match.target, "Target", "#2ca02c", len(df))

    fig.text(0.01, 0.01, match.detail, fontsize=8, color="gray")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def render_forming_pattern_chart(df: pd.DataFrame, symbol: str, timeframe: str,
                                  pivots, candidate_names: list,
                                  prz_low: float, prz_high: float, out_path: str) -> str:
    """Forming-pattern (Phase 2) chart: X-A-B-C plotted, projected PRZ
    shaded ahead of current price since D hasn't happened yet."""
    fig, axes = mpf.plot(**_mpf_plot_kwargs(
        df, symbol, f"{symbol}  {timeframe}  -  forming: {'/'.join(candidate_names)}"))
    ax = axes[0]
    xs, ys = [], []
    for p in pivots:
        try:
            x = df.index.get_loc(p.timestamp)
        except KeyError:
            continue
        xs.append(x)
        ys.append(p.price)
    if xs:
        ax.plot(xs, ys, color="#1f77b4", linewidth=1.8, marker="o", markersize=5, zorder=5)
        for i, (x, y) in enumerate(zip(xs, ys)):
            offset = 10 if pivots[i].kind == "H" else -14
            ax.annotate("XABC"[i] if i < 4 else str(i), (x, y), textcoords="offset points",
                        xytext=(0, offset), ha="center", fontsize=10, fontweight="bold",
                        color="#1f77b4")

    # Projected zone drawn ahead of current price (extends past the right edge)
    ax.axhspan(prz_low, prz_high, xmin=0.7, xmax=1.0, color="gold", alpha=0.2, zorder=1)
    ax.text(len(df) - 1, prz_high, "projected PRZ", fontsize=7.5, color="#8a6d00",
            ha="right", va="bottom")

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _draw_level(ax, price: float, label: str, color: str, n_bars: int):
    ax.axhline(price, color=color, linewidth=1.2, linestyle="--", zorder=4)
    ax.text(n_bars - 1, price, f" {label} {price:.4f}", fontsize=8, color=color,
            va="center", ha="left")

"""
Main entrypoint. Run with: python scan.py

For every timeframe group (intraday/swing/position) x every symbol in
that group's watchlist:
  1. fetch primary/proximate/distal OHLCV (closed candles only)
  2. skip if the regime/volatility filter says conditions are too dead
  3. read the distal-timeframe trend state (price structure + RSI-50
     regime) for alignment tagging -- a TAG, never a hard block
  4. detect harmonic patterns (XABCD family + Shark) and classic chart
     patterns on the primary timeframe
  5. apply the confluence-minimum bar (stricter on 5m)
  6. run RSI BAMM as a HARD GATE -- no valid indicator confirmation,
     no completion alert
  7. render the chart (PRZ zone, Terminal Price Bar, entry/stop/target)
     and send it to Telegram, opening an outcome watch
Before all of that, resolve_open_watches() checks every pattern
already being tracked against current price for target/stop/expiry
and sends a threaded reply with the outcome.
"""

import os
import tempfile
import traceback
from typing import Optional

import yaml

import data
import patterns
import chart_patterns as cp
import indicators as ind
import trend
import state as state_mod
import telegram_notify as tg
from charting import render_harmonic_chart, render_chart_pattern_chart


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _zigzag_threshold(cfg: dict, timeframe: str) -> float:
    table = cfg["zigzag_threshold_pct"]
    return table.get(timeframe, table.get("default", 2.0))


def _confluence_minimum(cfg: dict, st: dict, symbol: str, timeframe: str, pattern_name: str) -> int:
    bucket = f"{pattern_name}|{timeframe}|{symbol}"
    override = st.get("bucket_overrides", {}).get(bucket, {}).get("confluence_minimum")
    if override is not None:
        return override
    table = cfg["confluence_minimum"]
    return table.get(timeframe, table["default"])


def _heuristic_param(st: dict, symbol: str, timeframe: str, param_name: str, default: float) -> float:
    """Looks up a calibrated override for a heuristic-pattern parameter
    (Rounding's min_r2, Triangle's flat_slope_frac) -- these are keyed
    by (param_name, timeframe, symbol) rather than a specific pattern
    name, since the pattern name itself isn't known until after
    detection for these two. See report.py for how proposals for
    these get generated."""
    bucket = f"{param_name}|{timeframe}|{symbol}"
    override = st.get("bucket_overrides", {}).get(bucket, {}).get(param_name)
    return override if override is not None else default


def _map_timestamp_to_index(df, timestamp) -> Optional[int]:
    """Maps a timestamp from one timeframe's pivots onto the nearest
    bar index in a different timeframe's dataframe (used to translate
    the Primary pattern's B/D pivots onto Proximate-timeframe bars)."""
    if df.empty:
        return None
    try:
        pos = df.index.get_indexer([timestamp], method="nearest")[0]
        return int(pos) if pos >= 0 else None
    except Exception:
        return None


def _compute_target(match, ratio: float) -> float:
    """IPO-style target: measured back from D as a % of the XA leg,
    continuing in the pattern's own direction."""
    X, A = match.pivots[0], match.pivots[1]
    xa_size = abs(X.price - A.price)
    D = match.pivots[-1]
    sign = 1 if match.direction == "bullish" else -1
    return D.price + sign * ratio * xa_size


def _watch_key(symbol: str, timeframe: str, name: str, direction: str, completion) -> str:
    return f"{symbol}|{timeframe}|{name}|{direction}|{completion}"


# ------------------------------------------------------------------ #
# Outcome resolution -- checked BEFORE scanning for new patterns
# ------------------------------------------------------------------ #

def resolve_open_watches(cfg: dict, st: dict, errors: list) -> None:
    timeout_table = cfg["resolution_timeout_bars"]

    for watch_key, w in list(state_mod.due_watches(st)):
        try:
            symbol_cfg = _find_symbol_cfg(cfg, w["symbol"])
            if symbol_cfg is None:
                continue
            df = data.fetch_ohlcv(w["symbol"], w["timeframe"], symbol_cfg["source"])
            if df.empty:
                continue

            recent = df.tail(5)
            direction = w["direction"]
            target, stop = w["target"], w["stop"]

            hit_target = (recent["High"].max() >= target if direction == "bullish"
                          else recent["Low"].min() <= target)
            hit_stop = (recent["Low"].min() <= stop if direction == "bullish"
                        else recent["High"].max() >= stop)

            outcome = None
            exit_price = None
            if hit_stop and hit_target:
                # Ambiguous within this batch of bars -- check bar order
                # by whichever is closer to the start of the recent window.
                outcome, exit_price = ("stop", stop)  # conservative default
            elif hit_target:
                outcome, exit_price = "target", target
            elif hit_stop:
                outcome, exit_price = "stop", stop
            elif w.get("bars_open", 0) >= timeout_table.get(w["timeframe"], 40):
                outcome, exit_price = "expired", float(df["Close"].iloc[-1])

            if outcome:
                record = state_mod.resolve_watch(st, watch_key, outcome, exit_price)
                r_multiple = _r_multiple(record)
                text = _outcome_text(record, r_multiple)
                tg.send_outcome_reply(w["message_id"], text)

        except Exception as e:  # noqa: BLE001
            errors.append(f"resolve {watch_key}: {e}")
            traceback.print_exc()

    state_mod.tick_watches(st)


def _r_multiple(record: dict) -> float:
    entry, stop, exit_price = record["entry"], record["stop"], record["exit_price"]
    risk = abs(entry - stop) or 1e-9
    reward = (exit_price - entry) if record["direction"] == "bullish" else (entry - exit_price)
    return round(reward / risk, 2)


def _outcome_text(record: dict, r_multiple: float) -> str:
    if record["outcome"] == "target":
        return f"\u2705 Target hit -- {record['pattern']} on {record['symbol']} ({record['timeframe']}), ~{r_multiple}R"
    if record["outcome"] == "stop":
        return f"\u274c Stop hit -- {record['pattern']} on {record['symbol']} ({record['timeframe']}), ~{r_multiple}R"
    return f"\u26aa No clear outcome -- {record['pattern']} on {record['symbol']} ({record['timeframe']}) expired unresolved"


def _find_symbol_cfg(cfg: dict, symbol: str) -> dict:
    for wl in cfg["watchlists"].values():
        for s in wl:
            if s["symbol"] == symbol:
                return s
    return None


# ------------------------------------------------------------------ #
# Scanning for new patterns
# ------------------------------------------------------------------ #

def scan_group(group_name: str, group_cfg: dict, cfg: dict, st: dict,
                tmpdir: str, errors: list) -> int:
    alerts_sent = 0
    primary_tf = group_cfg["primary"]
    proximate_tf = group_cfg.get("proximate")
    distal_tf = group_cfg.get("distal")
    watchlist = cfg["watchlists"][group_cfg["watchlist"]]

    zz_threshold = _zigzag_threshold(cfg, primary_tf)
    min_bars = cfg["min_pattern_bars"]
    tol_pct = cfg["chart_pattern_tolerance_pct"]
    outcome_ratio = cfg["outcome_target_ratio"]
    vol_floor_pct = cfg["volatility_floor_percentile"]

    for sym_cfg in watchlist:
        symbol, source, label = sym_cfg["symbol"], sym_cfg["source"], sym_cfg["label"]
        try:
            primary_df = data.fetch_ohlcv(symbol, primary_tf, source)
            if primary_df.empty or len(primary_df) < min_bars + 5:
                continue

            if not trend.volatility_regime_ok(primary_df, floor_percentile=vol_floor_pct):
                continue

            htf_state = None
            if distal_tf:
                distal_df = data.fetch_ohlcv(symbol, distal_tf, source)
                if not distal_df.empty:
                    htf_state = trend.get_trend_state(distal_df, _zigzag_threshold(cfg, distal_tf))

            # Proximate timeframe: per Vol.3 Ch.8, indicator confirmation
            # (RSI BAMM) is read here, not on the Primary timeframe the
            # pattern itself was identified on.
            proximate_df = None
            if proximate_tf:
                proximate_df = data.fetch_ohlcv(symbol, proximate_tf, source)

            pivots = patterns.zigzag_pivots(primary_df, zz_threshold)

            # Detect both families up front so each can check the other
            # for cross-family confluence before any alert is sent.
            harmonics = patterns.find_all_harmonics(primary_df, zz_threshold, min_bars)
            min_r2 = _heuristic_param(st, symbol, primary_tf, "rounding_min_r2", 0.7)
            flat_slope_frac = _heuristic_param(st, symbol, primary_tf, "triangle_flat_slope_frac", 0.15)
            chart_matches = cp.find_all_chart_patterns(
                primary_df, pivots, tol_pct, min_r2=min_r2, flat_slope_frac=flat_slope_frac)

            # --- Harmonic patterns ---
            for m in harmonics:
                confluence_min = _confluence_minimum(cfg, st, symbol, primary_tf, m.name)
                if m.confluence_score < confluence_min:
                    continue

                key = _watch_key(symbol, primary_tf, m.name, m.direction, m.completion_time)
                if state_mod.already_alerted(st, key, str(m.completion_time)):
                    continue

                rsi_conf = _confirm_with_proximate(proximate_df, primary_df, m)
                if cfg["rsi"]["require_confirmation"] and not rsi_conf.confirmed:
                    continue  # hard gate -- no indicator confirmation, no alert

                alignment = "neutral"
                if htf_state:
                    alignment = trend.pattern_alignment_tag(m.direction, htf_state)

                cross_confluence = _cross_family_match(m.prz_low, m.prz_high, m.direction, chart_matches)

                target = _compute_target(m, outcome_ratio)
                chart_path = os.path.join(
                    tmpdir, f"{symbol}_{primary_tf}_{m.name.replace(' ', '')}.png")
                render_harmonic_chart(primary_df, label, primary_tf, m, chart_path, target=target)

                confluence_line = f"Confluence: {m.confluence_score} | PRZ width: {m.prz_width_pct:.1f}%"
                if cross_confluence:
                    confluence_line += f"\n\U0001F31F High confidence: also agrees with {cross_confluence} nearby"
                caption = (
                    f"<b>{m.name}</b> ({m.direction}) -- {label}, {primary_tf}\n"
                    f"{confluence_line}\n"
                    f"RSI ({rsi_conf.bamm_type or 'n/a'}, on {proximate_tf or primary_tf}): {rsi_conf.reason}\n"
                    f"Trend alignment: {alignment}\n"
                    f"Stop {m.stop_loss:.5f} | Target {target:.5f}\n"
                    f"{m.detail}"
                )
                msg_id = tg.send_photo(chart_path, caption)
                entry = m.prz_high if m.direction == "bullish" else m.prz_low
                state_mod.open_watch(st, key, symbol, primary_tf, m.name, m.direction,
                                      entry, m.stop_loss, target, msg_id,
                                      os.environ.get("TELEGRAM_CHAT_ID", ""),
                                      cfg["resolution_timeout_bars"].get(primary_tf, 40))
                state_mod.mark_alerted(st, key, str(m.completion_time))
                alerts_sent += 1

            # --- Classic chart patterns ---
            for m in chart_matches:
                key = _watch_key(symbol, primary_tf, m.name, m.direction, m.completion_time)
                if state_mod.already_alerted(st, key, str(m.completion_time)):
                    continue

                chart_path = os.path.join(
                    tmpdir, f"{symbol}_{primary_tf}_{m.name.replace(' ', '')}_chart.png")
                render_chart_pattern_chart(primary_df, label, primary_tf, m, chart_path)

                alignment = trend.pattern_alignment_tag(m.direction, htf_state) if htf_state else "neutral"
                experimental_tag = ("\u26a0\ufe0f experimental pattern type -- heuristic geometry, "
                                     "not a sourced ratio definition\n"
                                     if m.name in cp.HEURISTIC_PATTERN_NAMES else "")
                level = m.target if m.target is not None else m.stop_loss
                cross_confluence = ""
                if level is not None:
                    hm = _cross_family_match_harmonics(level, m.direction, harmonics)
                    if hm:
                        cross_confluence = f"\U0001F31F High confidence: also agrees with {hm} nearby\n"
                caption = (
                    f"<b>{m.name}</b> ({m.direction}) -- {label}, {primary_tf}\n"
                    f"{experimental_tag}{cross_confluence}"
                    f"Trend alignment: {alignment}\n"
                    + (f"Target {m.target:.5f}\n" if m.target is not None else "")
                    + f"{m.detail}"
                )
                msg_id = tg.send_photo(chart_path, caption)

                if m.target is not None and m.stop_loss is not None:
                    state_mod.open_watch(st, key, symbol, primary_tf, m.name, m.direction,
                                          primary_df["Close"].iloc[-1], m.stop_loss, m.target,
                                          msg_id, os.environ.get("TELEGRAM_CHAT_ID", ""),
                                          cfg["resolution_timeout_bars"].get(primary_tf, 40))
                state_mod.mark_alerted(st, key, str(m.completion_time))
                alerts_sent += 1

        except Exception as e:  # noqa: BLE001
            errors.append(f"{group_name}/{symbol}: {e}")
            traceback.print_exc()

    return alerts_sent


def _cross_family_match(prz_low: float, prz_high: float, direction: str,
                         chart_matches: list, tolerance_pct: float = 1.0) -> Optional[str]:
    """Checks whether any classic chart pattern's own key level (target,
    or stop if no target) falls within `tolerance_pct`% of a harmonic's
    PRZ and shares its direction -- independent confirmation from a
    different pattern family is stronger evidence than either alone."""
    mid = (prz_low + prz_high) / 2
    band = mid * tolerance_pct / 100
    lo, hi = prz_low - band, prz_high + band
    for cm in chart_matches:
        if cm.direction != direction:
            continue
        level = cm.target if cm.target is not None else cm.stop_loss
        if level is not None and lo <= level <= hi:
            return cm.name
    return None


def _cross_family_match_harmonics(level: float, direction: str, harmonics: list,
                                   tolerance_pct: float = 1.0) -> Optional[str]:
    """Mirror of _cross_family_match, checked from the chart-pattern side."""
    for hm in harmonics:
        if hm.direction != direction:
            continue
        band = ((hm.prz_high - hm.prz_low) / 2 or hm.prz_high * tolerance_pct / 100)
        if (hm.prz_low - band) <= level <= (hm.prz_high + band):
            return hm.name
    return None


def _confirm_with_proximate(proximate_df, primary_df, m) -> "ind.RsiConfirmation":
    """Maps the pattern's B and D pivots (from the Primary timeframe)
    onto the Proximate timeframe's bar indices and runs RSI BAMM there,
    per Vol.3 Ch.8. Falls back to the Primary timeframe if no proximate
    data is available (e.g. the 'position' group has no shorter TF
    configured below 1D in this setup)."""
    if proximate_df is not None and not proximate_df.empty:
        b_ts = m.pivots[-4].timestamp if len(m.pivots) >= 4 else m.pivots[0].timestamp
        d_ts = m.pivots[-1].timestamp
        b_idx = _map_timestamp_to_index(proximate_df, b_ts)
        d_idx = _map_timestamp_to_index(proximate_df, d_ts)
        if b_idx is not None and d_idx is not None and d_idx > b_idx:
            return ind.check_rsi_bamm(proximate_df, m.direction, b_idx, d_idx)

    # Fallback: no proximate timeframe configured/available.
    b_idx = max(0, m.pivots[-4].index if len(m.pivots) >= 4 else m.pivots[0].index)
    d_idx = m.pivots[-1].index
    return ind.check_rsi_bamm(primary_df, m.direction, b_idx, d_idx)


def main():
    cfg = load_config()
    st = state_mod.load_state()
    errors = []

    resolve_open_watches(cfg, st, errors)

    alerts_sent = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        for group_name, group_cfg in cfg["timeframe_groups"].items():
            alerts_sent += scan_group(group_name, group_cfg, cfg, st, tmpdir, errors)

    state_mod.expire_stale_calibrations(st, cfg["reporting"]["calibration_expiry_days"])
    state_mod.save_state(st)

    print(f"Scan complete. Alerts sent: {alerts_sent}. Errors: {len(errors)}")
    for e in errors:
        print("  -", e)
    if len(errors) >= 5:
        try:
            tg.send_message(f"Trading agent scan had {len(errors)} errors this run. Check logs.")
        except Exception:
            pass


if __name__ == "__main__":
    main()

"""
Reporting and self-calibration. Run with: python report.py
Intended to run on its own daily schedule (separate from scan.py's
30-minute cadence) -- internally decides whether a weekly digest
and/or monthly analysis is actually due, and skips silently if not.

Weekly digest: plain tally of alerts/outcomes since the last one.
No analysis, no calibration -- just status.

Monthly analysis: aggregates resolved outcomes per (pattern, symbol,
timeframe) bucket. For any bucket with enough samples and a win rate
outside a healthy band, PROPOSES a calibration change (never applies
it automatically) via a Telegram message with Accept/Decline buttons.
Nothing changes until you respond -- see apply_calibration_responses().
"""

import uuid
from datetime import datetime, timedelta, timezone
from collections import defaultdict

import yaml

import state as state_mod
import telegram_notify as tg
from chart_patterns import HEURISTIC_PATTERN_NAMES

WEEKLY_INTERVAL_DAYS = 7
MONTHLY_INTERVAL_DAYS = 30

# Healthy win-rate band -- outside this, with enough samples, a
# calibration change gets proposed.
HEALTHY_WIN_RATE = (0.40, 0.65)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _due(last_iso: str, interval_days: int) -> bool:
    if not last_iso:
        return True
    last = datetime.fromisoformat(last_iso)
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - last >= timedelta(days=interval_days)


def _outcomes_since(st: dict, since_iso: str) -> list:
    outcomes = st.get("outcomes", [])
    if not since_iso:
        return outcomes
    since = datetime.fromisoformat(since_iso)
    if since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)
    result = []
    for rec in outcomes:
        closed = datetime.fromisoformat(rec["closed_at"])
        if closed.tzinfo is None:
            closed = closed.replace(tzinfo=timezone.utc)
        if closed >= since:
            result.append(rec)
    return result


# ------------------------------------------------------------------ #
# Weekly digest -- status only, no analysis
# ------------------------------------------------------------------ #

def weekly_digest(cfg: dict, st: dict) -> None:
    if not cfg["reporting"].get("weekly_digest", True):
        return
    if not _due(st.get("last_weekly_report"), WEEKLY_INTERVAL_DAYS):
        return

    new_outcomes = _outcomes_since(st, st.get("last_weekly_report"))
    open_count = len(st.get("watching", {}))

    if not new_outcomes and open_count == 0:
        st["last_weekly_report"] = datetime.now(timezone.utc).isoformat()
        return  # nothing new -- skip sending noise

    by_tf = defaultdict(lambda: {"target": 0, "stop": 0, "expired": 0})
    for rec in new_outcomes:
        by_tf[rec["timeframe"]][rec["outcome"]] += 1

    lines = ["<b>Weekly status</b>"]
    if new_outcomes:
        for tf, counts in by_tf.items():
            lines.append(f"{tf}: {counts['target']} hit target, {counts['stop']} hit stop, "
                         f"{counts['expired']} expired")
    else:
        lines.append("No patterns resolved this week.")
    lines.append(f"Currently tracking {open_count} open pattern(s).")

    tg.send_message("\n".join(lines))
    st["last_weekly_report"] = datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------ #
# Monthly analysis + calibration proposals
# ------------------------------------------------------------------ #

def _bucket_key(rec: dict) -> str:
    return f"{rec['pattern']}|{rec['timeframe']}|{rec['symbol']}"


def monthly_report(cfg: dict, st: dict) -> None:
    if not cfg["reporting"].get("monthly_analysis", True):
        return
    if not _due(st.get("last_monthly_report"), MONTHLY_INTERVAL_DAYS):
        return

    new_outcomes = _outcomes_since(st, st.get("last_monthly_report"))
    if not new_outcomes:
        st["last_monthly_report"] = datetime.now(timezone.utc).isoformat()
        return

    buckets = defaultdict(list)
    for rec in new_outcomes:
        buckets[_bucket_key(rec)].append(rec)

    lines = ["<b>Monthly analysis</b>"]
    min_sample = cfg["reporting"]["calibration_sample_min"]
    overrides = st.setdefault("bucket_overrides", {})

    for bucket, recs in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        decisive = [r for r in recs if r["outcome"] in ("target", "stop")]
        if not decisive:
            continue
        wins = sum(1 for r in decisive if r["outcome"] == "target")
        win_rate = wins / len(decisive)
        lines.append(f"{bucket}: {wins}/{len(decisive)} ({win_rate:.0%}) over {len(decisive)} samples")

        if len(decisive) < min_sample:
            continue  # not enough data to propose anything yet

        pattern, timeframe, symbol = bucket.split("|")
        proposal = None  # (param, old_val, new_val, reasoning, override_bucket)

        if pattern in HEURISTIC_PATTERN_NAMES:
            # These have no confluence_score at all -- calibrate their
            # OWN detection threshold instead.
            if pattern.startswith("Rounding"):
                param, default = "rounding_min_r2", 0.7
                param_bucket = f"{param}|{timeframe}|{symbol}"
                current = overrides.get(param_bucket, {}).get(param, default)
                if win_rate < HEALTHY_WIN_RATE[0]:
                    proposal = (param, current, round(min(current + 0.05, 0.95), 2), param_bucket,
                                f"win rate {win_rate:.0%} over {len(decisive)} samples -- requiring "
                                f"a tighter curve fit (higher R^2) to reduce false positives")
                elif win_rate > HEALTHY_WIN_RATE[1] and current > 0.5:
                    proposal = (param, current, round(max(current - 0.05, 0.5), 2), param_bucket,
                                f"win rate {win_rate:.0%} over {len(decisive)} samples -- this bucket "
                                f"may tolerate a looser curve-fit threshold")
            else:  # a Triangle variant
                param, default = "triangle_flat_slope_frac", 0.15
                param_bucket = f"{param}|{timeframe}|{symbol}"
                current = overrides.get(param_bucket, {}).get(param, default)
                if win_rate < HEALTHY_WIN_RATE[0]:
                    proposal = (param, current, round(max(current - 0.03, 0.05), 2), param_bucket,
                                f"win rate {win_rate:.0%} over {len(decisive)} samples -- requiring "
                                f"a truly flatter side to reduce false triangle calls")
                elif win_rate > HEALTHY_WIN_RATE[1]:
                    proposal = (param, current, round(min(current + 0.03, 0.30), 2), param_bucket,
                                f"win rate {win_rate:.0%} over {len(decisive)} samples -- this bucket "
                                f"may tolerate a looser flatness threshold")
        elif pattern in ("Double Top", "Double Bottom", "Head & Shoulders",
                         "Inverse Head & Shoulders", "Broadening Top", "Broadening Bottom",
                         "Selling Climax"):
            pass  # no tunable detection parameter for these yet -- report only
        else:
            # Harmonic pattern (or Shark) -- confluence_minimum applies.
            current_min = overrides.get(bucket, {}).get("confluence_minimum")
            base_min = cfg["confluence_minimum"].get(timeframe, cfg["confluence_minimum"]["default"])
            current_min = current_min if current_min is not None else base_min
            if win_rate < HEALTHY_WIN_RATE[0]:
                proposal = ("confluence_minimum", current_min, min(current_min + 1, 6), bucket,
                            f"win rate {win_rate:.0%} over {len(decisive)} samples -- raising the "
                            f"confluence bar to filter weaker setups")
            elif win_rate > HEALTHY_WIN_RATE[1] and current_min > 2:
                proposal = ("confluence_minimum", current_min, max(current_min - 1, 2), bucket,
                            f"win rate {win_rate:.0%} over {len(decisive)} samples -- this bucket "
                            f"may tolerate a looser confluence bar")

        if proposal:
            param, old_val, new_val, override_bucket, reasoning = proposal
            change_id = str(uuid.uuid4())[:8]
            text = (f"<b>Calibration proposal</b>\nBucket: {bucket}\n"
                    f"{param}: {old_val} -> {new_val}\nReason: {reasoning}")
            msg_id = tg.send_calibration_proposal(change_id, text)
            state_mod.add_pending_calibration(st, change_id, override_bucket, param, old_val,
                                               new_val, reasoning, msg_id)

    tg.send_message("\n".join(lines))
    st["last_monthly_report"] = datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------ #
# Applying Accept/Decline responses (checked every scan run, cheap)
# ------------------------------------------------------------------ #

def apply_calibration_responses(st: dict) -> None:
    offset = st.get("telegram_update_offset")
    responses = tg.get_pending_callback_responses(offset=offset)

    for r in responses:
        st["telegram_update_offset"] = r["update_id"] + 1
        change = state_mod.resolve_pending_calibration(
            st, r["change_id"], "accepted" if r["action"] == "accept" else "declined")
        if not change:
            continue

        tg.answer_callback_query(r["callback_query_id"],
                                  "Applied" if r["action"] == "accept" else "Declined")

        if r["action"] == "accept":
            overrides = st.setdefault("bucket_overrides", {})
            overrides.setdefault(change["bucket"], {})[change["parameter"]] = change["new_value"]
            new_text = (f"\u2705 Applied -- {change['bucket']}: {change['parameter']} "
                       f"set to {change['new_value']}")
        else:
            new_text = f"\u274c Declined -- {change['bucket']} left unchanged"

        tg.edit_message_text(r["message_id"], new_text)


def main():
    cfg = load_config()
    st = state_mod.load_state()

    apply_calibration_responses(st)
    weekly_digest(cfg, st)
    monthly_report(cfg, st)
    state_mod.expire_stale_calibrations(st, cfg["reporting"]["calibration_expiry_days"])

    state_mod.save_state(st)


if __name__ == "__main__":
    main()

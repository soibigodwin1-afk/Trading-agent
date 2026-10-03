"""
Persistent state, committed back to the repo each run (also serves as
the GitHub Actions heartbeat that resets the 60-day auto-disable clock).

Structure:
{
  "last_run": iso timestamp,
  "alerted": {dedup_key: completion_time_iso},          # avoid re-alerting
  "watching": {watch_key: {...pattern info, message_id, opened_at}},
  "outcomes": [ {resolved pattern record...}, ... ],      # append-only log
  "forming": {forming_key: {...projected PRZ, expiry}},   # Phase 2
  "pending_calibrations": [ {change proposal...}, ... ],
  "last_weekly_report": iso timestamp or null,
  "last_monthly_report": iso timestamp or null
}
"""

import json
import os
from datetime import datetime, timedelta, timezone

STATE_PATH = "state.json"
MAX_AGE_DAYS = 45          # prune alerted-dedup entries older than this
OUTCOME_RETENTION_DAYS = 400  # keep resolved outcomes for calibration/history


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_state() -> dict:
    if not os.path.exists(STATE_PATH):
        return {
            "last_run": None, "alerted": {}, "watching": {}, "outcomes": [],
            "forming": {}, "pending_calibrations": [],
            "last_weekly_report": None, "last_monthly_report": None,
        }
    with open(STATE_PATH, "r") as f:
        state = json.load(f)
    for key, default in (("watching", {}), ("outcomes", []), ("forming", {}),
                          ("pending_calibrations", []),
                          ("last_weekly_report", None), ("last_monthly_report", None)):
        state.setdefault(key, default)
    return state


# ---- dedup (unchanged behaviour) ---- #

def already_alerted(state: dict, key: str, completion_time: str) -> bool:
    return state.get("alerted", {}).get(key) == completion_time


def mark_alerted(state: dict, key: str, completion_time: str) -> None:
    state.setdefault("alerted", {})[key] = completion_time


# ---- outcome tracking ---- #

def open_watch(state: dict, watch_key: str, symbol: str, timeframe: str,
                pattern_name: str, direction: str, entry: float, stop: float,
                target: float, message_id: int, chat_id: str,
                max_bars: int) -> None:
    state["watching"][watch_key] = {
        "symbol": symbol, "timeframe": timeframe, "pattern": pattern_name,
        "direction": direction, "entry": entry, "stop": stop, "target": target,
        "message_id": message_id, "chat_id": chat_id,
        "opened_at": _now_iso(), "bars_open": 0, "max_bars": max_bars,
    }


def tick_watches(state: dict) -> None:
    """Call once per scan run: increments the bar-age of every open watch."""
    for w in state.get("watching", {}).values():
        w["bars_open"] = w.get("bars_open", 0) + 1


def resolve_watch(state: dict, watch_key: str, outcome: str, exit_price: float) -> dict:
    """outcome: 'target', 'stop', or 'expired'. Returns the closed record."""
    w = state["watching"].pop(watch_key, None)
    if w is None:
        return {}
    record = dict(w)
    record.update({
        "watch_key": watch_key, "outcome": outcome, "exit_price": exit_price,
        "closed_at": _now_iso(),
    })
    state.setdefault("outcomes", []).append(record)
    return record


def due_watches(state: dict):
    """Yields (watch_key, watch_dict) for every currently open watch."""
    for k, w in list(state.get("watching", {}).items()):
        yield k, w


# ---- forming-pattern tracking (Phase 2) ---- #

def upsert_forming(state: dict, key: str, payload: dict) -> None:
    state.setdefault("forming", {})[key] = payload


def drop_forming(state: dict, key: str) -> None:
    state.get("forming", {}).pop(key, None)


# ---- calibration proposals ---- #

def add_pending_calibration(state: dict, change_id: str, bucket: str,
                             parameter: str, old_value, new_value, reasoning: str,
                             message_id: int) -> None:
    state.setdefault("pending_calibrations", []).append({
        "change_id": change_id, "bucket": bucket, "parameter": parameter,
        "old_value": old_value, "new_value": new_value, "reasoning": reasoning,
        "message_id": message_id, "proposed_at": _now_iso(), "status": "pending",
    })


def resolve_pending_calibration(state: dict, change_id: str, status: str) -> dict:
    """status: 'accepted', 'declined', or 'expired'."""
    for c in state.get("pending_calibrations", []):
        if c["change_id"] == change_id and c["status"] == "pending":
            c["status"] = status
            c["resolved_at"] = _now_iso()
            return c
    return {}


def expire_stale_calibrations(state: dict, max_age_days: int = 14) -> list:
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    expired = []
    for c in state.get("pending_calibrations", []):
        if c["status"] != "pending":
            continue
        proposed = datetime.fromisoformat(c["proposed_at"])
        if proposed.tzinfo is None:
            proposed = proposed.replace(tzinfo=timezone.utc)
        if proposed < cutoff:
            c["status"] = "expired"
            c["resolved_at"] = _now_iso()
            expired.append(c)
    return expired


# ---- save / prune ---- #

def save_state(state: dict) -> None:
    state["last_run"] = _now_iso()

    cutoff = datetime.now(timezone.utc) - timedelta(days=MAX_AGE_DAYS)
    pruned = {}
    for k, v in state.get("alerted", {}).items():
        try:
            ts = datetime.fromisoformat(v)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= cutoff:
                pruned[k] = v
        except (ValueError, TypeError):
            pruned[k] = v
    state["alerted"] = pruned

    outcome_cutoff = datetime.now(timezone.utc) - timedelta(days=OUTCOME_RETENTION_DAYS)
    kept_outcomes = []
    for rec in state.get("outcomes", []):
        try:
            ts = datetime.fromisoformat(rec.get("closed_at", ""))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= outcome_cutoff:
                kept_outcomes.append(rec)
        except (ValueError, TypeError):
            kept_outcomes.append(rec)
    state["outcomes"] = kept_outcomes

    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)

"""
All Telegram I/O via plain HTTP calls to the Bot API.

Covers:
  - sending a pattern alert photo (returns the message_id so outcome
    updates can reply to it)
  - sending a threaded reply (outcome updates)
  - sending an inline Accept/Decline keyboard (calibration proposals)
  - editing a message after a button is pressed
  - polling getUpdates for button presses (callback_query), since the
    scanner isn't a long-running server -- it just checks on each run
"""

import os
import requests

API_ROOT = "https://api.telegram.org/bot{token}/{method}"


def _creds():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set as env vars / repo secrets."
        )
    return token, chat_id


def send_photo(photo_path: str, caption: str, reply_to_message_id: int = None) -> int:
    """Returns the sent message's message_id (needed for later replies)."""
    token, chat_id = _creds()
    url = API_ROOT.format(token=token, method="sendPhoto")
    data = {"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"}
    if reply_to_message_id:
        data["reply_to_message_id"] = reply_to_message_id
    with open(photo_path, "rb") as f:
        resp = requests.post(url, data=data, files={"photo": f}, timeout=30)
    if not resp.ok:
        raise RuntimeError(f"Telegram sendPhoto failed: {resp.status_code} {resp.text}")
    return resp.json()["result"]["message_id"]


def send_message(text: str, reply_to_message_id: int = None,
                  reply_markup: dict = None) -> int:
    token, chat_id = _creds()
    url = API_ROOT.format(token=token, method="sendMessage")
    data = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    if reply_to_message_id:
        data["reply_to_message_id"] = reply_to_message_id
    if reply_markup:
        import json
        data["reply_markup"] = json.dumps(reply_markup)
    resp = requests.post(url, data=data, timeout=30)
    if not resp.ok:
        raise RuntimeError(f"Telegram sendMessage failed: {resp.status_code} {resp.text}")
    return resp.json()["result"]["message_id"]


def send_outcome_reply(original_message_id: int, text: str) -> int:
    """Threaded reply under the original pattern alert."""
    return send_message(text, reply_to_message_id=original_message_id)


def send_calibration_proposal(change_id: str, text: str) -> int:
    """Sends a calibration proposal with Accept/Decline inline buttons.
    The callback_data encodes the change_id so the response can be
    matched back to the right pending_calibrations entry."""
    keyboard = {
        "inline_keyboard": [[
            {"text": "\u2705 Accept", "callback_data": f"cal_accept:{change_id}"},
            {"text": "\u274c Decline", "callback_data": f"cal_decline:{change_id}"},
        ]]
    }
    return send_message(text, reply_markup=keyboard)


def edit_message_text(message_id: int, text: str) -> None:
    token, chat_id = _creds()
    url = API_ROOT.format(token=token, method="editMessageText")
    resp = requests.post(url, data={
        "chat_id": chat_id, "message_id": message_id, "text": text, "parse_mode": "HTML",
    }, timeout=30)
    if not resp.ok:
        raise RuntimeError(f"Telegram editMessageText failed: {resp.status_code} {resp.text}")


def get_pending_callback_responses(offset: int = None) -> list:
    """
    Polls getUpdates for any button presses since the last check.
    Returns a list of {"change_id": ..., "action": "accept"|"decline",
    "update_id": ...} -- caller should track the highest update_id seen
    and pass offset=<that + 1> next time to avoid re-processing.
    """
    token, _ = _creds()
    url = API_ROOT.format(token=token, method="getUpdates")
    params = {"timeout": 0}
    if offset is not None:
        params["offset"] = offset
    resp = requests.get(url, params=params, timeout=30)
    if not resp.ok:
        raise RuntimeError(f"Telegram getUpdates failed: {resp.status_code} {resp.text}")

    results = []
    for update in resp.json().get("result", []):
        cq = update.get("callback_query")
        if not cq:
            continue
        data = cq.get("data", "")
        if data.startswith("cal_accept:"):
            results.append({"change_id": data.split(":", 1)[1], "action": "accept",
                             "update_id": update["update_id"],
                             "callback_query_id": cq["id"],
                             "message_id": cq["message"]["message_id"]})
        elif data.startswith("cal_decline:"):
            results.append({"change_id": data.split(":", 1)[1], "action": "decline",
                             "update_id": update["update_id"],
                             "callback_query_id": cq["id"],
                             "message_id": cq["message"]["message_id"]})
    return results


def answer_callback_query(callback_query_id: str, text: str = "") -> None:
    """Clears the loading spinner on the tapped button."""
    token, _ = _creds()
    url = API_ROOT.format(token=token, method="answerCallbackQuery")
    requests.post(url, data={"callback_query_id": callback_query_id, "text": text}, timeout=15)

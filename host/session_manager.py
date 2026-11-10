"""Session Management Engine for nuro.

Allows creating, listing, and switching named calculation sessions:
- Active session persists in a local state JSON file
- Calculations tagged with the active session
- Syncs seamlessly to Obsidian under [[Session - <name>]]
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

STATE_FILE = Path(__file__).parent / "session_state.json"

DEFAULT_STATE: Dict[str, Any] = {
    "active_session": "Main-Session",
    "sessions": {
        "Main-Session": {
            "created": "2026-11-10",
            "description": "Default scientific calculation session",
            "calc_count": 0,
        }
    },
    "last_result": None,
    "last_query": None,
    "last_domain": None,
}


def load_state() -> Dict[str, Any]:
    """Load session state from disk or initialize default."""
    if not STATE_FILE.exists():
        save_state(DEFAULT_STATE)
        return DEFAULT_STATE.copy()
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if "active_session" not in data:
            data["active_session"] = "Main-Session"
        return data
    except Exception:
        return DEFAULT_STATE.copy()


def save_state(state: Dict[str, Any]) -> None:
    """Save session state to disk."""
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception:
        pass


def get_active_session() -> str:
    """Get the current active session name."""
    state = load_state()
    return state.get("active_session", "Main-Session")


def set_active_session(name: str, description: str = "") -> str:
    """Switch to an existing session or create a new one."""
    state = load_state()
    clean_name = "".join(c if c.isalnum() or c in ("-", "_", " ") else "_" for c in name).strip()
    if not clean_name:
        clean_name = "Main-Session"

    if clean_name not in state.get("sessions", {}):
        import datetime
        state.setdefault("sessions", {})[clean_name] = {
            "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "description": description or f"Session {clean_name}",
            "calc_count": 0,
        }
    state["active_session"] = clean_name
    save_state(state)
    return clean_name


def list_sessions() -> List[Tuple[str, bool, int, str]]:
    """Return list of (session_name, is_active, calc_count, description)."""
    state = load_state()
    active = state.get("active_session", "Main-Session")
    sessions = state.get("sessions", {})
    res = []
    for s_name, info in sorted(sessions.items()):
        is_act = (s_name == active)
        cnt = info.get("calc_count", 0)
        desc = info.get("description", "")
        res.append((s_name, is_act, cnt, desc))
    return res


def record_calc_in_session(domain: str, query: str, result_text: str) -> None:
    """Increment calculation count and cache last calculation."""
    state = load_state()
    active = state.get("active_session", "Main-Session")
    if active in state.get("sessions", {}):
        state["sessions"][active]["calc_count"] = state["sessions"][active].get("calc_count", 0) + 1
    state["last_result"] = result_text
    state["last_query"] = query
    state["last_domain"] = domain
    save_state(state)


def get_last_calculation() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (domain, query, result_text) of last executed calculation."""
    state = load_state()
    return state.get("last_domain"), state.get("last_query"), state.get("last_result")

"""Obsidian Graph View & Dynamic Vault Sync for nuro.

Capabilities:
1. Explicit Vault Targeting:
   - Saves calculation ONLY when explicitly instructed with a vault name:
     e.g., 'to obsidian Acids, Bases and Salts', 'save to obsidian ICSE Class 10', 'to obsidian Radioactivity'
   - Fuzzy resolves existing vaults in G:\\obsidian or creates new ones.
2. Dynamic Session Tracking:
   - Groups notes by active session: [[Session - <name>]]
   - Auto-updates session index inside the selected Obsidian vault.
3. Interactive Graph View Launch:
   - Directly opens the targeted vault via 'obsidian://open?vault=...'
"""
from __future__ import annotations

import datetime
import json
import os
import re
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

OBSIDIAN_ROOT = Path("G:/obsidian")
DEFAULT_VAULT_NAME = "Neural Loop Calculations"
OBSIDIAN_EXE = Path("G:/obsidian/Obsidian.com")

C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_CYAN = "\033[38;5;75m"
C_GREEN = "\033[38;5;78m"
C_AMBER = "\033[38;5;214m"
C_PURPLE = "\033[38;5;141m"
C_GRAY = "\033[38;5;244m"
C_WHITE = "\033[38;5;255m"


def extract_obsidian_directive(query: str) -> Tuple[str, Optional[str]]:
    """Check if query requests saving to Obsidian with a vault name.
    Examples:
      - '2SO2 + O2 -> 2SO3 to obsidian Acids, Bases and Salts'
      - 'pascals law F1=50 save to obsidian ICSE Class 10'
      - 'x^3 - 3*x obsidian: DevStack'
    Returns (cleaned_query, vault_name or None).
    """
    patterns = [
        r"(?:save\s+)?to\s+obsidian(?:\s+vault)?(?:\s*[:=]\s*|\s+)([\w\s,\-]+)$",
        r"\bobsidian(?:\s+vault)?(?:\s*[:=]\s*|\s+)([\w\s,\-]+)$",
        r"--(?:vault|obsidian)\s+([\w\s,\-]+)$",
    ]
    for p in patterns:
        m = re.search(p, query, flags=re.I)
        if m:
            vault_name = m.group(1).strip()
            # Clean off trailing quotation marks if any
            vault_name = vault_name.strip("\"'")
            cleaned_query = query[:m.start()].strip()
            return cleaned_query, vault_name

    # Check for simple bare 'to obsidian' without vault name
    m_bare = re.search(r"(?:save\s+)?to\s+obsidian\s*$", query, flags=re.I)
    if m_bare:
        cleaned_query = query[:m_bare.start()].strip()
        return cleaned_query, DEFAULT_VAULT_NAME

    return query, None


def resolve_vault_path(vault_name: Optional[str] = None) -> Tuple[Path, str]:
    """Resolve vault name against existing vaults in G:\\obsidian or create a new one."""
    OBSIDIAN_ROOT.mkdir(parents=True, exist_ok=True)
    if not vault_name:
        vault_name = DEFAULT_VAULT_NAME

    # Check exact match first
    target = OBSIDIAN_ROOT / vault_name
    if target.exists() and target.is_dir():
        return target, target.name

    # Fuzzy match among existing directories
    v_low = vault_name.lower()
    for d in OBSIDIAN_ROOT.iterdir():
        if d.is_dir() and not d.name.startswith("."):
            d_low = d.name.lower()
            if v_low == d_low or v_low in d_low or d_low in v_low:
                return d, d.name

    # If not found, create new vault directory
    target.mkdir(parents=True, exist_ok=True)
    return target, vault_name


def ensure_vault_initialized(vault_path: Path) -> None:
    """Initialize Obsidian vault folder and graph.json color groups."""
    vault_path.mkdir(parents=True, exist_ok=True)
    obsidian_dir = vault_path / ".obsidian"
    obsidian_dir.mkdir(parents=True, exist_ok=True)

    core_plugins_path = obsidian_dir / "core-plugins.json"
    if not core_plugins_path.exists():
        core_plugins_path.write_text(
            json.dumps(["graph", "backlink", "outgoing-link", "tag-pane", "search", "markdown-importer"], indent=2),
            encoding="utf-8",
        )

    app_json = obsidian_dir / "app.json"
    if not app_json.exists():
        app_json.write_text(
            json.dumps({"showLineNumber": True, "useTab": True, "tabSize": 2}, indent=2),
            encoding="utf-8",
        )

    graph_json = obsidian_dir / "graph.json"
    if not graph_json.exists():
        graph_config = {
            "collapse-filter": False,
            "search": "",
            "showTags": True,
            "showAttachments": False,
            "hideUnresolved": False,
            "showOrphans": True,
            "collapse-color-groups": False,
            "colorGroups": [
                {"query": "path:\"Session\"", "color": {"a": 1, "rgb": 16498468}},        # Gold
                {"query": "path:\"Math\"", "color": {"a": 1, "rgb": 3900150}},            # Cyan
                {"query": "path:\"Chemistry\"", "color": {"a": 1, "rgb": 1096065}},       # Emerald
                {"query": "path:\"Physics\"", "color": {"a": 1, "rgb": 16096779}},        # Amber
                {"query": "path:\"Theories\"", "color": {"a": 1, "rgb": 11032311}},       # Purple
            ],
            "nodeSizeMultiplier": 1.4,
            "lineSizeMultiplier": 1.3,
            "showArrow": True,
            "scale": 0.35,
            "close": False,
        }
        graph_json.write_text(json.dumps(graph_config, indent=2), encoding="utf-8")


def sanitize_filename(name: str) -> str:
    """Sanitize strings for Windows filenames."""
    clean = "".join(c if c.isalnum() or c in (" ", "-", "_", "+", "^") else "_" for c in name)
    return clean.strip()[:60]


def record_session_calculation(
    domain: str,
    title: str,
    query: str,
    result_text: str,
    tags: List[str],
    related_links: List[str],
    vault_name: Optional[str] = None,
) -> Path:
    """Record calculation to the explicitly requested Obsidian vault with session link."""
    from host.session_manager import get_active_session

    active_session = get_active_session()
    vault_path, actual_vault_name = resolve_vault_path(vault_name)
    ensure_vault_initialized(vault_path)

    # Save inside a dedicated Session folder in the vault
    session_dir = vault_path / f"Session - {sanitize_filename(active_session)}"
    session_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_title = f"{sanitize_filename(title)}.md"
    note_path = session_dir / file_title

    # Stripping ANSI escape sequences
    ansi_stripped = ""
    in_escape = False
    for char in result_text:
        if char == "\033":
            in_escape = True
        elif in_escape:
            if char.isalpha():
                in_escape = False
        else:
            ansi_stripped += char

    wikilinks = [
        f"[[Session - {active_session}]]",
        f"[[Domain - {domain.capitalize()}]]",
    ]
    for link in related_links:
        wikilinks.append(f"[[{link}]]")

    content = f"""---
title: "{title}"
domain: "{domain.capitalize()}"
session: "{active_session}"
vault: "{actual_vault_name}"
timestamp: "{timestamp}"
tags: [{', '.join(f'"{t}"' for t in tags)}]
query: "{query.replace('"', "'")}"
---

# {title}

> **Vault:** `{actual_vault_name}` | **Session:** [[Session - {active_session}]] | **Logged:** `{timestamp}`
> **Connected Nodes:** {' '.join(wikilinks)}
> **Tags:** {' '.join(f'#{t}' for t in tags)}

## ⚡ Query
```text
{query}
```

## 📊 Solution & Model
```text
{ansi_stripped.strip()}
```

## 🔗 Knowledge Graph Connections
- [[Session - {active_session}]]
- [[Domain - {domain.capitalize()}]]
{"".join(f"- [[{link}]]\n" for link in related_links)}

---
*nuro v2.0 dynamic session sync*
"""
    note_path.write_text(content, encoding="utf-8")

    # Update session MOC note
    update_session_moc(vault_path, actual_vault_name, active_session)
    return note_path


def update_session_moc(vault_path: Path, vault_name: str, session_name: str) -> None:
    """Update or create the Session Map of Content note in the vault."""
    session_dir = vault_path / f"Session - {sanitize_filename(session_name)}"
    notes = sorted([f for f in session_dir.glob("*.md")])

    content = f"""---
title: "Session - {session_name}"
type: "session_moc"
vault: "{vault_name}"
total_calculations: {len(notes)}
---

# 📚 Session — {session_name}

> **Obsidian Vault:** `{vault_name}`
> **Active Notes:** {len(notes)} calculations & models

## Calculations & Theories in this Session
{"".join(f"- [[{n.stem}]]\n" for n in notes)}

---
*Press `Ctrl + G` in Obsidian to see this session's interconnected visual graph!*
"""
    (vault_path / f"Session - {sanitize_filename(session_name)}.md").write_text(content, encoding="utf-8")


def launch_obsidian_vault(vault_name: Optional[str] = None) -> str:
    """Open specified Obsidian vault in Obsidian."""
    vault_path, actual_name = resolve_vault_path(vault_name)
    ensure_vault_initialized(vault_path)

    uri = f"obsidian://open?vault={actual_name.replace(' ', '%20')}"
    try:
        webbrowser.open(uri)
    except Exception:
        pass

    if OBSIDIAN_EXE.exists():
        try:
            subprocess.Popen([str(OBSIDIAN_EXE), str(vault_path)], shell=True)
        except Exception:
            pass

    out = [
        f"{C_BOLD}{C_GREEN}⚡ OBSIDIAN VAULT SYNCED & LAUNCHED:{C_RESET}",
        f"  {C_GRAY}Target Vault    :{C_RESET} {C_BOLD}{C_WHITE}{actual_name}{C_RESET}",
        f"  {C_GRAY}Directory Path  :{C_RESET} {C_CYAN}{vault_path}{C_RESET}",
        f"  {C_BOLD}{C_CYAN}Obsidian URI    :{C_RESET} {uri}",
        f"  {C_PURPLE}Graph View Key  :{C_RESET} Press {C_BOLD}{C_WHITE}Ctrl + G{C_RESET} inside Obsidian to open the node web!",
    ]
    return "\n".join(out)

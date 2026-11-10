"""Obsidian Graph View & Session Calculation Sync for nuro.

Syncs all calculations, reactions, and physics formulas solved across nuro sessions
into an interconnected Obsidian vault at 'G:\\obsidian\\Neural Loop Calculations'
with a high-contrast color-coded Graph View and bidirectional [[wikilinks]].
"""
from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Dict, Any, List, Optional

VAULT_PATH = Path("G:/obsidian/Neural Loop Calculations")
OBSIDIAN_EXE = Path("G:/obsidian/Obsidian.com")

C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_CYAN = "\033[38;5;75m"
C_GREEN = "\033[38;5;78m"
C_AMBER = "\033[38;5;214m"
C_PURPLE = "\033[38;5;141m"
C_GRAY = "\033[38;5;244m"
C_WHITE = "\033[38;5;255m"


def ensure_vault_initialized() -> None:
    """Initialize Obsidian vault folder and graph.json color groups."""
    VAULT_PATH.mkdir(parents=True, exist_ok=True)
    obsidian_dir = VAULT_PATH / ".obsidian"
    obsidian_dir.mkdir(parents=True, exist_ok=True)

    # Core plugins config
    core_plugins_path = obsidian_dir / "core-plugins.json"
    if not core_plugins_path.exists():
        core_plugins_path.write_text(
            json.dumps(["graph", "backlink", "outgoing-link", "tag-pane", "search", "markdown-importer"], indent=2),
            encoding="utf-8",
        )

    # App config
    app_json = obsidian_dir / "app.json"
    if not app_json.exists():
        app_json.write_text(
            json.dumps({"showLineNumber": True, "useTab": True, "tabSize": 2, "attachmentFolderPath": "attachments"}, indent=2),
            encoding="utf-8",
        )

    # High-contrast color-coded graph configuration
    graph_json = obsidian_dir / "graph.json"
    graph_config = {
        "collapse-filter": False,
        "search": "",
        "showTags": True,
        "showAttachments": False,
        "hideUnresolved": False,
        "showOrphans": True,
        "collapse-color-groups": False,
        "colorGroups": [
            {"query": "path:\"00 - Session Hub\"", "color": {"a": 1, "rgb": 16498468}},  # Gold (MOC)
            {"query": "path:\"Math\"", "color": {"a": 1, "rgb": 3900150}},              # Cyan (Math)
            {"query": "path:\"Chemistry\"", "color": {"a": 1, "rgb": 1096065}},         # Emerald Green (Chem)
            {"query": "path:\"Physics\"", "color": {"a": 1, "rgb": 16096779}},          # Amber (Physics)
            {"query": "path:\"Catalysts\"", "color": {"a": 1, "rgb": 15680580}},        # Rose Red (Catalysts)
            {"query": "path:\"Concepts\"", "color": {"a": 1, "rgb": 11032311}},         # Purple (Concepts)
        ],
        "collapse-display": False,
        "showArrow": True,
        "textFadeMultiplier": -2.5,
        "nodeSizeMultiplier": 1.45,
        "lineSizeMultiplier": 1.3,
        "collapse-forces": False,
        "centerStrength": 0.45,
        "repelStrength": 16.0,
        "linkStrength": 0.75,
        "linkDistance": 180,
        "scale": 0.35,
        "close": False,
    }
    # Write graph config
    graph_json.write_text(json.dumps(graph_config, indent=2), encoding="utf-8")


def sanitize_filename(name: str) -> str:
    """Sanitize strings for Windows filenames."""
    clean = re_clean = "".join(c if c.isalnum() or c in (" ", "-", "_", "+", "^") else "_" for c in name)
    return clean.strip()[:60]


def record_session_calculation(
    domain: str,  # 'Math', 'Chemistry', 'Physics'
    title: str,
    query: str,
    result_text: str,
    tags: List[str],
    related_links: List[str],
) -> Path:
    """Record a calculation into the Obsidian vault with bidirectional [[wikilinks]]."""
    ensure_vault_initialized()

    domain_clean = domain.capitalize()
    domain_dir = VAULT_PATH / domain_clean
    domain_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_title = f"{sanitize_filename(title)}.md"
    note_path = domain_dir / file_title

    # Build Wikilinks block
    wikilinks = [f"[[00 - Session Hub (MOC)]]", f"[[Domain - {domain_clean}]]"]
    for link in related_links:
        wikilinks.append(f"[[{link}]]")

    links_str = " ".join(wikilinks)
    tags_str = " ".join(f"#{t.replace(' ', '_')}" for t in tags)

    # Strip ANSI colors for Obsidian note
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

    content = f"""---
title: "{title}"
domain: "{domain_clean}"
timestamp: "{timestamp}"
tags: [{', '.join(f'"{t}"' for t in tags)}]
query: "{query.replace('"', "'")}"
---

# {title}

> **Domain:** [[Domain - {domain_clean}]] | **Logged:** `{timestamp}`
> **Connected Nodes:** {links_str}
> **Tags:** {tags_str}

## ⚡ Query
```text
{query}
```

## 📊 Result & Solution
```text
{ansi_stripped.strip()}
```

## 🔗 Related Knowledge Nodes
{"".join(f"- [[{link}]]\n" for link in related_links)}
- [[00 - Session Hub (MOC)]]
- [[Domain - {domain_clean}]]

---
*Generated by nuro v2.0 session engine*
"""
    note_path.write_text(content, encoding="utf-8")

    # Update Domain index and Hub MOC
    update_domain_index(domain_clean)
    update_hub_moc()

    return note_path


def update_domain_index(domain: str) -> None:
    """Regenerate domain MOC index note."""
    domain_dir = VAULT_PATH / domain
    notes = sorted([f for f in domain_dir.glob("*.md") if f.name != f"Index - {domain}.md"])

    content = f"""---
title: "Domain - {domain}"
domain: "{domain}"
---

# 🌐 {domain} — Calculation Index

> **Connected Hub:** [[00 - Session Hub (MOC)]]
> **Total Notes:** {len(notes)}

## Calculations & Models
{"".join(f"- [[{n.stem}]]\n" for n in notes)}

---
*Neural Loop Obsidian Knowledge Graph*
"""
    (VAULT_PATH / f"Domain - {domain}.md").write_text(content, encoding="utf-8")


def update_hub_moc() -> None:
    """Regenerate Master Map of Content (MOC)."""
    domains = ["Math", "Chemistry", "Physics"]
    total_notes = len(list(VAULT_PATH.glob("**/*.md")))

    sections = []
    for d in domains:
        d_path = VAULT_PATH / d
        if d_path.exists():
            notes = sorted(list(d_path.glob("*.md")))
            links = "\n".join(f"  - [[{n.stem}]]" for n in notes[:10])
            if len(notes) > 10:
                links += f"\n  - *...and {len(notes) - 10} more in [[Domain - {d}]]*"
            sections.append(f"### [[Domain - {d}]] ({len(notes)} items)\n{links}")

    content = f"""---
title: "Neural Loop Session Hub (MOC)"
tags: [hub, moc, nuro, neural_loop]
---

# 🧠 Neural Loop — Master Calculation & Science Hub

Welcome to the live interactive knowledge graph of calculations, chemistry reaction pathways, and physics simulations solved during `nuro` sessions.

## 📌 Domain Maps
- [[Domain - Math]] — Derivatives, Integrals, Roots, Multi-function Braille plots
- [[Domain - Chemistry]] — Balanced reactions, Catalytic activation barriers, Stoichiometry, Titrations
- [[Domain - Physics]] — Ballistics, RLC Transients, Special Relativity, Quantum Transitions, Orbital Mechanics

## 📚 Recent Session Calculations
{chr(10).join(sections)}

---
*Press `Ctrl+G` inside Obsidian to open the interactive Color-Coded Graph View!*
"""
    (VAULT_PATH / "00 - Session Hub (MOC).md").write_text(content, encoding="utf-8")


def open_obsidian_graph_view() -> str:
    """Open the Obsidian vault and instruct the user how to view the Graph."""
    ensure_vault_initialized()
    update_hub_moc()

    vault_name = "Neural Loop Calculations"
    uri = f"obsidian://open?vault={vault_name.replace(' ', '%20')}"

    # Try launching Obsidian via URI or executable
    try:
        webbrowser.open(uri)
    except Exception:
        pass

    if OBSIDIAN_EXE.exists():
        try:
            subprocess.Popen([str(OBSIDIAN_EXE), str(VAULT_PATH)], shell=True)
        except Exception:
            pass

    out = [
        f"{C_BOLD}{C_GREEN}⚡ OBSIDIAN GRAPH VIEW SYNCED & LAUNCHED:{C_RESET}",
        f"  {C_GRAY}Vault Directory :{C_RESET} {C_WHITE}{VAULT_PATH}{C_RESET}",
        f"  {C_BOLD}{C_CYAN}Obsidian URI    :{C_RESET} {C_CYAN}{uri}{C_RESET}",
        f"  {C_AMBER}Master Hub Note :{C_RESET} {C_WHITE}00 - Session Hub (MOC).md{C_RESET}",
        f"  {C_PURPLE}Graph View Key  :{C_RESET} Press {C_BOLD}{C_WHITE}Ctrl + G{C_RESET} in Obsidian to expand the color-coded node web!",
    ]
    return "\n".join(out)

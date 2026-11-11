"""Standalone executable entry point for Neural Loop (nuro.exe).

When double-clicked directly in Windows Explorer (no arguments), it opens the
live 60 FPS interactive TUI workstation with dynamic plug-and-play auto-detection.
When called with CLI arguments (e.g. 'nuro.exe doctor', 'nuro.exe bench'), it
delegates directly to the unified CLI.
"""

import sys
from host.cli import main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Default action when double-clicked with no CLI flags: open the live TUI!
        sys.argv.append("tui")
    main()

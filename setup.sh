#!/usr/bin/env bash
# Plug-and-play bootstrap for Linux and macOS.
#
#   ./setup.sh              # create .venv, install the project, run the doctor
#   ./setup.sh --recreate --confirm RECREATE
#
# If the board is on /dev/ttyACM* or /dev/ttyUSB* and cannot be opened:
#   sudo usermod -aG dialout "$USER"   # then log out and back in
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo"

python=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
            python="$candidate"
            break
        fi
    fi
done

if [ -z "$python" ]; then
    echo "Python 3.10+ was not found. Install it with your package manager first." >&2
    exit 2
fi

export PYTHONPATH="$repo"
exec "$python" -m host.cli setup "$@"

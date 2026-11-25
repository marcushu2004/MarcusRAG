"""Backward compatible entry point.

The original `main.py` held business logic and was imported by `src/` modules,
which created the circular imports. It is now a thin shim over the CLI:

    python main.py index
    python main.py ask "什么是 policy network?"
    python main.py info
"""

from __future__ import annotations

from marcusrag.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

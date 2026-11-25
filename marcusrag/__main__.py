"""Allow `python -m marcusrag ...` as a shortcut for `python -m marcusrag.cli`."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())

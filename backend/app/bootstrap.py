from __future__ import annotations

import sys
from pathlib import Path


def ensure_src_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root

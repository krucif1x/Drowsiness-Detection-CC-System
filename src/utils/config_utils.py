from __future__ import annotations

from typing import Any, Dict


def as_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def as_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def sec_to_frames(sec: Any, fps: float, default_sec: float) -> int:
    return max(1, int(as_float(sec, default_sec) * float(fps)))


def get_section(root: Dict[str, Any], key: str) -> Dict[str, Any]:
    v = root.get(key, {})
    return v if isinstance(v, dict) else {}
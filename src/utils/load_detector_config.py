import logging
from typing import Any, Dict

import yaml

log = logging.getLogger(__name__)


def load_yaml_section(path: str, section: str) -> Dict[str, Any]:
    """
    Load a YAML section as dict. Supports dotted paths like:
      - "detectors.drowsiness"
      - "detectors.distraction"
    Returns {} on error/missing.
    """
    try:
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            return {}

        node: Any = raw
        for key in (section or "").split("."):
            if not key:
                continue
            if not isinstance(node, dict):
                return {}
            node = node.get(key, {})

        return node if isinstance(node, dict) else {}
    except Exception as e:
        log.warning(f"Config error for section '{section}' at '{path}': {e}. Using defaults.")
        return {}
from __future__ import annotations

from typing import Any, Dict

from src.utils.load_detector_config import load_yaml_section
from src.utils.config_utils import as_float, as_int, get_section


def load_distraction_config(path: str) -> Dict[str, Any]:
    root = load_yaml_section(path, "detectors.distraction")

    thresholds = get_section(root, "thresholds")
    timing = get_section(root, "timing")
    smoothing = get_section(root, "smoothing")
    severity = get_section(root, "severity")

    return {
        "thresholds": {
            "yaw_deg": as_float(thresholds.get("yaw_deg"), 45.0),
            "pitch_down_deg": as_float(thresholds.get("pitch_down_deg"), 20.0),
            "pitch_up_deg": as_float(thresholds.get("pitch_up_deg"), 20.0),
        },
        "timing": {
            "hands_visible_sec": as_float(timing.get("hands_visible_sec"), 1.5),
            "gaze_sec": as_float(timing.get("gaze_sec"), 1.2),
        },
        "smoothing": {
            "history_frames": as_int(smoothing.get("history_frames"), 5),
            "required_frames": as_int(smoothing.get("required_frames"), 3),
            "face_visibility_history_frames": as_int(smoothing.get("face_visibility_history_frames"), 10),
            "face_present_min_samples": as_int(smoothing.get("face_present_min_samples"), 5),
            "partial_face_threshold": as_float(smoothing.get("partial_face_threshold"), 0.4),
        },
        "severity": {
            "long_duration_high_sec": as_float(severity.get("long_duration_high_sec"), 3.0),
        },
    }
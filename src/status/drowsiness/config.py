from __future__ import annotations

from typing import Any, Dict

from src.utils.load_detector_config import load_yaml_section
from src.utils.config_utils import as_float, as_int, get_section, sec_to_frames


def load_drowsiness_config(path: str, fps: float) -> Dict[str, Any]:
    root = load_yaml_section(path, "detectors.drowsiness")

    episode = get_section(root, "episode")
    ear = get_section(root, "ear")
    blink = get_section(root, "blink")
    yawn = get_section(root, "yawn")
    sup = get_section(root, "expression_suppression")
    sev_root = get_section(root, "severity")
    sev = get_section(sev_root, "drowsiness")

    ear_low = as_float(ear.get("low_threshold"), 0.22)
    ear_high = as_float(ear.get("high_threshold"), 0.26)
    ear_high_ratio = (ear_high / ear_low) if ear_low > 0 else 1.2

    # NEW: perclos section (safe defaults)
    perclos = get_section(root, "perclos")
    perclos_window_sec = as_float(perclos.get("window_sec"), 30.0)

    return {
        "episode": {
            "start_frames": sec_to_frames(episode.get("start_threshold_sec"), fps, 0.5),
            "end_frames": sec_to_frames(episode.get("end_grace_sec"), fps, 1.5),
            "min_episode_sec": as_float(episode.get("min_episode_sec"), 2.0),
            # NEW: shorten onset when sudden EAR drop is detected
            "drop_start_multiplier": as_float(episode.get("drop_start_multiplier"), 0.6),
        },
        "ear": {
            "low": ear_low,
            "high": ear_high,
            "high_ratio": float(ear_high_ratio),
            "drop": as_float(ear.get("drop_threshold"), 0.10),
            "drop_window_frames": as_int(ear.get("drop_window_frames"), 15),
            "history_frames": sec_to_frames(ear.get("history_sec"), fps, 1.0),
            # NEW: smoothing for EAR
            "ema_alpha": as_float(ear.get("ema_alpha"), 0.35),
        },
        "perclos": {
            "window_frames": sec_to_frames(perclos_window_sec, fps, 30.0),
            "threshold": as_float(perclos.get("threshold"), 0.25),
        },
        "blink": {
            "min_closed_frames": as_int(blink.get("min_closed_frames"), 1),
            "max_closed_frames": as_int(blink.get("max_closed_frames"), 10),
        },
        "yawn": {
            "thresh_frames": sec_to_frames(yawn.get("threshold_sec"), fps, 0.27),
            "cooldown_frames": sec_to_frames(yawn.get("cooldown_sec"), fps, 2.0),
            "hand_cover_distance_norm": as_float(yawn.get("hand_cover_distance_norm"), 0.15),
            "frequency_window_sec": as_float(yawn.get("frequency_window_sec"), 120.0),
            "high_frequency_count": as_int(yawn.get("high_frequency_count"), 3),
            "timestamps_max": as_int(yawn.get("timestamps_max"), 10),
            # NEW: reduce false yawns when hand is near face
            "covered_mar_min": as_float(yawn.get("covered_mar_min"), 0.45),
        },
        "expression_suppression": {
            "smile_frames": sec_to_frames(sup.get("smile_suppress_sec"), fps, 0.5),
            "laugh_frames": sec_to_frames(sup.get("laugh_suppress_sec"), fps, 0.67),
        },
        "severity": {
            "drowsiness": {
                "medium_sec": as_float(sev.get("medium_sec"), 2.0),
                "high_sec": as_float(sev.get("high_sec"), 3.0),
                "critical_sec": as_float(sev.get("critical_sec"), 5.0),
            }
        },
    }
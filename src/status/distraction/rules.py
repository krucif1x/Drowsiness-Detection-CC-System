from __future__ import annotations

from typing import Optional, Tuple


def determine_violation(
    *,
    pitch: float,
    yaw: float,
    cal_pitch: float,
    cal_yaw: float,
    yaw_thr: float,
    pitch_down_thr: float,
    pitch_up_thr: float,
    hands_count: int,
    is_drowsy: bool,
    time_hands_visible: float,
    time_gaze: float,
) -> Tuple[Optional[str], Optional[float], Optional[str], str]:
    """
    Returns: (violation_type, time_threshold_sec, alert_detail, base_severity)
    """
    hands_visible = hands_count > 0

    dp = pitch - cal_pitch
    dy = abs(yaw - cal_yaw)

    looking_aside = dy > yaw_thr
    looking_down = dp > pitch_down_thr
    looking_up = dp < -pitch_up_thr

    if is_drowsy and looking_down:
        return None, None, None, "Medium"

    if hands_visible:
        if hands_count == 1:
            return "ONE HAND OFF WHEEL", time_hands_visible, "One Hand Off Wheel", "Medium"
        return "BOTH HANDS OFF WHEEL", time_hands_visible, "Both Hands Off Wheel", "High"

    if looking_aside:
        return "LOOKING ASIDE", time_gaze, "Looking Away from Road", "Medium"
    if looking_up:
        return "LOOKING UP", time_gaze, "Looking Up Away from Road", "Medium"
    if looking_down and not is_drowsy:
        return "LOOKING DOWN", time_gaze, "Looking Down at Device", "Medium"

    return None, None, None, "Medium"


def final_severity(violation_type: str, elapsed_sec: float, long_duration_high_sec: float, base_severity: str) -> str:
    if "BOTH HANDS" in violation_type:
        return "High"
    if elapsed_sec >= long_duration_high_sec:
        return "High"
    return base_severity
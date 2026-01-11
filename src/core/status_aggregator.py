from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class FinalStatus:
    label: str
    color_bgr: Tuple[int, int, int]
    should_log_distraction: bool = False
    distraction_event_name: Optional[str] = None
    distraction_duration_sec: float = 0.0
    distraction_alert_detail: Optional[str] = None
    distraction_severity: Optional[str] = None


class StatusAggregator:
    """
    Decide the single status to display/log based on detector outputs.

    Priority (with fainting removed):
        1) Drowsiness (DROWSY/YAWN/SLEEP)
        2) Distraction
        3) NORMAL
    """

    @staticmethod
    def aggregate(
        *,
        drowsy_status: str,
        drowsy_color_bgr: Tuple[int, int, int],
        is_distracted: bool,
        distraction_type: str,
        should_log_distraction: bool,
        distraction_info: Optional[Dict[str, Any]],
    ) -> FinalStatus:
        # Priority 1: Drowsiness
        if any(k in (drowsy_status or "") for k in ("DROWSY", "YAWN", "SLEEP")):
            return FinalStatus(label=drowsy_status, color_bgr=drowsy_color_bgr)

        # Priority 2: Distraction
        if is_distracted:
            reason = distraction_type or "DISTRACTED"

            # Default mapping
            label = "DISTRACTED"
            color = (0, 0, 255)
            alert_detail = "Driver Distracted"
            severity = "Medium"

            if "BOTH HANDS" in reason:
                label = "BOTH HANDS VISIBLE!"
                color = (0, 0, 255)
                alert_detail = "Both Hands Off Wheel"
                severity = "High"
            elif "ONE HAND" in reason:
                label = "ONE HAND VISIBLE"
                color = (0, 165, 255)
                alert_detail = "One Hand Off Wheel"
                severity = "Medium"
            elif "ASIDE" in reason:
                label = "LOOKING ASIDE"
                color = (0, 255, 255)
                alert_detail = "Looking Away from Road"
                severity = "Medium"
            elif "DOWN" in reason:
                label = "LOOKING DOWN"
                color = (0, 255, 255)
                alert_detail = "Looking Down at Device"
                severity = "Medium"
            elif "UP" in reason:
                label = "LOOKING UP"
                color = (0, 255, 255)
                alert_detail = "Looking Up Away from Road"
                severity = "Medium"

            duration = 0.0
            if distraction_info:
                try:
                    duration = float(distraction_info.get("duration", 0.0) or 0.0)
                except Exception:
                    duration = 0.0

            return FinalStatus(
                label=label,
                color_bgr=color,
                should_log_distraction=bool(should_log_distraction and distraction_info),
                distraction_event_name=f"DISTRACTION_{reason}",
                distraction_duration_sec=duration,
                distraction_alert_detail=(distraction_info or {}).get("alert_detail", alert_detail) if distraction_info else alert_detail,
                distraction_severity=(distraction_info or {}).get("severity", severity) if distraction_info else severity,
            )

        # Priority 3: Normal
        return FinalStatus(label="NORMAL", color_bgr=(0, 255, 0))
import time
import logging
from collections import deque

from src.utils.load_detector_config import load_yaml_section

log = logging.getLogger(__name__)


class DistractionDetector:
    """
    Simple camera-appropriate distraction detector:
    - Camera shows face + shoulders only (hands normally NOT visible)
    - If ANY hand appears in frame = driver not holding wheel properly
    - Also checks extreme head angles (looking away)
    """

    def __init__(self, fps=30.0, camera_pitch=0.0, camera_yaw=0.0, config_path="config/detector_config.yaml"):
        self.fps = fps
        self.cfg = self._load_config(config_path, fps)

        # State
        self.is_distracted = False
        self.distraction_type = "NORMAL"
        self.start_time = None

        # Calibration
        self.cal = {"pitch": camera_pitch, "yaw": camera_yaw}

        # History: track violations over last N frames (from YAML)
        self.history = deque(maxlen=int(self.cfg["history_frames"]))

        # Metrics
        self.metrics = {"total_distractions": 0}

        # Track partial face visibility (from YAML)
        self.face_visibility_history = deque(maxlen=int(self.cfg["face_visibility_history_frames"]))
        self.partial_face_threshold = float(self.cfg["partial_face_threshold"])
        self.face_present_min_samples = int(self.cfg["face_present_min_samples"])

        log.info("DistractionDetector initialized (Simple hand-based)")

    def _load_config(self, path, fps):
        defaults = {
            # thresholds
            "yaw_deg": 45.0,
            "pitch_down_deg": 20.0,
            "pitch_up_deg": 20.0,
            # timing
            "hands_visible_sec": 1.5,
            "gaze_sec": 1.2,
            # smoothing
            "history_frames": 5,
            "required_frames": 3,
            "face_visibility_history_frames": 10,
            "face_present_min_samples": 5,
            "partial_face_threshold": 0.4,
            # severity
            "long_duration_high_sec": 3.0,
        }

        cfg = load_yaml_section(path, "detectors.distraction")
        thresholds = cfg.get("thresholds", {}) if isinstance(cfg.get("thresholds", {}), dict) else {}
        timing = cfg.get("timing", {}) if isinstance(cfg.get("timing", {}), dict) else {}
        smoothing = cfg.get("smoothing", {}) if isinstance(cfg.get("smoothing", {}), dict) else {}
        severity = cfg.get("severity", {}) if isinstance(cfg.get("severity", {}), dict) else {}

        return {
            "yaw_threshold": float(thresholds.get("yaw_deg", defaults["yaw_deg"])),
            "pitch_down_threshold": float(thresholds.get("pitch_down_deg", defaults["pitch_down_deg"])),
            "pitch_up_threshold": float(thresholds.get("pitch_up_deg", defaults["pitch_up_deg"])),
            "time_hands_visible": float(timing.get("hands_visible_sec", defaults["hands_visible_sec"])),
            "time_gaze": float(timing.get("gaze_sec", defaults["gaze_sec"])),
            "history_frames": int(smoothing.get("history_frames", defaults["history_frames"])),
            "required_frames": int(smoothing.get("required_frames", defaults["required_frames"])),
            "face_visibility_history_frames": int(
                smoothing.get("face_visibility_history_frames", defaults["face_visibility_history_frames"])
            ),
            "face_present_min_samples": int(smoothing.get("face_present_min_samples", defaults["face_present_min_samples"])),
            "partial_face_threshold": float(smoothing.get("partial_face_threshold", defaults["partial_face_threshold"])),
            "long_duration_high_sec": float(severity.get("long_duration_high_sec", defaults["long_duration_high_sec"])),
        }

    def set_face_visibility(self, face_detection_confidence):
        """
        Call this each frame with face detection confidence (0.0-1.0)
        Helps distinguish between "face turned away" vs "no face detected"
        """
        self.face_visibility_history.append(face_detection_confidence if face_detection_confidence else 0.0)

    def _is_face_present(self):
        """Check if face is at least partially visible"""
        if len(self.face_visibility_history) < self.face_present_min_samples:
            return True  # Benefit of doubt during startup

        recent_vis = list(self.face_visibility_history)[-self.face_present_min_samples :]
        avg_vis = sum(recent_vis) / len(recent_vis)
        return avg_vis >= self.partial_face_threshold

    def analyze(self, pitch, yaw, roll, hands=None, face=None, is_drowsy=False, is_fainting=False):
        """
        Simple logic for face+shoulders camera view with priority handling.

        Returns: (is_distracted, is_new_event, distraction_info)
        """
        # If fainting detected, don't report distraction
        if is_fainting:
            self.start_time = None
            self.is_distracted = False
            return False, False, None

        if not (abs(pitch) < 90 and abs(yaw) < 90):
            return False, False, None

        hands_visible = bool(hands and len(hands) > 0)
        dp = pitch - self.cal["pitch"]
        dy = abs(yaw - self.cal["yaw"])

        # Separate gaze violations based on direction
        looking_aside = dy > self.cfg["yaw_threshold"]
        looking_down = dp > self.cfg["pitch_down_threshold"]
        looking_up = dp < -self.cfg["pitch_up_threshold"]

        # If drowsy and looking down, don't count as distraction
        if is_drowsy and looking_down:
            self.start_time = None
            self.is_distracted = False
            return False, False, None

        violation_type = None
        time_threshold = None
        alert_detail = None
        base_severity = "Medium"

        if hands_visible:
            num_hands = len(hands)
            if num_hands == 1:
                violation_type = "ONE HAND OFF WHEEL"
                alert_detail = "One Hand Off Wheel"
                base_severity = "Medium"
            else:
                violation_type = "BOTH HANDS OFF WHEEL"
                alert_detail = "Both Hands Off Wheel"
                base_severity = "High"
            time_threshold = self.cfg["time_hands_visible"]
        elif looking_aside:
            violation_type = "LOOKING ASIDE"
            alert_detail = "Looking Away from Road"
            base_severity = "Medium"
            time_threshold = self.cfg["time_gaze"]
        elif looking_up:
            violation_type = "LOOKING UP"
            alert_detail = "Looking Up Away from Road"
            base_severity = "Medium"
            time_threshold = self.cfg["time_gaze"]
        elif looking_down and not is_drowsy:
            violation_type = "LOOKING DOWN"
            alert_detail = "Looking Down at Device"
            base_severity = "Medium"
            time_threshold = self.cfg["time_gaze"]

        # Track in history (from YAML)
        self.history.append(violation_type is not None)

        # Require N out of history_frames frames to have violation (from YAML)
        history_sum = sum(self.history)
        if history_sum >= self.cfg["required_frames"] and violation_type:
            if self.start_time is None or self.distraction_type != violation_type:
                self.start_time = time.time()
                self.distraction_type = violation_type

            elapsed = time.time() - self.start_time

            if elapsed >= time_threshold:
                if not self.is_distracted:
                    # New distraction event
                    self.is_distracted = True
                    self.metrics["total_distractions"] += 1

                    # Determine final severity based on type + duration (from YAML)
                    if "BOTH HANDS" in violation_type:
                        final_severity = "High"
                    elif elapsed >= self.cfg["long_duration_high_sec"]:
                        final_severity = "High"
                    else:
                        final_severity = base_severity

                    distraction_info = {"alert_detail": alert_detail, "severity": final_severity, "duration": elapsed}
                    return True, True, distraction_info
                else:
                    return True, False, None

            return False, False, None

        # No stable violation - reset
        self.start_time = None
        self.is_distracted = False
        self.distraction_type = "NORMAL"
        return False, False, None

    def get_status(self):
        return {"is_distracted": self.is_distracted, "type": self.distraction_type, "metrics": self.metrics}
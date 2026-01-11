import time
import logging
from collections import deque

from src.status.distraction.config import load_distraction_config
from src.status.distraction.rules import determine_violation, final_severity

log = logging.getLogger(__name__)


class DistractionDetector:
    def __init__(self, fps=30.0, camera_pitch=0.0, camera_yaw=0.0, config_path="config/detector_config.yaml"):
        self.fps = fps
        self.cfg = load_distraction_config(config_path)

        self.is_distracted = False
        self.distraction_type = "NORMAL"
        self.start_time = None

        self.cal = {"pitch": camera_pitch, "yaw": camera_yaw}

        self.history = deque(maxlen=int(self.cfg["smoothing"]["history_frames"]))
        self.metrics = {"total_distractions": 0}

        self.face_visibility_history = deque(maxlen=int(self.cfg["smoothing"]["face_visibility_history_frames"]))
        self.partial_face_threshold = float(self.cfg["smoothing"]["partial_face_threshold"])
        self.face_present_min_samples = int(self.cfg["smoothing"]["face_present_min_samples"])

        log.info("DistractionDetector initialized")

    def set_face_visibility(self, face_detection_confidence):
        self.face_visibility_history.append(face_detection_confidence if face_detection_confidence else 0.0)

    def _is_face_present(self):
        if len(self.face_visibility_history) < self.face_present_min_samples:
            return True
        recent_vis = list(self.face_visibility_history)[-self.face_present_min_samples :]
        return (sum(recent_vis) / len(recent_vis)) >= self.partial_face_threshold

    def analyze(self, pitch, yaw, roll, hands=None, face=None, is_drowsy=False, is_fainting=False):
        if is_fainting:
            self.start_time = None
            self.is_distracted = False
            self.distraction_type = "NORMAL"
            return False, False, None

        if not (abs(pitch) < 90 and abs(yaw) < 90):
            return False, False, None

        hands_count = len(hands) if hands else 0

        violation_type, time_threshold, alert_detail, base_sev = determine_violation(
            pitch=float(pitch),
            yaw=float(yaw),
            cal_pitch=float(self.cal["pitch"]),
            cal_yaw=float(self.cal["yaw"]),
            yaw_thr=float(self.cfg["thresholds"]["yaw_deg"]),
            pitch_down_thr=float(self.cfg["thresholds"]["pitch_down_deg"]),
            pitch_up_thr=float(self.cfg["thresholds"]["pitch_up_deg"]),
            hands_count=int(hands_count),
            is_drowsy=bool(is_drowsy),
            time_hands_visible=float(self.cfg["timing"]["hands_visible_sec"]),
            time_gaze=float(self.cfg["timing"]["gaze_sec"]),
        )

        self.history.append(violation_type is not None)

        if sum(self.history) >= int(self.cfg["smoothing"]["required_frames"]) and violation_type:
            if self.start_time is None or self.distraction_type != violation_type:
                self.start_time = time.time()
                self.distraction_type = violation_type

            elapsed = time.time() - self.start_time
            if elapsed >= float(time_threshold):
                if not self.is_distracted:
                    self.is_distracted = True
                    self.metrics["total_distractions"] += 1

                    sev = final_severity(
                        violation_type,
                        float(elapsed),
                        float(self.cfg["severity"]["long_duration_high_sec"]),
                        base_sev,
                    )
                    return True, True, {"alert_detail": alert_detail, "severity": sev, "duration": elapsed}

                return True, False, None

            return False, False, None

        self.start_time = None
        self.is_distracted = False
        self.distraction_type = "NORMAL"
        return False, False, None

    def get_status(self):
        return {"is_distracted": self.is_distracted, "type": self.distraction_type, "metrics": self.metrics}
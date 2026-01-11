import time
import logging
import math
import numpy as np
from collections import deque
from src.services.system_logger import SystemLogger
from src.utils.load_detector_config import load_yaml_section

class DrowsinessDetector:
    """
    Detects drowsiness (low EAR over time) and yawning (MAR/expr/hands).
    Fainting is not judged here; use FaintingDetector separately.
    """

    def __init__(self, logger: SystemLogger, fps: float = 30.0, config_path: str = "config/detector_config.yaml"):
        self.logger = logger
        self.fps = fps
        self.config = self._load_config(config_path)
        self.user = None

        # State Tracking
        self._last_frame_rgb = None

        # Use YAML-driven history length (seconds -> frames)
        self.ear_history = deque(maxlen=int(self.config["ear_history_frames"]))
        self.ear_drop_detected = False

        # Dynamic EAR thresholds (loaded from user profile or config)
        self.dynamic_ear_thresh = self.config["ear_low"]

        self.counters = {
            'DROWSINESS': 0, 'RECOVERY': 0, 'YAW': 0, 'YAW_COOL': 0,
            'BLINK': 0, 'EYES_CLOSED': 0, 'SMILE_SUP': 0, 'LAUGH_SUP': 0,
        }

        self.states = {'IS_DROWSY': False, 'IS_YAWNING': False, 'EYES_CLOSED': False}
        self.episode = {'active': False, 'start_time': None, 'min_ear': 1.0, 'start_frame': None}

        # Yawn frequency tracking
        self.yawn_timestamps = deque(maxlen=10)  # Track last 10 yawns
        self.yawn_count = 0
        
        logging.info("DrowsinessDetector initialized (Clean, no fainting)")

    def _load_config(self, path_str: str):
        # Defaults (fallback only)
        defaults = {
            "episode": {
                "start_threshold_sec": 0.5,
                "end_grace_sec": 1.5,
                "min_episode_sec": 2.0,
            },
            "ear": {
                "low_threshold": 0.22,
                "high_threshold": 0.26,
                "drop_threshold": 0.10,
                "drop_window_frames": 15,
                "history_sec": 1.0,
            },
            "blink": {"min_closed_frames": 1, "max_closed_frames": 10},
            "yawn": {
                "threshold_sec": 0.27,
                "cooldown_sec": 2.0,
                "hand_cover_distance_norm": 0.15,
                "frequency_window_sec": 120.0,
                "high_frequency_count": 3,
                "timestamps_max": 10,
            },
            "expression_suppression": {"smile_suppress_sec": 0.5, "laugh_suppress_sec": 0.67},
            "severity": {"drowsiness": {"medium_sec": 2.0, "high_sec": 3.0, "critical_sec": 5.0}},
        }

        root = load_yaml_section(path_str, "detectors.drowsiness")

        episode = root.get("episode", {}) if isinstance(root.get("episode", {}), dict) else {}
        ear = root.get("ear", {}) if isinstance(root.get("ear", {}), dict) else {}
        blink = root.get("blink", {}) if isinstance(root.get("blink", {}), dict) else {}
        yawn = root.get("yawn", {}) if isinstance(root.get("yawn", {}), dict) else {}
        sup = root.get("expression_suppression", {}) if isinstance(root.get("expression_suppression", {}), dict) else {}
        sev_root = root.get("severity", {}) if isinstance(root.get("severity", {}), dict) else {}
        sev = sev_root.get("drowsiness", {}) if isinstance(sev_root.get("drowsiness", {}), dict) else {}

        start_sec = float(episode.get("start_threshold_sec", defaults["episode"]["start_threshold_sec"]))
        end_sec = float(episode.get("end_grace_sec", defaults["episode"]["end_grace_sec"]))
        history_sec = float(ear.get("history_sec", defaults["ear"]["history_sec"]))

        return {
            # episode
            "drowsy_start": int(start_sec * self.fps),
            "drowsy_end": int(end_sec * self.fps),
            "min_episode_sec": float(episode.get("min_episode_sec", defaults["episode"]["min_episode_sec"])),

            # ear
            "ear_low": float(ear.get("low_threshold", defaults["ear"]["low_threshold"])),
            "ear_high": float(ear.get("high_threshold", defaults["ear"]["high_threshold"])),
            "ear_drop": float(ear.get("drop_threshold", defaults["ear"]["drop_threshold"])),
            "ear_drop_window_frames": int(ear.get("drop_window_frames", defaults["ear"]["drop_window_frames"])),
            "ear_history_frames": max(1, int(history_sec * self.fps)),

            # blink
            "blink_min_closed_frames": int(blink.get("min_closed_frames", defaults["blink"]["min_closed_frames"])),
            "blink_max_closed_frames": int(blink.get("max_closed_frames", defaults["blink"]["max_closed_frames"])),

            # yawn
            "yawn_thresh": int(float(yawn.get("threshold_sec", defaults["yawn"]["threshold_sec"])) * self.fps),
            "yawn_cool": int(float(yawn.get("cooldown_sec", defaults["yawn"]["cooldown_sec"])) * self.fps),
            "hand_cover_distance_norm": float(
                yawn.get("hand_cover_distance_norm", defaults["yawn"]["hand_cover_distance_norm"])
            ),
            "yawn_frequency_window_sec": float(
                yawn.get("frequency_window_sec", defaults["yawn"]["frequency_window_sec"])
            ),
            "yawn_high_frequency_count": int(
                yawn.get("high_frequency_count", defaults["yawn"]["high_frequency_count"])
            ),
            "yawn_timestamps_max": int(yawn.get("timestamps_max", defaults["yawn"]["timestamps_max"])),

            # expression suppression
            "smile_sup": int(float(sup.get("smile_suppress_sec", defaults["expression_suppression"]["smile_suppress_sec"])) * self.fps),
            "laugh_sup": int(float(sup.get("laugh_suppress_sec", defaults["expression_suppression"]["laugh_suppress_sec"])) * self.fps),

            # severity
            "sev_medium_sec": float(sev.get("medium_sec", defaults["severity"]["drowsiness"]["medium_sec"])),
            "sev_high_sec": float(sev.get("high_sec", defaults["severity"]["drowsiness"]["high_sec"])),
            "sev_critical_sec": float(sev.get("critical_sec", defaults["severity"]["drowsiness"]["critical_sec"])),
        }

    def set_last_frame(self, frame):
        self._last_frame_rgb = frame.copy() if frame is not None else None

    def set_active_user(self, user_profile):
        self.user = user_profile
        base_ear = user_profile.ear_threshold if user_profile else self.config['ear_low']
        self.dynamic_ear_thresh = base_ear
        self.config['ear_low'] = base_ear
        self.config['ear_high'] = base_ear * 1.2
        self._reset_state()

    def _reset_state(self):
        keys = [
            'DROWSINESS', 'RECOVERY', 'YAW', 'YAWN_COOL',
            'BLINK', 'EYES_CLOSED', 'SMILE_SUP', 'LAUGH_SUP'
        ]
        self.counters = {k: 0 for k in keys}
        self.episode['active'] = False
        self.states = {k: False for k in self.states}
        self.ear_history.clear()
        
    def detect(self, ear, mar, expression, hands_data=None, face_center=None, pitch=None):
        self.ear_history.append(ear)
        self._detect_sudden_ear_drop()
        self.states['EYES_CLOSED'] = ear < self.config['ear_low']

        # Update suppressions and events
        self._update_suppression(expression)
        self._update_blink(ear)
        self._update_drowsiness(ear)
        self._update_yawn(mar, expression, hands_data, face_center)

        if self.states['IS_DROWSY']:
            return "DROWSY", (0, 0, 255)
        if self.states['IS_YAWNING']:
            return "YAWNING", (0, 255, 255)
        return "NORMAL", (0, 255, 0)

    def _detect_sudden_ear_drop(self):
        if len(self.ear_history) < 15:
            return
        drop = self.ear_history[-15] - self.ear_history[-1]
        self.ear_drop_detected = (drop > self.config['ear_drop'])

    def _update_suppression(self, expr):
        if expr == "SMILE":
            self.counters['SMILE_SUP'] = self.config['smile_sup']
        elif expr == "LAUGH":
            self.counters['LAUGH_SUP'] = self.config['laugh_sup']
        else:
            if self.counters['SMILE_SUP'] > 0:
                self.counters['SMILE_SUP'] -= 1
            if self.counters['LAUGH_SUP'] > 0:
                self.counters['LAUGH_SUP'] -= 1

    def _update_drowsiness(self, ear):
        is_suppressed = self.counters['SMILE_SUP'] > 0 or self.counters['LAUGH_SUP'] > 0

        if self.episode['active']:
            self.episode['min_ear'] = min(self.episode['min_ear'], ear)
            if ear >= self.config['ear_high'] or is_suppressed:
                self.counters['RECOVERY'] += 1
                if self.counters['RECOVERY'] >= self.config['drowsy_end']:
                    dur = time.time() - self.episode['start_time']
                    if dur >= self.config['min_episode_sec']:
                        # Determine severity
                        if dur >= 5.0:
                            severity = "Critical"
                        elif dur >= 3.0:
                            severity = "High"
                        elif dur >= 2.0:
                            severity = "Medium"
                        else:
                            severity = "Low"
                        
                        # Updated call - removed technical fields
                        self.logger.log_event(
                            getattr(self.user, 'user_id', 0),
                            "DROWSY",
                            dur,
                            self.episode['min_ear'],  # Store EAR in 'value' field
                            self.episode['start_frame'],
                            alert_category="Drowsiness",
                            alert_detail="Eyes Closed Too Long",
                            severity=severity
                        )
                    self.episode['active'] = False
                    self.counters['DROWSINESS'] = 0
            else:
                self.counters['RECOVERY'] = 0
        elif ear < self.config['ear_low'] and not is_suppressed:
            self.counters['DROWSINESS'] += 1
            if self.counters['DROWSINESS'] >= self.config['drowsy_start']:
                self.episode.update({'active': True, 'start_time': time.time(), 'start_frame': self._last_frame_rgb, 'min_ear': 1.0})
                self.counters['RECOVERY'] = 0
        else:
            self.counters['DROWSINESS'] = 0

        self.states['IS_DROWSY'] = self.episode['active']

    def _update_blink(self, ear):
        """Track blinks for statistics."""
        if ear < self.dynamic_ear_thresh:
            self.counters['EYES_CLOSED'] += 1
        else:
            # Valid blink: eyes closed for 1-10 frames
            if 0 < self.counters['EYES_CLOSED'] < 10:
                self.counters['BLINK'] += 1
            self.counters['EYES_CLOSED'] = 0

    def _update_yawn(self, mar, expr, hands, face):
        if self.counters['YAWN_COOL'] > 0:
            self.counters['YAWN_COOL'] -= 1
            self.states['IS_YAWNING'] = False
            return

        covered = False
        if hands and face:
            for h in hands:
                if math.sqrt((h[12][0]-face[0])**2 + (h[12][1]-face[1])**2) < 0.15:
                    covered = True
                    break

        if (expr == "YAWN") or (covered and expr not in ["SMILE", "LAUGH"]):
            self.counters['YAWN'] += 1
        else:
            self.counters['YAWN'] = 0

        if self.counters['YAWN'] >= self.config['yawn_thresh']:
            if not self.states['IS_YAWNING']:
                now = time.time()
                self.yawn_timestamps.append(now)
                self.yawn_count += 1
                
                # Determine severity based on yawn frequency
                # High: 3+ yawns in last 2 minutes
                recent_yawns = sum(1 for t in self.yawn_timestamps if now - t < 120)
                
                if recent_yawns >= 3:
                    severity = "High"
                    alert_detail = f"Frequent Yawning ({recent_yawns} in 2 min)"
                    if covered:
                        alert_detail += " with Hand Near Face"
                elif covered:
                    severity = "Medium"
                    alert_detail = "Yawning with Hand Near Face"
                else:
                    severity = "Low"
                    alert_detail = "Yawning"
                
                # Updated call with dynamic severity
                self.logger.log_event(
                    getattr(self.user, 'user_id', 0),
                    "YAWN",
                    0.0,
                    mar,  # Store MAR in 'value' field
                    self._last_frame_rgb,
                    alert_category="Drowsiness",
                    alert_detail=alert_detail,
                    severity=severity
                )
                self.counters['YAWN_COOL'] = self.config['yawn_cool']
                self.states['IS_YAWNING'] = True

    def get_detailed_state(self):
        return {
            'is_drowsy': self.states['IS_DROWSY'],
            'is_yawning': self.states['IS_YAWNING'],
            'eyes_closed': self.states['EYES_CLOSED'],
            'blink_count': self.counters['BLINK'],
            'ear_trend': "STABLE"
        }
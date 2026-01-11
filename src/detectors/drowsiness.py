import time
import logging
import yaml
import math
import numpy as np
from collections import deque
from src.services.system_logger import SystemLogger

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
        self.ear_history = deque(maxlen=int(1.0 * fps))  # 1 second buffer
        self.ear_drop_detected = False

        # Dynamic EAR thresholds (loaded from user profile or config)
        self.dynamic_ear_thresh = self.config['ear_low']

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

    def _load_config(self, path_str):
        default = {
            'drowsy_start': int(0.8 * self.fps), 'drowsy_end': int(1.5 * self.fps),
            'min_episode_sec': 2.0, 'ear_low': 0.22, 'ear_high': 0.26, 'ear_drop': 0.10,
            'yawn_thresh': int(0.4 * self.fps), 'yawn_cool': int(3.0 * self.fps),
            'smile_sup': int(0.5 * self.fps), 'laugh_sup': int(1.0 * self.fps)
        }
        try:
            with open(path_str, 'r') as f:
                raw = yaml.safe_load(f) or {}
                d, y, e = raw.get('drowsiness', {}), raw.get('yawn', {}), raw.get('expression', {})
                return {
                    'drowsy_start': int(d.get('start_threshold_sec', 0.8) * self.fps),
                    'drowsy_end': int(d.get('end_grace_sec', 1.5) * self.fps),
                    'min_episode_sec': d.get('min_episode_sec', 2.0),
                    'ear_low': d.get('ear_low_threshold', 0.22),
                    'ear_high': d.get('ear_high_threshold', 0.26),
                    'ear_drop': d.get('ear_drop_threshold', 0.10),
                    'yawn_thresh': int(y.get('threshold_sec', 0.4) * self.fps),
                    'yawn_cool': int(y.get('cooldown_sec', 3.0) * self.fps),
                    'smile_sup': int(e.get('smile_suppress_sec', 0.5) * self.fps),
                    'laugh_sup': int(e.get('laugh_suppress_sec', 1.0) * self.fps),
                }
        except Exception as e:
            logging.warning(f"Config load error: {e}. Using defaults.")
            return default

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
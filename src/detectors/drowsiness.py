import time
import logging
import yaml
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from src.infrastructure.data.models import UserProfile
from src.services.system_logger import SystemLogger

@dataclass
class DetectorConfig:
    drowsy_start_frames: int
    drowsy_end_grace_frames: int
    min_episode_frames: int
    yawn_threshold_frames: int
    yawn_cooldown_frames: int
    smile_suppress_frames: int
    laugh_suppress_frames: int
    blink_closed_frames: int
    min_episode_sec: float
    
    @classmethod
    def from_yaml(cls, config_path: str, fps: float) -> 'DetectorConfig':
        path = Path(config_path)
        if not path.exists(): return cls.default(fps)
        try:
            with open(path, 'r') as f:
                cfg = yaml.safe_load(f)
            return cls(
                drowsy_start_frames=int(cfg['drowsiness']['start_threshold_sec'] * fps),
                drowsy_end_grace_frames=int(cfg['drowsiness']['end_grace_sec'] * fps),
                min_episode_frames=int(cfg['drowsiness']['min_episode_sec'] * fps),
                min_episode_sec=cfg['drowsiness']['min_episode_sec'],
                blink_closed_frames=cfg['blink']['closed_frames'],
                yawn_threshold_frames=int(cfg['yawn']['threshold_sec'] * fps),
                yawn_cooldown_frames=int(cfg['yawn']['cooldown_sec'] * fps),
                smile_suppress_frames=int(cfg['expression']['smile_suppress_sec'] * fps),
                laugh_suppress_frames=int(cfg['expression']['laugh_suppress_sec'] * fps),
            )
        except: return cls.default(fps)
    
    @classmethod
    def default(cls, fps):
        return cls(int(1.0*fps), int(0.5*fps), int(1.0*fps), 1.0, 2, int(0.27*fps), int(2.0*fps), int(0.5*fps), int(0.67*fps))

class DrowsinessDetector:
    def __init__(self, logger: SystemLogger, fps: float = 30.0, config_path: str = "config/detector_config.yaml"):
        self.logger = logger
        self.config = DetectorConfig.from_yaml(config_path, fps)
        self.user = None
        self.dynamic_ear_threshold = 0.22
        self._last_frame_rgb = None
        self.counters = {'DROWSINESS_FRAMES': 0, 'DROWSY_RECOVERY_FRAMES': 0, 'YAWN_FRAMES': 0, 'YAWN_COOLDOWN': 0, 'BLINK_COUNT': 0, 'EYE_CLOSED_FRAMES': 0, 'SMILE_SUPPRESS_COUNTER': 0, 'LAUGH_SUPPRESS_COUNTER': 0}
        self.episode = {'active': False, 'start_time': None, 'start_frame_rgb': None, 'min_ear': 1.0, 'max_ear': 0.0}
        self.states = {'IS_DROWSY': False, 'IS_YAWNING': False}

    def set_last_frame(self, frame, color_space="RGB"):
        if frame is not None: self._last_frame_rgb = frame.copy()

    def set_active_user(self, user_profile):
        self.user = user_profile
        self.dynamic_ear_threshold = user_profile.ear_threshold if user_profile else 0.22
        self.counters = {k: 0 for k in self.counters}
        self.episode['active'] = False
        self.states = {k: False for k in self.states}
        self.logger.stop_alert()

    def detect(self, current_ear, mar, mouth_expression):
        self._update_expression_suppression(mouth_expression)
        self._update_blink_counter(current_ear)
        self._update_drowsiness_episode(current_ear)
        self._update_yawn_state(mar, mouth_expression)
        
        if self.states['IS_DROWSY']:
            self.logger.alert("critical")
            return "DROWSY", (255, 0, 0)
        elif self.states['IS_YAWNING']:
            self.logger.alert("warning")
            return "YAWNING", (255, 255, 0)
        else:
            self.logger.stop_alert()
            return "NORMAL", (0, 255, 0)

    def _update_expression_suppression(self, expr):
        if expr == "SMILE": self.counters['SMILE_SUPPRESS_COUNTER'] = self.config.smile_suppress_frames
        elif expr == "LAUGH": self.counters['LAUGH_SUPPRESS_COUNTER'] = self.config.laugh_suppress_frames
        else: 
            self.counters['SMILE_SUPPRESS_COUNTER'] = max(0, self.counters['SMILE_SUPPRESS_COUNTER'] - 1)
            self.counters['LAUGH_SUPPRESS_COUNTER'] = max(0, self.counters['LAUGH_SUPPRESS_COUNTER'] - 1)

    def _update_drowsiness_episode(self, ear):
        # Precompute thresholds once per user, not every frame
        if not hasattr(self, '_ear_low'):
            self._ear_low = self.dynamic_ear_threshold * 1.0
            self._ear_high = self.dynamic_ear_threshold * 1.2

        is_suppressed = self.counters['SMILE_SUPPRESS_COUNTER'] > 0 or self.counters['LAUGH_SUPPRESS_COUNTER'] > 0
        low = ear < self._ear_low
        high = ear >= self._ear_high
        
        if self.episode['active']:
            self.episode['min_ear'] = min(self.episode['min_ear'], ear)
            if high or is_suppressed:
                self.counters['DROWSY_RECOVERY_FRAMES'] += 1
                if self.counters['DROWSY_RECOVERY_FRAMES'] >= self.config.drowsy_end_grace_frames:
                    duration = time.time() - self.episode['start_time']
                    if duration >= self.config.min_episode_sec:
                        self.logger.log_event(self.user.user_id if self.user else 0, "DROWSINESS_EPISODE", duration, self.episode['min_ear'], self.episode['start_frame_rgb'])
                    self.episode['active'] = False
            else: self.counters['DROWSY_RECOVERY_FRAMES'] = 0
        else:
            if low and not is_suppressed:
                self.counters['DROWSINESS_FRAMES'] += 1
                if self.counters['DROWSINESS_FRAMES'] >= self.config.drowsy_start_frames:
                    self.episode.update({'active': True, 'start_time': time.time(), 'start_frame_rgb': self._last_frame_rgb.copy() if self._last_frame_rgb is not None else None, 'min_ear': 1.0})
            else: self.counters['DROWSINESS_FRAMES'] = 0
        self.states['IS_DROWSY'] = self.episode['active']

    def _update_blink_counter(self, ear):
        if ear < self.dynamic_ear_threshold: self.counters['EYE_CLOSED_FRAMES'] += 1
        else:
            if self.counters['EYE_CLOSED_FRAMES'] >= self.config.blink_closed_frames: self.counters['BLINK_COUNT'] += 1
            self.counters['EYE_CLOSED_FRAMES'] = 0

    def _update_yawn_state(self, mar, expr):
        if self.counters['YAWN_COOLDOWN'] > 0: self.counters['YAWN_COOLDOWN'] -= 1
        self.counters['YAWN_FRAMES'] = (self.counters['YAWN_FRAMES'] + 1) if expr == "YAWN" else 0
        is_yawning = self.counters['YAWN_FRAMES'] >= self.config.yawn_threshold_frames
        
        if is_yawning and not self.states['IS_YAWNING'] and self.counters['YAWN_COOLDOWN'] == 0:
            self.logger.log_event(self.user.user_id if self.user else 0, "YAWN", 0.0, mar, self._last_frame_rgb)
            self.counters['YAWN_COOLDOWN'] = self.config.yawn_cooldown_frames
        self.states['IS_YAWNING'] = is_yawning
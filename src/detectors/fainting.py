import time
import logging
import math
import yaml
import numpy as np
from collections import deque

log = logging.getLogger(__name__)

class FaintingDetector:
    """
    Robust fainting detector using combined signals:
    - Extreme pitch relative to camera calibration
    - Face visibility hidden while shoulders visible (classic slump/face-away)
    - Downward face trend (slump) over a short window
    - Excludes phone posture with demotions
    """

    def __init__(self, fps: float = 30.0, camera_pitch: float = 0.0, config_path: str = "config/detector_config.yaml"):
        self.fps = fps
        self.cfg = self._load_config(config_path, fps)
        self.cal = {'pitch': camera_pitch}

        # State
        self.is_fainting = False
        self.faint_probability = 0.0
        self.metrics = {'extreme_pitch_frames': 0, 'pitch_stable_frames': 0, 'total_faints': 0}

        # Histories
        win = self.cfg['history_frames']
        self.pitch_history = deque(maxlen=win)
        self.face_y_history = deque(maxlen=win)

        # Context (updated per frame)
        self.context = {'drowsy': False, 'eyes_closed': False, 'phone': False, 'wheel': 0}

        # Visibility and smoothing
        self.pose_vis = {'face': None, 'l_sh': None, 'r_sh': None}
        self.face_hidden_hist = deque(maxlen=int(self.cfg['vis_window_sec'] * fps))
        self.shoulders_visible_hist = deque(maxlen=int(self.cfg['vis_window_sec'] * fps))

        log.info("FaintingDetector initialized (robust)")

    def _load_config(self, path: str, fps: float):
        defaults = {
            'pitch_extreme': 55.0,          # INCREASED from 45.0 to reduce false positives
            'rapid_change': 15.0,           # INCREASED from 12.0
            'history_sec': 0.8,             # INCREASED from 0.6 for more stability
            'vis_window_sec': 1.0,          # INCREASED from 0.6 for more stability
            'slump_trend_min': 0.008,       # INCREASED from 0.006
            'min_confirm_sec': 1.2,         # INCREASED from 0.8
            'phone_pitch_min': 18.0,
            'phone_pitch_max': 38.0,
            'var_stable_max': 5.0,
            'var_unstable_min': 10.0,
            'prob_threshold': 0.75,         # INCREASED from 0.7
            'vis_face_thr': 0.40,
            'vis_sh_thr': 0.35
        }
        try:
            with open(path, 'r') as f:
                raw = yaml.safe_load(f) or {}
                faint = raw.get('fainting', {})
                return {
                    'pitch_extreme': faint.get('pitch_extreme', defaults['pitch_extreme']),
                    'rapid_change': faint.get('rapid_change', defaults['rapid_change']),
                    'history_frames': int(faint.get('history_sec', defaults['history_sec']) * fps),
                    'vis_window_sec': faint.get('vis_window_sec', defaults['vis_window_sec']),
                    'slump_trend_min': faint.get('face_slump_trend_min', defaults['slump_trend_min']),
                    'min_confirm_frames': int(faint.get('min_confirm_sec', defaults['min_confirm_sec']) * fps),
                    'phone_pitch_min': faint.get('phone_pitch_min', defaults['phone_pitch_min']),
                    'phone_pitch_max': faint.get('phone_pitch_max', defaults['phone_pitch_max']),
                    'var_stable_max': faint.get('var_stable_max', defaults['var_stable_max']),
                    'var_unstable_min': faint.get('var_unstable_min', defaults['var_unstable_min']),
                    'prob_threshold': faint.get('prob_threshold', defaults['prob_threshold']),
                    'vis_face_thr': faint.get('vis_face_thr', defaults['vis_face_thr']),
                    'vis_sh_thr': faint.get('vis_sh_thr', defaults['vis_sh_thr']),
                }
        except Exception as e:
            log.warning(f"Failed to load fainting config: {e}")
            return {
                'pitch_extreme': defaults['pitch_extreme'],
                'rapid_change': defaults['rapid_change'],
                'history_frames': int(defaults['history_sec'] * fps),
                'vis_window_sec': defaults['vis_window_sec'],
                'slump_trend_min': defaults['slump_trend_min'],
                'min_confirm_frames': int(defaults['min_confirm_sec'] * fps),
                'phone_pitch_min': defaults['phone_pitch_min'],
                'phone_pitch_max': defaults['phone_pitch_max'],
                'var_stable_max': defaults['var_stable_max'],
                'var_unstable_min': defaults['var_unstable_min'],
                'prob_threshold': defaults['prob_threshold'],
                'vis_face_thr': defaults['vis_face_thr'],
                'vis_sh_thr': defaults['vis_sh_thr'],
            }

    def set_context(self, drowsy: bool, eyes_closed: bool, phone: bool, wheel_count: int):
        self.context.update({'drowsy': drowsy, 'eyes_closed': eyes_closed, 'phone': phone, 'wheel': wheel_count})

    def set_pose_visibility(self, face_vis: float | None, left_sh_vis: float | None, right_sh_vis: float | None):
        self.pose_vis['face'] = face_vis
        self.pose_vis['l_sh'] = left_sh_vis
        self.pose_vis['r_sh'] = right_sh_vis

    def _median_slope(self, ys):
        if len(ys) < 5:
            return 0.0
        x = np.arange(len(ys))
        # robust slope via Theil-Sen (approx with median of pairwise slopes)
        slopes = []
        for i in range(len(ys) - 1):
            for j in range(i + 1, len(ys)):
                dy = ys[j] - ys[i]
                dx = j - i
                if dx != 0:
                    slopes.append(dy / dx)
        return float(np.median(slopes)) if slopes else 0.0

    def _indicators(self, pitch: float, face_center):
        inds = {'rapid': False, 'extreme': False, 'slump': False, 'phone_pose': False,
                'face_hidden': False, 'shoulders_visible': False}

        # Update histories
        self.pitch_history.append(pitch)
        if face_center:
            self.face_y_history.append(face_center[1])

        # Rapid pitch change across history window
        if len(self.pitch_history) >= 2:
            if (pitch - self.pitch_history[0]) > self.cfg['rapid_change']:
                inds['rapid'] = True

        # Extreme pitch relative to calibration
        delta = pitch - self.cal['pitch']
        if delta > self.cfg['pitch_extreme']:
            inds['extreme'] = True
            self.metrics['extreme_pitch_frames'] += 1
        else:
            self.metrics['extreme_pitch_frames'] = 0

        # Phone-like stable pitch posture
        if len(self.pitch_history) >= 10:
            segment = list(self.pitch_history)[-10:]
            var = np.var(segment)
            if self.cfg['phone_pitch_min'] < delta < self.cfg['phone_pitch_max'] and var < self.cfg['var_stable_max']:
                inds['phone_pose'] = True

        # Slump trend (robust)
        if len(self.face_y_history) >= int(0.6 * self.fps):
            slope = self._median_slope(list(self.face_y_history))
            if slope > self.cfg['slump_trend_min']:
                inds['slump'] = True

        # Visibility gates
        fv = self.pose_vis['face']
        lv = self.pose_vis['l_sh']
        rv = self.pose_vis['r_sh']
        vis_face_thr = self.cfg['vis_face_thr']
        vis_sh_thr = self.cfg['vis_sh_thr']

        face_hidden = (fv is None) or (fv < vis_face_thr)
        shoulders_visible = ((lv is not None and lv >= vis_sh_thr) or (rv is not None and rv >= vis_sh_thr))

        self.face_hidden_hist.append(1 if face_hidden else 0)
        self.shoulders_visible_hist.append(1 if shoulders_visible else 0)

        if len(self.face_hidden_hist) == self.face_hidden_hist.maxlen:
            inds['face_hidden'] = sum(self.face_hidden_hist) >= int(0.7 * self.face_hidden_hist.maxlen)
        if len(self.shoulders_visible_hist) == self.shoulders_visible_hist.maxlen:
            inds['shoulders_visible'] = sum(self.shoulders_visible_hist) >= int(0.7 * self.shoulders_visible_hist.maxlen)

        return inds

    def _score(self, inds):
        # Base score from strong indicators
        score = 0.0
        if inds['extreme'] and inds['face_hidden'] and inds['shoulders_visible']:
            score += 0.55  # strong slump pattern
        if inds['rapid'] and self.context['eyes_closed']:
            score += 0.25
        if inds['slump']:
            score += 0.20
        if self.context['drowsy']:
            score += 0.10

        # Demotions for phone posture
        if inds['phone_pose'] and self.context['phone']:
            score -= 0.40
        if self.context['phone'] and not self.context['eyes_closed']:
            score -= 0.25

        # Minimum duration requirement
        return max(0.0, min(1.0, score))

    def analyze(self, pitch: float, yaw: float, roll: float, hands=None, face_center=None):
        # sanity on angles
        if not (abs(pitch) < 90 and abs(yaw) < 90):
            return False, False, None

        inds = self._indicators(pitch, face_center)
        score = self._score(inds)

        # Require indicators to persist for a minimum duration
        confirm_frames = self.cfg['min_confirm_frames']
        sustained = (
            self.metrics['extreme_pitch_frames'] >= min(confirm_frames, 5) and
            inds['face_hidden'] and inds['shoulders_visible']
        )

        # Combine sustained gate and score
        self.faint_probability = 0.5 * score + (0.5 if sustained else 0.0)

        if self.faint_probability > self.cfg['prob_threshold']:
            if not self.is_fainting:
                self.is_fainting = True
                self.metrics['total_faints'] += 1
                log.critical(f"FAINT DETECTED (prob={self.faint_probability:.2f})")
                
                # Fainting is ALWAYS Critical - no duration needed
                fainting_info = {
                    'probability': self.faint_probability,
                    'severity': 'Critical',  # Always critical
                    'alert_detail': 'Possible Fainting or Collapse'
                }
                return True, True, fainting_info
            return True, False, None
        else:
            self.is_fainting = False
            return False, False, None

    def get_status(self):
        return {'is_fainting': self.is_fainting, 'faint_prob': self.faint_probability, 'metrics': self.metrics}
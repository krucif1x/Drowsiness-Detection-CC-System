import time
import logging
import yaml
from collections import deque

log = logging.getLogger(__name__)

class DistractionDetector:
    """
    Simple camera-appropriate distraction detector:
    - Camera shows face + shoulders only (hands normally NOT visible)
    - If ANY hand appears in frame = driver not holding wheel properly
    - Also checks extreme head angles (looking away)
    """

    def __init__(self, fps=30.0, camera_pitch=0.0, camera_yaw=0.0, config_path="config/detector_config.yaml"):
        self.cfg = self._load_config(config_path, fps)
        self.fps = fps

        # State
        self.is_distracted = False
        self.distraction_type = "NORMAL"
        self.start_time = None

        # Calibration
        self.cal = {'pitch': camera_pitch, 'yaw': camera_yaw}

        # History: track violations over last N frames
        self.history = deque(maxlen=5)

        # Metrics
        self.metrics = {'total_distractions': 0}

        # NEW: Track partial face visibility
        self.face_visibility_history = deque(maxlen=10)
        self.partial_face_threshold = 0.4  # 40% of face visible is enough

        log.info("DistractionDetector initialized (Simple hand-based)")

    def _load_config(self, path, fps):
        defaults = {
            'yaw_threshold': 45.0,           # degrees - extreme head turn
            'pitch_down_threshold': 20.0,    # degrees - looking way down
            'pitch_up_threshold': 20.0,      # degrees - looking way up
            'time_hands_visible': 1.5,       # seconds with hands visible before unsafe
            'time_gaze': 1.2,                # seconds looking away (reduced from 2.5)
        }
        try:
            with open(path, 'r') as f:
                raw = yaml.safe_load(f) or {}
                dist = raw.get('distraction', {})
                return {
                    'yaw_threshold': dist.get('yaw_threshold', defaults['yaw_threshold']),
                    'pitch_down_threshold': dist.get('pitch_down_threshold', defaults['pitch_down_threshold']),
                    'pitch_up_threshold': dist.get('pitch_up_threshold', defaults['pitch_up_threshold']),
                    'time_hands_visible': dist.get('time_hands_visible', defaults['time_hands_visible']),
                    'time_gaze': dist.get('time_gaze', defaults['time_gaze']),
                }
        except Exception as e:
            log.warning(f"Failed to load distraction config: {e}, using defaults")
            return defaults

    def set_face_visibility(self, face_detection_confidence):
        """
        Call this each frame with face detection confidence (0.0-1.0)
        Helps distinguish between "face turned away" vs "no face detected"
        """
        self.face_visibility_history.append(face_detection_confidence if face_detection_confidence else 0.0)

    def _is_face_present(self):
        """Check if face is at least partially visible"""
        if len(self.face_visibility_history) < 5:
            return True  # Benefit of doubt during startup
        
        recent_vis = list(self.face_visibility_history)[-5:]
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

        hands_visible = hands and len(hands) > 0
        dp = pitch - self.cal['pitch']
        dy = abs(yaw - self.cal['yaw'])
        
        # Separate gaze violations based on direction
        looking_aside = dy > self.cfg['yaw_threshold']
        looking_down = dp > self.cfg['pitch_down_threshold']
        looking_up = dp < -self.cfg['pitch_up_threshold']
        
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
                base_severity = "High"  # Both hands always High
            time_threshold = self.cfg['time_hands_visible']
        elif looking_aside:
            violation_type = "LOOKING ASIDE"
            alert_detail = "Looking Away from Road"
            base_severity = "Medium"
            time_threshold = self.cfg['time_gaze']
        elif looking_up:
            violation_type = "LOOKING UP"
            alert_detail = "Looking Up Away from Road"
            base_severity = "Medium"
            time_threshold = self.cfg['time_gaze']
        elif looking_down and not is_drowsy:
            violation_type = "LOOKING DOWN"
            alert_detail = "Looking Down at Device"
            base_severity = "Medium"
            time_threshold = self.cfg['time_gaze']

        # Track in history
        self.history.append(violation_type is not None)

        # Require 3 out of 5 frames to have violation
        history_sum = sum(self.history)
        if history_sum >= 3 and violation_type:
            if self.start_time is None or self.distraction_type != violation_type:
                self.start_time = time.time()
                self.distraction_type = violation_type
            
            elapsed = time.time() - self.start_time
            
            if elapsed >= time_threshold:
                if not self.is_distracted:
                    # New distraction event
                    self.is_distracted = True
                    self.metrics['total_distractions'] += 1
                    
                    # Determine final severity based on type + duration
                    if "BOTH HANDS" in violation_type:
                        final_severity = "High"  # Always high
                    elif elapsed >= 3.0:
                        final_severity = "High"  # Long duration
                    else:
                        final_severity = base_severity
                    
                    # Return info dict for logging
                    distraction_info = {
                        'alert_detail': alert_detail,
                        'severity': final_severity,
                        'duration': elapsed
                    }
                    return True, True, distraction_info
                else:
                    # Ongoing distraction
                    return True, False, None
            
            return False, False, None
        else:
            # No stable violation - reset
            self.start_time = None
            self.is_distracted = False
            self.distraction_type = "NORMAL"
            return False, False, None

    def get_status(self):
        return {
            'is_distracted': self.is_distracted,
            'type': self.distraction_type,
            'metrics': self.metrics
        }
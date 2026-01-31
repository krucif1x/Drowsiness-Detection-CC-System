"""
HEAD POSE ESTIMATOR - YOUR WORKING VERSION + CAMERA SPECS
Minimal changes to your original working code.
"""

import cv2
import numpy as np
import logging
from src.utils.landmarks.constants import MODEL_POINTS, LANDMARK_INDICES

logger = logging.getLogger(__name__)

class HeadPoseEstimator:
    """Simple head pose estimator with camera specs."""
    
    def __init__(self, camera_specs=None):
        """
        Args:
            camera_specs: Optional dict with 'focal_mm', 'sensor_w_mm', 'sensor_h_mm'
                         If None, uses simple focal_length = img_w approximation
        """
        self.model_points = MODEL_POINTS
        self.landmark_indices = LANDMARK_INDICES
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))
        
        # Camera specs (optional - if None, will use simple approximation)
        self.camera_specs = camera_specs or {
            "focal_mm": 4.74,
            "sensor_w_mm": 6.45,
            "sensor_h_mm": 3.63
        }
        self.use_camera_specs = camera_specs is not None
        
        # PnP state
        self.rvec = None
        self.tvec = None
        
        # Smoothing state
        self.prev_pitch = 0.0
        self.prev_yaw = 0.0
        self.prev_roll = 0.0
        self.first_frame = True
        
        # Smoothing parameters
        self.DEADZONE_THRESH = 0.5
        self.ALPHA_PITCH = 0.3
        self.ALPHA_YAW = 0.5
        self.ALPHA_ROLL = 0.3
        
        logger.info("HeadPoseEstimator initialized")
        
    def _unwrap_angle(self, prev_deg: float, curr_deg: float, period: float = 180.0, threshold: float = 90.0) -> float:
        """
        Enforce continuity by shifting curr_deg by ±period to minimize jump vs prev_deg.
        Use period=180 because RQDecomp can switch branches by ~180 deg.
        """
        delta = curr_deg - prev_deg
        if delta > threshold:
            curr_deg -= period
        elif delta < -threshold:
            curr_deg += period
        return curr_deg

    def calculate_pose(self, face_landmarks, img_w, img_h):
        """Calculate head pose angles."""
        try:
            # Initialize camera matrix
            if self.camera_matrix is None:
                if self.use_camera_specs:
                    # Use accurate camera specs
                    focal_length_x = (self.camera_specs["focal_mm"] / self.camera_specs["sensor_w_mm"]) * img_w
                    focal_length_y = (self.camera_specs["focal_mm"] / self.camera_specs["sensor_h_mm"]) * img_h
                    center = (img_w / 2, img_h / 2)
                    self.camera_matrix = np.array([
                        [focal_length_x, 0, center[0]],
                        [0, focal_length_y, center[1]],
                        [0, 0, 1]
                    ], dtype=np.float64)
                    logger.info(f"Camera matrix: fx={focal_length_x:.2f}, fy={focal_length_y:.2f}")
                else:
                    # Simple approximation (your original)
                    focal_length = img_w
                    center = (img_w / 2, img_h / 2)
                    self.camera_matrix = np.array([
                        [focal_length, 0, center[0]],
                        [0, focal_length, center[1]],
                        [0, 0, 1]
                    ], dtype=np.float64)
            
            # Get 2D image points
            image_points = np.array([
                (face_landmarks.landmark[i].x * img_w,
                 face_landmarks.landmark[i].y * img_h)
                for i in self.landmark_indices
            ], dtype=np.float64)
            
            # Solve PnP
            if self.rvec is None:
                success, self.rvec, self.tvec = cv2.solvePnP(
                    self.model_points, image_points,
                    self.camera_matrix, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
            else:
                success, self.rvec, self.tvec = cv2.solvePnP(
                    self.model_points, image_points,
                    self.camera_matrix, self.dist_coeffs,
                    rvec=self.rvec, tvec=self.tvec,
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
            
            if not success:
                return (self.prev_pitch, self.prev_yaw, self.prev_roll)
            
            # Convert to rotation matrix
            rmat, _ = cv2.Rodrigues(self.rvec)
            
            # Extract Euler angles using OpenCV
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            
            pitch_raw = angles[0]
            yaw_raw = angles[1]
            roll_raw = angles[2]
            
            # Normalize roll to [-90, +90] range
            if roll_raw > 90:
                roll_raw = roll_raw - 180
            elif roll_raw < -90:
                roll_raw = roll_raw + 180
            
            # First frame: no smoothing
            if self.first_frame:
                self.prev_pitch = pitch_raw
                self.prev_yaw = yaw_raw
                self.prev_roll = roll_raw
                self.first_frame = False
                return (pitch_raw, yaw_raw, roll_raw)
            
            # Apply deadzone (ignore small jitter)
            if abs(pitch_raw - self.prev_pitch) < self.DEADZONE_THRESH:
                pitch_raw = self.prev_pitch
            if abs(yaw_raw - self.prev_yaw) < self.DEADZONE_THRESH:
                yaw_raw = self.prev_yaw
            if abs(roll_raw - self.prev_roll) < self.DEADZONE_THRESH:
                roll_raw = self.prev_roll
            
            # Clamp to prevent impossible angles
            pitch_raw = max(-90, min(90, pitch_raw))
            yaw_raw = max(-90, min(90, yaw_raw))
            roll_raw = max(-90, min(90, roll_raw))
            
            # Apply exponential moving average
            smooth_pitch = (self.ALPHA_PITCH * pitch_raw) + ((1 - self.ALPHA_PITCH) * self.prev_pitch)
            smooth_yaw = (self.ALPHA_YAW * yaw_raw) + ((1 - self.ALPHA_YAW) * self.prev_yaw)
            smooth_roll = (self.ALPHA_ROLL * roll_raw) + ((1 - self.ALPHA_ROLL) * self.prev_roll)
            
            # Update state
            self.prev_pitch = smooth_pitch
            self.prev_yaw = smooth_yaw
            self.prev_roll = smooth_roll
            
            return (smooth_pitch, smooth_yaw, smooth_roll)
        
        except Exception as e:
            logger.error(f"Pose calculation error: {e}")
            return (self.prev_pitch, self.prev_yaw, self.prev_roll)

    def reset(self):
        """Reset estimator state."""
        self.rvec = None
        self.tvec = None
        self.prev_pitch = 0.0
        self.prev_yaw = 0.0
        self.prev_roll = 0.0
        self.first_frame = True
        logger.info("Head pose estimator reset")
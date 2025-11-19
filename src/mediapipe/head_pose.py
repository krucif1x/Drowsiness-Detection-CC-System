import cv2
import numpy as np
import logging

log = logging.getLogger(__name__)


MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # 1. Nose tip (0, 0, 0)
    (0.0, 63.6, -12.0),          # 2. Chin (Y=-63.6 becomes Y=+63.6)
    (-45.0, -17.0, -20.0),       # 3. Left Eye (Y=+17.0 becomes Y=-17.0)
    (45.0, -17.0, -20.0),        # 4. Right Eye (Y=+17.0 becomes Y=-17.0)
    (-30.0, 50.0, -12.0),        # 5. Left Mouth Corner (Y=-50.0 becomes Y=+50.0)
    (30.0, 50.0, -12.0)          # 6. Right Mouth Corner (Y=-50.0 becomes Y=+50.0)
], dtype=np.float64)

LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]

class HeadPoseEstimator:
    def __init__(self):
        self.model_points = MODEL_POINTS
        self.landmark_indices = LANDMARK_INDICES
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))
        
        # --- SMOOTHING STATE ---
        self.rvec = None
        self.tvec = None
        self.prev_pitch = 0.0
        self.prev_yaw = 0.0
        self.prev_roll = 0.0
        self.first_frame = True
        
        self.DEADZONE_THRESH = 1.5 
        
        # Smoothing Factors (Strong smoothing for Pitch/Roll)
        self.ALPHA = 0.4 
        self.ALPHA_YAW = 0.7 # Keep yaw responsive

        log.info("HeadPoseEstimator initialized (Y-Axis Flipped + Stabilized)")

    def calculate_pose(self, face_landmarks, img_w, img_h):
        try:
            # 1. Init Camera Matrix
            if self.camera_matrix is None:
                focal_length = img_w
                center = (img_w / 2, img_h / 2)
                self.camera_matrix = np.array([
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1]
                ], dtype=np.float64)

            # 2. Get 2D Image Points
            image_points = np.array([
                (face_landmarks.landmark[i].x * img_w, 
                 face_landmarks.landmark[i].y * img_h)
                for i in self.landmark_indices
            ], dtype=np.float64)

            # 3. Solve PnP with EXTRINSIC GUESS (Stabilization)
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

            # 4. Convert to Euler Angles (RQDecomp3x3)
            rmat, _ = cv2.Rodrigues(self.rvec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            
            pitch_raw = angles[0]
            yaw_raw   = angles[1]
            roll_raw  = angles[2]

            # 5. STABILIZATION LOGIC (Deadzone + EMA)
            if self.first_frame:
                self.prev_pitch, self.prev_yaw, self.prev_roll = pitch_raw, yaw_raw, roll_raw
                self.first_frame = False
                return (pitch_raw, yaw_raw, roll_raw)

            # A. Deadzone Check (Ignore small changes)
            if abs(pitch_raw - self.prev_pitch) < self.DEADZONE_THRESH: pitch_raw = self.prev_pitch
            if abs(yaw_raw - self.prev_yaw) < self.DEADZONE_THRESH:     yaw_raw = self.prev_yaw
            if abs(roll_raw - self.prev_roll) < self.DEADZONE_THRESH:   roll_raw = self.prev_roll

            # B. Exponential Moving Average (Use ALPHA_YAW for yaw, standard ALPHA for others)
            alpha_pitch = self.ALPHA
            alpha_yaw = self.ALPHA_YAW

            smooth_pitch = (alpha_pitch * pitch_raw) + ((1 - alpha_pitch) * self.prev_pitch)
            smooth_yaw   = (alpha_yaw   * yaw_raw)   + ((1 - alpha_yaw)   * self.prev_yaw)
            smooth_roll  = (alpha_pitch * roll_raw)  + ((1 - alpha_pitch) * self.prev_roll)

            # Update State
            self.prev_pitch = smooth_pitch
            self.prev_yaw   = smooth_yaw
            self.prev_roll  = smooth_roll
            
            return (smooth_pitch, smooth_yaw, smooth_roll)

        except Exception as e:
            log.error(f"Pose calculation error: {e}")
            return (0.0, 0.0, 0.0)
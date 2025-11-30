"""
HEAD POSE ESTIMATOR - ENHANCED VERSION
Based on your working code, with improvements for:
- Accurate camera calibration (Logitech C920x specs)
- Extended angle range
- Better Euler angle extraction
- EAR calculation with foreshortening compensation
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


# Proven 3D face model (from your working code)
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, 330.0, -65.0),         # Chin
    (-225.0, -170.0, -135.0),    # Left eye corner
    (225.0, -170.0, -135.0),     # Right eye corner
    (-150.0, 150.0, -125.0),     # Left mouth corner
    (150.0, 150.0, -125.0)       # Right mouth corner
], dtype=np.float64)

LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]

# Eye landmarks for EAR
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


class HeadPoseEstimator:
    """
    Enhanced head pose estimator with accurate camera calibration.
    
    Improvements over original:
    1. Uses real camera specs (Logitech C920x: 3.67mm focal, 4.8×3.6mm sensor)
    2. Better Euler angle extraction (proper trigonometry)
    3. Extended angle range (-80° to +80° instead of -50° to +50°)
    4. EAR calculation with foreshortening compensation
    5. Baseline calibration support
    """
    
    def __init__(self, camera_type="logitech_c920x", use_accurate_camera=True):
        """
        Initialize head pose estimator.
        
        Args:
            camera_type: "logitech_c920x" or "generic"
            use_accurate_camera: If True, use real camera specs. If False, use approximation.
        """
        self.model_points = MODEL_POINTS
        self.landmark_indices = LANDMARK_INDICES
        self.camera_type = camera_type
        self.use_accurate_camera = use_accurate_camera
        
        # Camera specs for Logitech C920x
        self.camera_specs = {
            "logitech_c920x": {
                "focal_mm": 3.67,
                "sensor_w_mm": 4.8,
                "sensor_h_mm": 3.6
            },
            "generic": {
                "focal_mm": None,  # Will use approximation
                "sensor_w_mm": None,
                "sensor_h_mm": None
            }
        }
        
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))
        
        # PnP state
        self.rvec = None
        self.tvec = None
        
        # Smoothing state
        self.prev_pitch = 0.0
        self.prev_yaw = 0.0
        self.prev_roll = 0.0
        self.prev_ear = 0.3
        self.first_frame = True
        
        # Smoothing parameters (your proven values)
        self.DEADZONE_THRESH = 0.5
        self.ALPHA_PITCH = 0.3
        self.ALPHA_YAW = 0.5
        self.ALPHA_ROLL = 0.3
        self.ALPHA_EAR = 0.4
        
        # Baseline calibration
        self.baseline_pitch = 0.0
        self.baseline_yaw = 0.0
        self.baseline_roll = 0.0
        self.baseline_samples = []
        self.baseline_enabled = False
        
        logger.info(f"HeadPoseEstimator initialized ({camera_type}, accurate={use_accurate_camera})")

    def _build_camera_matrix(self, img_w, img_h):
        """Build camera matrix with accurate or approximated focal length."""
        if self.use_accurate_camera and self.camera_type in self.camera_specs:
            specs = self.camera_specs[self.camera_type]
            if specs["focal_mm"] is not None:
                # Calculate accurate focal length in pixels
                focal_x = (specs["focal_mm"] / specs["sensor_w_mm"]) * img_w
                focal_y = (specs["focal_mm"] / specs["sensor_h_mm"]) * img_h
                logger.info(f"Using accurate camera matrix: fx={focal_x:.1f}px, fy={focal_y:.1f}px")
            else:
                # Fallback to approximation
                focal_x = focal_y = img_w
                logger.info(f"Using approximated focal length: {img_w}px")
        else:
            # Original approximation method
            focal_x = focal_y = img_w
            logger.info(f"Using approximated focal length: {img_w}px")
        
        center_x = img_w / 2.0
        center_y = img_h / 2.0
        
        return np.array([
            [focal_x, 0, center_x],
            [0, focal_y, center_y],
            [0, 0, 1]
        ], dtype=np.float64)

    def rotation_matrix_to_euler_angles(self, R):
        """
        Extract Euler angles from rotation matrix.
        
        This uses standard ZYX convention which matches cv2.solvePnP output.
        More accurate than RQDecomp3x3 for head pose.
        """
        # Check for gimbal lock
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        
        singular = sy < 1e-6
        
        if not singular:
            # Standard case
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
            roll = np.arctan2(R[2, 1], R[2, 2])
        else:
            # Gimbal lock (pitch near ±90°)
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(-R[0, 1], R[1, 1])
            roll = 0
        
        # Convert to degrees
        return np.degrees(pitch), np.degrees(yaw), np.degrees(roll)

    def calculate_ear(self, face_landmarks, img_w, img_h, yaw_angle):
        """
        Calculate Eye Aspect Ratio with foreshortening compensation.
        
        Args:
            face_landmarks: MediaPipe face mesh results
            img_w, img_h: Image dimensions
            yaw_angle: Current yaw angle in degrees
        
        Returns:
            dict with raw_ear, compensated_ear, and per-eye values
        """
        def get_eye_points(eye_indices):
            return np.array([
                [face_landmarks.landmark[i].x * img_w,
                 face_landmarks.landmark[i].y * img_h]
                for i in eye_indices
            ])
        
        def compute_ear(eye_points):
            # Two vertical distances
            v1 = np.linalg.norm(eye_points[1] - eye_points[5])
            v2 = np.linalg.norm(eye_points[2] - eye_points[4])
            
            # One horizontal distance
            h = np.linalg.norm(eye_points[0] - eye_points[3])
            
            if h < 0.001:
                return 0.0
            
            return (v1 + v2) / (2.0 * h)
        
        left_eye = get_eye_points(LEFT_EYE)
        right_eye = get_eye_points(RIGHT_EYE)
        
        left_ear = compute_ear(left_eye)
        right_ear = compute_ear(right_eye)
        
        raw_ear = (left_ear + right_ear) / 2.0
        
        # Foreshortening compensation
        yaw_rad = np.radians(abs(yaw_angle))
        foreshortening_factor = max(np.cos(yaw_rad), 0.1)
        compensated_ear = raw_ear / foreshortening_factor
        
        return {
            'raw_ear': raw_ear,
            'compensated_ear': compensated_ear,
            'left_ear': left_ear,
            'right_ear': right_ear,
            'foreshortening': foreshortening_factor
        }

    def enable_baseline_calibration(self, num_samples=10):
        """
        Enable baseline calibration mode.
        
        Call this at startup, then the system will automatically collect
        samples and establish a neutral baseline.
        
        Args:
            num_samples: Number of frames to average for baseline
        """
        self.baseline_enabled = True
        self.baseline_samples = []
        self.baseline_count = num_samples
        logger.info(f"Baseline calibration enabled ({num_samples} samples)")

    def reset_baseline(self):
        """Reset baseline calibration."""
        self.baseline_samples = []
        self.baseline_pitch = 0.0
        self.baseline_yaw = 0.0
        self.baseline_roll = 0.0
        logger.info("Baseline reset")

    def _add_baseline_sample(self, pitch, yaw, roll):
        """Internal: Add sample for baseline calibration."""
        if not self.baseline_enabled:
            return False
        
        if len(self.baseline_samples) < self.baseline_count:
            self.baseline_samples.append((pitch, yaw, roll))
            
            if len(self.baseline_samples) == self.baseline_count:
                # Calculate baseline
                samples = np.array(self.baseline_samples)
                self.baseline_pitch = np.mean(samples[:, 0])
                self.baseline_yaw = np.mean(samples[:, 1])
                self.baseline_roll = np.mean(samples[:, 2])
                
                logger.info(f"Baseline established: P={self.baseline_pitch:.1f}° "
                          f"Y={self.baseline_yaw:.1f}° R={self.baseline_roll:.1f}°")
                return True
        return False

    def calculate_pose(self, face_landmarks, img_w, img_h, use_improved_angles=True):
        """
        Calculate head pose angles.
        
        Args:
            face_landmarks: MediaPipe face mesh results
            img_w, img_h: Image dimensions
            use_improved_angles: If True, use better Euler extraction. 
                                If False, use original RQDecomp3x3 method.
        
        Returns:
            dict with pitch, yaw, roll, ear, and status info
        """
        try:
            # Initialize camera matrix (once)
            if self.camera_matrix is None:
                self.camera_matrix = self._build_camera_matrix(img_w, img_h)
            
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
                return self._return_previous_state()
            
            # Convert to rotation matrix
            rmat, _ = cv2.Rodrigues(self.rvec)
            
            # Extract Euler angles
            if use_improved_angles:
                # NEW: Better method
                pitch_raw, yaw_raw, roll_raw = self.rotation_matrix_to_euler_angles(rmat)
            else:
                # ORIGINAL: Your proven method
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                pitch_raw = angles[0]
                yaw_raw = angles[1]
                roll_raw = angles[2]
                
                # Normalize roll
                if roll_raw > 90:
                    roll_raw = roll_raw - 180
                elif roll_raw < -90:
                    roll_raw = roll_raw + 180
            
            # Apply baseline if calibrated
            if self.baseline_enabled and len(self.baseline_samples) == self.baseline_count:
                pitch_raw -= self.baseline_pitch
                yaw_raw -= self.baseline_yaw
                roll_raw -= self.baseline_roll
            
            # First frame: initialize
            if self.first_frame:
                self.prev_pitch = pitch_raw
                self.prev_yaw = yaw_raw
                self.prev_roll = roll_raw
                self.first_frame = False
                
                # Start baseline collection
                if self.baseline_enabled:
                    self._add_baseline_sample(pitch_raw, yaw_raw, roll_raw)
                
                # Calculate EAR
                ear_data = self.calculate_ear(face_landmarks, img_w, img_h, yaw_raw)
                self.prev_ear = ear_data['compensated_ear']
                
                return {
                    'pitch': pitch_raw,
                    'yaw': yaw_raw,
                    'roll': roll_raw,
                    'ear': ear_data['compensated_ear'],
                    'raw_ear': ear_data['raw_ear'],
                    'baseline_ready': len(self.baseline_samples) == self.baseline_count if self.baseline_enabled else True
                }
            
            # Continue baseline collection
            if self.baseline_enabled and len(self.baseline_samples) < self.baseline_count:
                self._add_baseline_sample(pitch_raw, yaw_raw, roll_raw)
            
            # Apply deadzone (your proven values)
            if abs(pitch_raw - self.prev_pitch) < self.DEADZONE_THRESH:
                pitch_raw = self.prev_pitch
            if abs(yaw_raw - self.prev_yaw) < self.DEADZONE_THRESH:
                yaw_raw = self.prev_yaw
            if abs(roll_raw - self.prev_roll) < self.DEADZONE_THRESH:
                roll_raw = self.prev_roll
            
            # Clamp to realistic ranges
            pitch_raw = np.clip(pitch_raw, -90, 90)
            yaw_raw = np.clip(yaw_raw, -90, 90)
            roll_raw = np.clip(roll_raw, -90, 90)
            
            # Apply exponential moving average (your proven values)
            smooth_pitch = (self.ALPHA_PITCH * pitch_raw) + ((1 - self.ALPHA_PITCH) * self.prev_pitch)
            smooth_yaw = (self.ALPHA_YAW * yaw_raw) + ((1 - self.ALPHA_YAW) * self.prev_yaw)
            smooth_roll = (self.ALPHA_ROLL * roll_raw) + ((1 - self.ALPHA_ROLL) * self.prev_roll)
            
            # Calculate EAR
            ear_data = self.calculate_ear(face_landmarks, img_w, img_h, smooth_yaw)
            smooth_ear = (self.ALPHA_EAR * ear_data['compensated_ear']) + \
                        ((1 - self.ALPHA_EAR) * self.prev_ear)
            
            # Update state
            self.prev_pitch = smooth_pitch
            self.prev_yaw = smooth_yaw
            self.prev_roll = smooth_roll
            self.prev_ear = smooth_ear
            
            return {
                'pitch': smooth_pitch,
                'yaw': smooth_yaw,
                'roll': smooth_roll,
                'ear': smooth_ear,
                'raw_ear': ear_data['raw_ear'],
                'left_ear': ear_data['left_ear'],
                'right_ear': ear_data['right_ear'],
                'foreshortening': ear_data['foreshortening'],
                'baseline_ready': len(self.baseline_samples) == self.baseline_count if self.baseline_enabled else True
            }
        
        except Exception as e:
            logger.error(f"Pose calculation error: {e}")
            return self._return_previous_state()

    def _return_previous_state(self):
        """Return previous state on error."""
        return {
            'pitch': self.prev_pitch,
            'yaw': self.prev_yaw,
            'roll': self.prev_roll,
            'ear': self.prev_ear,
            'raw_ear': self.prev_ear,
            'baseline_ready': len(self.baseline_samples) == self.baseline_count if self.baseline_enabled else True
        }

    def reset(self):
        """Reset estimator state."""
        self.rvec = None
        self.tvec = None
        self.prev_pitch = 0.0
        self.prev_yaw = 0.0
        self.prev_roll = 0.0
        self.prev_ear = 0.3
        self.first_frame = True
        self.reset_baseline()
        logger.info("Head pose estimator reset")


# ============================================================================
# USAGE EXAMPLE & COMPARISON TEST
# ============================================================================

if __name__ == "__main__":
    import mediapipe as mp
    
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("ENHANCED HEAD POSE ESTIMATOR - COMPARISON MODE")
    print("="*70)
    print("\nModes:")
    print("  Mode 1: Original method (approximated focal, RQDecomp3x3)")
    print("  Mode 2: Accurate camera + improved angles (RECOMMENDED)")
    print("\nControls:")
    print("  '1' - Switch to original method")
    print("  '2' - Switch to improved method")
    print("  'c' - Calibrate baseline")
    print("  'r' - Reset")
    print("  'q' - Quit")
    print("\n")
    
    # Start with improved method
    estimator = HeadPoseEstimator(
        camera_type="logitech_c920x",
        use_accurate_camera=True
    )
    estimator.enable_baseline_calibration(num_samples=10)
    
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    use_improved = True
    current_mode = "Improved (accurate camera + better angles)"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            
            # Calculate pose
            pose = estimator.calculate_pose(face, w, h, use_improved_angles=use_improved)
            
            # Display mode
            mode_color = (0, 255, 0) if use_improved else (255, 255, 0)
            cv2.putText(frame, f"Mode: {current_mode}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
            
            # Baseline status
            if not pose['baseline_ready']:
                cv2.putText(frame, "CALIBRATING BASELINE... Keep head still", 
                           (w//2 - 250, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 255), 2)
            
            # Display angles
            y_pos = 80
            cv2.putText(frame, f"Pitch: {pose['pitch']:7.1f} deg", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y_pos += 40
            cv2.putText(frame, f"Yaw:   {pose['yaw']:7.1f} deg", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y_pos += 40
            cv2.putText(frame, f"Roll:  {pose['roll']:7.1f} deg", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y_pos += 50
            cv2.putText(frame, f"EAR:   {pose['ear']:7.3f}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            
            # Blink detection
            if pose['ear'] < 0.2:
                cv2.putText(frame, "BLINK!", (w - 200, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            # Visual angle indicators
            # Pitch arrow (up/down)
            pitch_y = int(h//2 - pose['pitch'] * 2)
            cv2.arrowedLine(frame, (w-80, h//2), (w-80, pitch_y), (0, 255, 0), 3)
            cv2.putText(frame, "P", (w-90, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Yaw arrow (left/right)
            yaw_x = int(w//2 + pose['yaw'] * 2)
            cv2.arrowedLine(frame, (w//2, h-80), (yaw_x, h-80), (0, 255, 0), 3)
            cv2.putText(frame, "Y", (w//2-10, h-90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Roll indicator (circle)
            roll_rad = np.radians(pose['roll'])
            roll_x = int(w-100 + 30 * np.sin(roll_rad))
            roll_y = int(100 - 30 * np.cos(roll_rad))
            cv2.circle(frame, (w-100, 100), 35, (100, 100, 100), 2)
            cv2.line(frame, (w-100, 100), (roll_x, roll_y), (0, 255, 0), 3)
            cv2.putText(frame, "R", (w-110, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        else:
            cv2.putText(frame, "NO FACE DETECTED", (w//2 - 150, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Instructions
        cv2.putText(frame, "1:Original | 2:Improved | c:Calibrate | r:Reset | q:Quit", 
                   (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("Enhanced Head Pose", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            # Switch to original method
            use_improved = False
            current_mode = "Original (approximated focal, RQDecomp3x3)"
            estimator.use_accurate_camera = False
            estimator.camera_matrix = None  # Force rebuild
            estimator.reset()
            print("\n→ Switched to ORIGINAL method")
        elif key == ord('2'):
            # Switch to improved method
            use_improved = True
            current_mode = "Improved (accurate camera + better angles)"
            estimator.use_accurate_camera = True
            estimator.camera_matrix = None  # Force rebuild
            estimator.reset()
            print("\n→ Switched to IMPROVED method")
        elif key == ord('c'):
            print("\n→ Recalibrating baseline... Keep head still!")
            estimator.reset_baseline()
            estimator.enable_baseline_calibration(num_samples=10)
        elif key == ord('r'):
            print("\n→ Full reset")
            estimator.reset()
    
    cap.release()
    cv2.destroyAllWindows()
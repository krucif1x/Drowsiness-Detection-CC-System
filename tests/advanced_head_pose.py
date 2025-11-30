"""
LOGITECH C920X HEAD POSE ESTIMATOR - FULLY EXPLAINED
Corrected sensor size: 4.8mm × 3.6mm

This code calculates where your head is pointing (pitch/yaw/roll angles)
and measures eye openness (EAR) from webcam video.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


# Which MediaPipe landmarks to use for head pose
LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]
# 1: Nose tip, 152: Chin, 33: Left eye outer corner, 263: Right eye outer corner
# 61: Left mouth corner, 291: Right mouth corner

# Eye landmarks for measuring eye openness
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


class HeadPoseEstimator:
    """
    Calculates head orientation (pitch/yaw/roll) and eye aspect ratio.
    
    LOGITECH C920X SPECS (CORRECTED):
    - Focal Length: 3.67mm
    - Sensor Size: 4.8mm (width) × 3.6mm (height)
    - Field of View: 78° diagonal
    """
    
    def __init__(self):
        """Set up the estimator with C920X camera specifications."""
        
        # === LOGITECH C920X HARDWARE SPECS ===
        self.focal_length_mm = 3.67    # Physical lens focal length
        self.sensor_width_mm = 4.8     # Physical sensor width (CORRECTED)
        self.sensor_height_mm = 3.6    # Physical sensor height (CORRECTED)
        self.fov_diagonal_deg = 78.0   # Total field of view
        
        # === WHICH LANDMARKS TO USE ===
        self.landmark_indices = LANDMARK_INDICES
        
        # === CAMERA CALIBRATION ===
        self.camera_matrix = None       # Will be calculated later
        self.dist_coeffs = np.zeros((4, 1))  # Assume no lens distortion
        
        # === PnP SOLVER STATE ===
        self.rvec = None  # Rotation vector (will store previous frame's result)
        self.tvec = None  # Translation vector
        
        # === SMOOTHING STATE (to reduce jitter) ===
        self.prev_pitch = 0.0
        self.prev_yaw = 0.0
        self.prev_roll = 0.0
        self.prev_ear = 0.3
        self.first_frame = True
        
        # === SMOOTHING PARAMETERS ===
        self.DEADZONE_THRESH = 1.0   # Ignore changes smaller than this (increased)
        self.ALPHA_PITCH = 0.4       # How fast pitch adapts (0=slow, 1=instant)
        self.ALPHA_YAW = 0.4         # Reduced for smoother tracking
        self.ALPHA_ROLL = 0.4
        self.ALPHA_EAR = 0.4
        
        logger.info("HeadPoseEstimator initialized for Logitech C920X")
        logger.info(f"  Sensor: {self.sensor_width_mm}mm × {self.sensor_height_mm}mm")

    def calculate_focal_length_pixels(self, img_w, img_h):
        """
        Convert focal length from millimeters to pixels.
        
        WHY: The camera matrix needs focal length in PIXELS, but the
        camera spec gives it in MILLIMETERS. We need to convert.
        
        FORMULA:
        focal_pixels = (focal_mm / sensor_mm) × image_pixels
        
        EXAMPLE:
        focal_x = (3.67mm / 4.8mm) × 1920px = 1465 pixels
        
        EXPLANATION:
        - If focal length = sensor size, then 1mm on sensor = 1 image width
        - Our focal is 3.67mm, sensor is 4.8mm, so ratio is 3.67/4.8 = 0.764
        - This ratio × image width gives focal length in pixels
        
        Args:
            img_w: Image width in pixels (e.g., 1920)
            img_h: Image height in pixels (e.g., 1080)
        
        Returns:
            (focal_x, focal_y) in pixels
        """
        focal_x = (self.focal_length_mm / self.sensor_width_mm) * img_w
        focal_y = (self.focal_length_mm / self.sensor_height_mm) * img_h
        
        logger.info(f"Focal length calculated: fx={focal_x:.1f}px, fy={focal_y:.1f}px")
        logger.info(f"  (from {img_w}×{img_h} image)")
        
        return focal_x, focal_y

    def rotation_matrix_to_euler_angles(self, R):
        """
        Extract pitch, yaw, roll angles from a 3×3 rotation matrix.
        
        FIXED: Uses YXZ Euler convention which matches MediaPipe's coordinate system.
        
        MediaPipe coordinate system:
        - X-axis: right (positive = right side of face)
        - Y-axis: down (positive = bottom of face)  
        - Z-axis: forward (positive = away from face, into camera)
        
        Angles:
        - Pitch: rotation around X-axis (nodding up/down)
        - Yaw: rotation around Y-axis (shaking head left/right)
        - Roll: rotation around Z-axis (tilting head side to side)
        
        Args:
            R: 3×3 rotation matrix from cv2.Rodrigues()
        
        Returns:
            (pitch, yaw, roll) in degrees
        """
        # YXZ Euler angle extraction (order: Yaw -> Pitch -> Roll)
        # This matches how head rotations naturally compose
        
        # Check for gimbal lock
        sin_pitch = -R[2, 1]
        
        # Clamp to valid range for arcsin
        sin_pitch = np.clip(sin_pitch, -1.0, 1.0)
        
        # Check if we're at gimbal lock (pitch near ±90°)
        if abs(sin_pitch) >= 0.99999:
            # Gimbal lock case
            pitch = np.arcsin(sin_pitch)
            yaw = np.arctan2(-R[0, 2], R[0, 0])
            roll = 0.0  # Set roll to zero by convention
        else:
            # Normal case
            pitch = np.arcsin(sin_pitch)
            yaw = np.arctan2(R[2, 0], R[2, 2])
            roll = np.arctan2(R[0, 1], R[1, 1])
        
        # Convert to degrees
        pitch_deg = np.degrees(pitch)
        yaw_deg = np.degrees(yaw)
        roll_deg = np.degrees(roll)
        
        return pitch_deg, yaw_deg, roll_deg

    def build_model_points_from_mediapipe(self, face_landmarks, scale=450):
        """
        Create a 3D face model using MediaPipe's 3D landmark positions.
        
        IMPROVED: Better scaling and coordinate alignment for accurate pose.
        
        MediaPipe coordinate system:
        - x: 0 (left) to 1 (right), 0.5 = center
        - y: 0 (top) to 1 (bottom), 0.5 = center
        - z: depth, negative = closer to camera
        
        Args:
            face_landmarks: MediaPipe face mesh results
            scale: Size of face model in mm (450 works well for most adults)
        
        Returns:
            Array of 3D points [[x1,y1,z1], [x2,y2,z2], ...]
        """
        model_points = []
        
        for i in self.landmark_indices:
            lm = face_landmarks.landmark[i]
            
            # Convert normalized coords to millimeters
            # Keep MediaPipe's coordinate system (don't flip z)
            x = (lm.x - 0.5) * scale
            y = (lm.y - 0.5) * scale
            z = lm.z * scale  # Keep original sign
            
            model_points.append([x, y, z])
        
        return np.array(model_points, dtype=np.float64)

    def calculate_ear(self, face_landmarks, img_w, img_h, yaw_angle):
        """
        Calculate Eye Aspect Ratio (EAR) - a measure of eye openness.
        
        WHY: EAR tells us if eyes are open or closed (for blink detection).
        
        WHAT IS EAR:
        EAR = (vertical_distance) / (horizontal_distance)
        - Open eye: EAR ≈ 0.3
        - Closed eye: EAR ≈ 0.0-0.1
        
        THE PROBLEM:
        When you turn your head sideways, eyes APPEAR narrower due to
        perspective (called "foreshortening"). This makes EAR drop even
        though eyes are still open.
        
        THE SOLUTION:
        Compensate using cos(yaw_angle). When head turns 60°, eyes appear
        50% narrower (cos(60°) = 0.5), so we divide EAR by 0.5 to correct.
        
        Eye landmark layout:
              p1
              |
        p0 ●――●――● p3
              |
              p2
        
        Args:
            face_landmarks: MediaPipe results
            img_w, img_h: Image dimensions
            yaw_angle: Current head yaw in degrees
        
        Returns:
            Dictionary with raw and compensated EAR values
        """
        
        def get_eye_points(eye_indices):
            """Get pixel coordinates for eye landmarks."""
            return np.array([
                [face_landmarks.landmark[i].x * img_w,
                 face_landmarks.landmark[i].y * img_h]
                for i in eye_indices
            ])
        
        def compute_ear(eye_points):
            """Calculate EAR from 6 eye landmarks."""
            # Two vertical distances (for robustness)
            v1 = np.linalg.norm(eye_points[1] - eye_points[5])
            v2 = np.linalg.norm(eye_points[2] - eye_points[4])
            
            # One horizontal distance
            h = np.linalg.norm(eye_points[0] - eye_points[3])
            
            if h < 0.001:  # Prevent division by zero
                return 0.0
            
            # EAR = average_vertical / horizontal
            return (v1 + v2) / (2.0 * h)
        
        # Get landmarks for both eyes
        left_eye_points = get_eye_points(LEFT_EYE)
        right_eye_points = get_eye_points(RIGHT_EYE)
        
        # Calculate EAR for each eye
        left_ear = compute_ear(left_eye_points)
        right_ear = compute_ear(right_eye_points)
        
        # Average both eyes
        raw_ear = (left_ear + right_ear) / 2.0
        
        # === FORESHORTENING COMPENSATION ===
        # Convert yaw angle to radians
        yaw_rad = np.radians(abs(yaw_angle))
        
        # Calculate how much eyes are compressed by perspective
        # cos(0°) = 1.0 (head-on, no compression)
        # cos(60°) = 0.5 (turned 60°, 50% narrower)
        # cos(90°) = 0.0 (profile view, edge-on)
        foreshortening_factor = np.cos(yaw_rad)
        
        # Don't divide by near-zero (would blow up the result)
        foreshortening_factor = max(foreshortening_factor, 0.1)
        
        # Correct the EAR: divide by compression factor
        # If eyes appear 50% narrower, multiply EAR by 2 to compensate
        compensated_ear = raw_ear / foreshortening_factor
        
        return {
            'raw_ear': raw_ear,                     # What we actually see
            'compensated_ear': compensated_ear,      # Corrected for angle
            'left_ear': left_ear,
            'right_ear': right_ear,
            'foreshortening_factor': foreshortening_factor
        }

    def calculate_pose(self, face_landmarks, img_w, img_h):
        """
        MAIN FUNCTION: Calculate head pose (pitch/yaw/roll) and EAR.
        
        === THE COMPLETE PIPELINE ===
        
        STEP 1: Build camera matrix (convert mm to pixels)
        STEP 2: Get 2D points (where landmarks appear on screen)
        STEP 3: Get 3D points (where landmarks are on a real face)
        STEP 4: Solve PnP (find rotation & translation)
        STEP 5: Convert rotation to angles (pitch/yaw/roll)
        STEP 6: Apply smoothing (reduce jitter)
        STEP 7: Calculate EAR with compensation
        STEP 8: Return results
        
        Args:
            face_landmarks: MediaPipe face mesh results
            img_w: Image width in pixels
            img_h: Image height in pixels
        
        Returns:
            Dictionary with pitch, yaw, roll, ear, and debug info
        """
        
        try:
            # ================================================================
            # STEP 1: BUILD CAMERA MATRIX (only once per video)
            # ================================================================
            if self.camera_matrix is None:
                # Convert focal length from mm to pixels
                focal_x, focal_y = self.calculate_focal_length_pixels(img_w, img_h)
                
                # Principal point (optical center) is at image center
                center_x = img_w / 2.0
                center_y = img_h / 2.0
                
                # Camera intrinsic matrix
                # [fx  0  cx]   fx, fy = focal length in pixels
                # [0  fy  cy]   cx, cy = image center
                # [0   0   1]
                self.camera_matrix = np.array([
                    [focal_x,      0, center_x],
                    [     0, focal_y, center_y],
                    [     0,      0,        1]
                ], dtype=np.float64)
                
                logger.info(f"Camera matrix created for {img_w}×{img_h}")
            
            # ================================================================
            # STEP 2: GET 2D IMAGE POINTS (where landmarks appear on screen)
            # ================================================================
            # Extract pixel coordinates from MediaPipe
            image_points = np.array([
                (face_landmarks.landmark[i].x * img_w,   # Convert 0-1 to pixels
                 face_landmarks.landmark[i].y * img_h)
                for i in self.landmark_indices
            ], dtype=np.float64)
            
            # Now we have: [[x1, y1], [x2, y2], ...] in pixels
            
            # ================================================================
            # STEP 3: GET 3D MODEL POINTS (real face geometry)
            # ================================================================
            # Build 3D face model from MediaPipe's 3D estimates
            model_points = self.build_model_points_from_mediapipe(face_landmarks)
            
            # Now we have: [[x1, y1, z1], [x2, y2, z2], ...] in millimeters
            
            # ================================================================
            # STEP 4: SOLVE PnP (find camera pose)
            # ================================================================
            # PnP = "Perspective-n-Point" problem
            # Question: "Given these 2D-3D point pairs, where is the camera?"
            # Answer: Rotation (rvec) and Translation (tvec)
            
            if self.rvec is None:
                # First frame: solve from scratch
                success, self.rvec, self.tvec = cv2.solvePnP(
                    model_points,          # 3D points in model space
                    image_points,          # 2D points in image space
                    self.camera_matrix,    # Camera properties
                    self.dist_coeffs,      # Lens distortion (zeros)
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
            else:
                # Subsequent frames: use previous result as starting guess
                # (this makes it faster and more stable)
                success, self.rvec, self.tvec = cv2.solvePnP(
                    model_points,
                    image_points,
                    self.camera_matrix,
                    self.dist_coeffs,
                    rvec=self.rvec,           # Previous rotation
                    tvec=self.tvec,           # Previous translation
                    useExtrinsicGuess=True,   # Start from previous
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
            
            if not success:
                # PnP failed - return previous values
                return {
                    'pitch': self.prev_pitch,
                    'yaw': self.prev_yaw,
                    'roll': self.prev_roll,
                    'ear': self.prev_ear,
                    'raw_ear': self.prev_ear
                }
            
            # Now we have:
            # self.rvec = rotation vector [rx, ry, rz]
            # self.tvec = translation vector [tx, ty, tz]
            
            # ================================================================
            # STEP 5: CONVERT ROTATION TO EULER ANGLES
            # ================================================================
            
            # 5a: Convert rotation vector to rotation matrix
            # Rotation vector → 3×3 matrix (Rodrigues' formula)
            rmat, _ = cv2.Rodrigues(self.rvec)
            
            # 5b: Extract pitch, yaw, roll from rotation matrix
            pitch_raw, yaw_raw, roll_raw = self.rotation_matrix_to_euler_angles(rmat)
            
            # Apply calibration offsets
            pitch_raw += self.pitch_offset
            yaw_raw += self.yaw_offset
            roll_raw += self.roll_offset
            
            # ================================================================
            # STEP 6: APPLY SMOOTHING
            # ================================================================
            
            # 6a: Initialize on first frame
            if self.first_frame:
                self.prev_pitch = pitch_raw
                self.prev_yaw = yaw_raw
                self.prev_roll = roll_raw
                self.first_frame = False
            
            # 6b: Deadzone - ignore tiny changes (noise)
            if abs(pitch_raw - self.prev_pitch) < self.DEADZONE_THRESH:
                pitch_raw = self.prev_pitch
            if abs(yaw_raw - self.prev_yaw) < self.DEADZONE_THRESH:
                yaw_raw = self.prev_yaw
            if abs(roll_raw - self.prev_roll) < self.DEADZONE_THRESH:
                roll_raw = self.prev_roll
            
            # 6c: Clamp to realistic ranges
            pitch_raw = np.clip(pitch_raw, -90, 90)
            yaw_raw = np.clip(yaw_raw, -90, 90)
            roll_raw = np.clip(roll_raw, -90, 90)
            
            # 6d: Exponential moving average (blend with previous frame)
            # new_value = α×current + (1-α)×previous
            # α near 0 = very smooth (slow to change)
            # α near 1 = very responsive (fast to change)
            smooth_pitch = self.ALPHA_PITCH * pitch_raw + \
                          (1 - self.ALPHA_PITCH) * self.prev_pitch
            smooth_yaw = self.ALPHA_YAW * yaw_raw + \
                        (1 - self.ALPHA_YAW) * self.prev_yaw
            smooth_roll = self.ALPHA_ROLL * roll_raw + \
                         (1 - self.ALPHA_ROLL) * self.prev_roll
            
            # ================================================================
            # STEP 7: CALCULATE EAR WITH FORESHORTENING COMPENSATION
            # ================================================================
            ear_data = self.calculate_ear(face_landmarks, img_w, img_h, smooth_yaw)
            
            # Smooth EAR too
            smooth_ear = self.ALPHA_EAR * ear_data['compensated_ear'] + \
                        (1 - self.ALPHA_EAR) * self.prev_ear
            
            # ================================================================
            # STEP 8: UPDATE STATE FOR NEXT FRAME
            # ================================================================
            self.prev_pitch = smooth_pitch
            self.prev_yaw = smooth_yaw
            self.prev_roll = smooth_roll
            self.prev_ear = smooth_ear
            
            # ================================================================
            # STEP 9: RETURN RESULTS
            # ================================================================
            return {
                'pitch': smooth_pitch,              # Up/down angle
                'yaw': smooth_yaw,                  # Left/right angle
                'roll': smooth_roll,                # Tilt angle
                'ear': smooth_ear,                  # Compensated (use this!)
                'raw_ear': ear_data['raw_ear'],     # Uncompensated
                'left_ear': ear_data['left_ear'],
                'right_ear': ear_data['right_ear'],
                'foreshortening': ear_data['foreshortening_factor']
            }
        
        except Exception as e:
            logger.error(f"Error in pose calculation: {e}")
            # Return previous values on error
            return {
                'pitch': self.prev_pitch,
                'yaw': self.prev_yaw,
                'roll': self.prev_roll,
                'ear': self.prev_ear,
                'raw_ear': self.prev_ear
            }

    def reset(self):
        """Reset all state (call when face is lost)."""
        self.rvec = None
        self.tvec = None
        self.prev_pitch = 0.0
        self.prev_yaw = 0.0
        self.prev_roll = 0.0
        self.prev_ear = 0.3
        self.first_frame = True
        logger.info("Head pose estimator reset")
    
    def calibrate_neutral(self, face_landmarks, img_w, img_h, num_samples=30):
        """
        Calibrate neutral position by averaging angles over multiple frames.
        
        HOW TO USE:
        1. Look straight at camera in neutral position
        2. Call this method
        3. It will sample 30 frames and calculate offsets
        
        Args:
            face_landmarks: MediaPipe face mesh results
            img_w, img_h: Image dimensions
            num_samples: Number of frames to average
        
        Returns:
            Success status and calculated offsets
        """
        pitch_samples = []
        yaw_samples = []
        roll_samples = []
        
        # Temporarily disable offsets for calibration
        old_pitch_offset = self.pitch_offset
        old_yaw_offset = self.yaw_offset
        old_roll_offset = self.roll_offset
        
        self.pitch_offset = 0.0
        self.yaw_offset = 0.0
        self.roll_offset = 0.0
        
        # Collect samples
        for _ in range(num_samples):
            pose = self.calculate_pose(face_landmarks, img_w, img_h)
            pitch_samples.append(pose['pitch'])
            yaw_samples.append(pose['yaw'])
            roll_samples.append(pose['roll'])
        
        # Calculate average offsets
        avg_pitch = np.mean(pitch_samples)
        avg_yaw = np.mean(yaw_samples)
        avg_roll = np.mean(roll_samples)
        
        # Set offsets to bring neutral to zero
        self.pitch_offset = -avg_pitch
        self.yaw_offset = -avg_yaw
        self.roll_offset = -avg_roll
        
        logger.info(f"Calibration complete:")
        logger.info(f"  Pitch offset: {self.pitch_offset:.1f}°")
        logger.info(f"  Yaw offset: {self.yaw_offset:.1f}°")
        logger.info(f"  Roll offset: {self.roll_offset:.1f}°")
        
        return True, (self.pitch_offset, self.yaw_offset, self.roll_offset)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    import mediapipe as mp
    
    print("\n" + "="*70)
    print("LOGITECH C920X HEAD POSE TRACKER - DEBUG MODE")
    print("="*70)
    print("\nCamera Specs:")
    print("  Focal Length: 3.67mm")
    print("  Sensor Size: 4.8mm × 3.6mm")
    print("\nControls:")
    print("  'c' - Calibrate neutral position (look straight at camera)")
    print("  'r' - Reset calibration")
    print("  'd' - Toggle debug info")
    print("  'q' - Quit")
    print("\n")
    
    # Initialize head pose estimator
    estimator = HeadPoseEstimator()
    
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    show_debug = True
    calibrating = False
    calibration_frames = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            
            # Calibration mode
            if calibrating:
                calibration_frames += 1
                cv2.putText(frame, f"CALIBRATING... {calibration_frames}/30", 
                           (w//2 - 150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.0, (0, 255, 255), 2)
                
                if calibration_frames >= 30:
                    estimator.calibrate_neutral(face, w, h)
                    calibrating = False
                    calibration_frames = 0
            
            # Calculate head pose and EAR
            pose = estimator.calculate_pose(face, w, h)
            
            # Display main results
            y_pos = 30
            cv2.putText(frame, f"Pitch: {pose['pitch']:7.1f} deg", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_pos += 35
            cv2.putText(frame, f"Yaw:   {pose['yaw']:7.1f} deg", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_pos += 35
            cv2.putText(frame, f"Roll:  {pose['roll']:7.1f} deg", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_pos += 45
            cv2.putText(frame, f"EAR:   {pose['ear']:7.3f}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            # Debug info
            if show_debug:
                y_pos += 50
                cv2.putText(frame, "DEBUG INFO:", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                y_pos += 25
                cv2.putText(frame, f"Raw EAR: {pose['raw_ear']:.3f}", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_pos += 20
                cv2.putText(frame, f"Foreshortening: {pose['foreshortening']:.3f}", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_pos += 20
                cv2.putText(frame, f"Offsets: P:{estimator.pitch_offset:.1f} Y:{estimator.yaw_offset:.1f} R:{estimator.roll_offset:.1f}", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Blink detection
            if pose['ear'] < 0.2:
                cv2.putText(frame, "BLINK!", (w - 200, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            # Angle indicators
            # Draw pitch arrow (up/down)
            pitch_y = int(h//2 - pose['pitch'] * 3)
            cv2.arrowedLine(frame, (w-50, h//2), (w-50, pitch_y), (0, 255, 0), 3)
            
            # Draw yaw arrow (left/right)
            yaw_x = int(w//2 + pose['yaw'] * 3)
            cv2.arrowedLine(frame, (w//2, h-50), (yaw_x, h-50), (0, 255, 0), 3)
            
            # Draw roll indicator (circle rotation)
            roll_rad = np.radians(pose['roll'])
            roll_x = int(w-100 + 30 * np.sin(roll_rad))
            roll_y = int(100 - 30 * np.cos(roll_rad))
            cv2.circle(frame, (w-100, 100), 35, (100, 100, 100), 2)
            cv2.line(frame, (w-100, 100), (roll_x, roll_y), (0, 255, 0), 3)
        
        else:
            cv2.putText(frame, "NO FACE DETECTED", 
                       (w//2 - 150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.0, (0, 0, 255), 2)
        
        # Instructions
        cv2.putText(frame, "c:Calibrate | r:Reset | d:Debug | q:Quit", 
                   (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow('C920X Head Pose Tracker', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            print("\nStarting calibration... Look straight at camera!")
            calibrating = True
            calibration_frames = 0
        elif key == ord('r'):
            print("\nResetting calibration...")
            estimator.pitch_offset = 0.0
            estimator.yaw_offset = 0.0
            estimator.roll_offset = 0.0
            estimator.reset()
        elif key == ord('d'):
            show_debug = not show_debug
    
    cap.release()
    cv2.destroyAllWindows()
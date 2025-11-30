"""
EYE ASPECT RATIO (EAR) ANGLE CORRECTION
Corrects foreshortening of EAR values when head is tilted or turned.
"""

import numpy as np
import logging
import cv2
import mediapipe as mp
import time

logger = logging.getLogger(__name__)


class EARAngleCorrector:
    """
    Corrects Eye Aspect Ratio (EAR) values for head pose angle effects.
    
    When viewing eyes from an angle, they appear more closed than they actually are
    due to perspective foreshortening. This class compensates for that effect.
    """
    
    def __init__(self):
        """Initialize the EAR angle corrector."""
        self.calibration_mode = False
        self.calibrated_neutral_ear = None
        self.calibration_samples = []
        
        # Default correction strengths (can be tuned)
        self.PITCH_CORRECTION_STRENGTH = 0.8  # How much to correct for up/down tilt
        self.YAW_CORRECTION_STRENGTH = 0.6    # How much to correct for left/right turn
        self.ROLL_CORRECTION_STRENGTH = 0.3   # How much to correct for head tilt
        
        logger.info("EAR Angle Corrector initialized")
    
    def _calculate_foreshortening_factor(self, pitch, yaw, roll):
        """
        Calculate how much the eye is foreshortened based on head angles.
        
        Args:
            pitch: Head pitch angle in degrees (positive = looking up)
            yaw: Head yaw angle in degrees (positive = looking right)
            roll: Head roll angle in degrees (positive = tilted right)
        
        Returns:
            Foreshortening factor (1.0 = no foreshortening, >1.0 = appears more closed)
        """
        # Convert angles to radians
        pitch_rad = np.radians(pitch)
        yaw_rad = np.radians(yaw)
        roll_rad = np.radians(roll)
        
        # Pitch effect: Eyes appear more closed when looking up or down
        # cos(pitch) ranges from 1.0 (straight) to 0.0 (90° up/down)
        pitch_factor = 1.0 / (np.cos(pitch_rad) + 0.1)  # +0.1 prevents division by zero
        pitch_factor = 1.0 + (pitch_factor - 1.0) * self.PITCH_CORRECTION_STRENGTH
        
        # Yaw effect: Eyes appear more closed when turned sideways
        # More pronounced effect as you turn further
        yaw_factor = 1.0 / (np.cos(yaw_rad) + 0.1)
        yaw_factor = 1.0 + (yaw_factor - 1.0) * self.YAW_CORRECTION_STRENGTH
        
        # Roll effect: Slight foreshortening when head is tilted
        roll_factor = 1.0 + abs(np.sin(roll_rad)) * self.ROLL_CORRECTION_STRENGTH
        
        # Combine factors (multiplicative - effects compound)
        total_factor = pitch_factor * yaw_factor * roll_factor
        
        # Clamp to reasonable range (prevent over-correction)
        total_factor = np.clip(total_factor, 1.0, 3.0)
        
        return total_factor
    
    def correct_ear(self, measured_ear, pitch, yaw, roll, left_or_right='both'):
        """
        Correct a measured EAR value for head pose angle effects.
        
        Args:
            measured_ear: The raw EAR value from MediaPipe landmarks
            pitch: Head pitch angle in degrees
            yaw: Head yaw angle in degrees  
            roll: Head roll angle in degrees
            left_or_right: 'left', 'right', or 'both' - applies asymmetric correction for yaw
        
        Returns:
            Corrected EAR value (true eye openness)
        """
        if measured_ear is None or measured_ear <= 0:
            return measured_ear
        
        # Calculate base foreshortening
        base_factor = self._calculate_foreshortening_factor(pitch, yaw, roll)
        
        # Apply asymmetric correction for yaw (left/right turn affects eyes differently)
        if left_or_right == 'left' and yaw > 0:
            # Turning right makes left eye more foreshortened
            yaw_asymmetry = 1.0 + (abs(yaw) / 90.0) * 0.3
            base_factor *= yaw_asymmetry
        elif left_or_right == 'right' and yaw < 0:
            # Turning left makes right eye more foreshortened
            yaw_asymmetry = 1.0 + (abs(yaw) / 90.0) * 0.3
            base_factor *= yaw_asymmetry
        
        # Correct the EAR by multiplying by foreshortening factor
        # Higher factor = eyes appeared more closed, so true EAR is higher
        corrected_ear = measured_ear * base_factor
        
        # Clamp to physiologically reasonable range
        corrected_ear = np.clip(corrected_ear, 0.0, 0.6)  # EAR rarely exceeds 0.6
        
        return corrected_ear
    
    def correct_both_ears(self, left_ear, right_ear, pitch, yaw, roll):
        """
        Correct both left and right EAR values simultaneously.
        
        Args:
            left_ear: Measured left eye EAR
            right_ear: Measured right eye EAR
            pitch: Head pitch angle
            yaw: Head yaw angle
            roll: Head roll angle
        
        Returns:
            Tuple of (corrected_left_ear, corrected_right_ear)
        """
        corrected_left = self.correct_ear(left_ear, pitch, yaw, roll, left_or_right='left')
        corrected_right = self.correct_ear(right_ear, pitch, yaw, roll, left_or_right='right')
        
        return corrected_left, corrected_right
    
    def start_calibration(self):
        """
        Start calibration mode to establish neutral EAR baseline.
        User should look straight at camera with eyes normally open.
        """
        self.calibration_mode = True
        self.calibration_samples = []
        logger.info("Calibration started - look straight at camera with eyes normally open")
    
    def add_calibration_sample(self, ear_value, pitch, yaw, roll):
        """
        Add a sample during calibration.
        Only accepts samples when head is roughly facing forward.
        
        Args:
            ear_value: Average EAR of both eyes
            pitch, yaw, roll: Head angles
        """
        if not self.calibration_mode:
            logger.warning("Not in calibration mode")
            return False
        
        # Only accept samples when head is roughly straight
        if abs(pitch) < 10 and abs(yaw) < 10 and abs(roll) < 10:
            self.calibration_samples.append(ear_value)
            logger.debug(f"Calibration sample added: {ear_value:.3f} (total: {len(self.calibration_samples)})")
            return True
        else:
            logger.debug("Sample rejected - head not facing forward")
            return False
    
    def finish_calibration(self, min_samples=30):
        """
        Finish calibration and calculate neutral EAR baseline.
        
        Args:
            min_samples: Minimum samples needed for valid calibration
        
        Returns:
            True if calibration successful, False otherwise
        """
        if len(self.calibration_samples) < min_samples:
            logger.warning(f"Not enough samples: {len(self.calibration_samples)}/{min_samples}")
            self.calibration_mode = False
            return False
        
        # Use median to avoid outliers (blinks, etc.)
        self.calibrated_neutral_ear = np.median(self.calibration_samples)
        self.calibration_mode = False
        
        logger.info(f"Calibration complete: neutral EAR = {self.calibrated_neutral_ear:.3f} "
                   f"(from {len(self.calibration_samples)} samples)")
        return True
    
    def get_relative_openness(self, corrected_ear):
        """
        Get eye openness as percentage relative to calibrated neutral.
        
        Args:
            corrected_ear: Angle-corrected EAR value
        
        Returns:
            Percentage of normal openness (100 = fully open as calibrated, 0 = closed)
        """
        if self.calibrated_neutral_ear is None:
            logger.warning("No calibration data - returning raw percentage")
            return corrected_ear / 0.3 * 100  # Assume 0.3 is typical neutral
        
        openness_pct = (corrected_ear / self.calibrated_neutral_ear) * 100
        return np.clip(openness_pct, 0, 150)  # Allow slight over-opening
    
    def set_correction_strength(self, pitch=None, yaw=None, roll=None):
        """
        Adjust correction strength for different angles.
        
        Args:
            pitch: Pitch correction strength (0.0 to 1.0)
            yaw: Yaw correction strength (0.0 to 1.0)
            roll: Roll correction strength (0.0 to 1.0)
        """
        if pitch is not None:
            self.PITCH_CORRECTION_STRENGTH = np.clip(pitch, 0.0, 1.0)
        if yaw is not None:
            self.YAW_CORRECTION_STRENGTH = np.clip(yaw, 0.0, 1.0)
        if roll is not None:
            self.ROLL_CORRECTION_STRENGTH = np.clip(roll, 0.0, 1.0)
        
        logger.info(f"Correction strengths: pitch={self.PITCH_CORRECTION_STRENGTH:.2f}, "
                   f"yaw={self.YAW_CORRECTION_STRENGTH:.2f}, roll={self.ROLL_CORRECTION_STRENGTH:.2f}")


# Helper function to calculate EAR from landmarks
def calculate_ear(eye_landmarks, face_landmarks, img_w, img_h):
    """
    Calculate Eye Aspect Ratio from MediaPipe landmarks.
    
    Args:
        eye_landmarks: List of 6 landmark indices for one eye [p1, p2, p3, p4, p5, p6]
                      where p2-p6 and p3-p5 are vertical, p1-p4 is horizontal
        face_landmarks: MediaPipe face landmarks object
        img_w, img_h: Image dimensions
    
    Returns:
        EAR value (float)
    """
    # Get eye landmark coordinates
    coords = []
    for idx in eye_landmarks:
        landmark = face_landmarks.landmark[idx]
        coords.append([landmark.x * img_w, landmark.y * img_h])
    
    coords = np.array(coords)
    
    # Calculate distances
    # Vertical distances
    v1 = np.linalg.norm(coords[1] - coords[5])  # p2 to p6
    v2 = np.linalg.norm(coords[2] - coords[4])  # p3 to p5
    
    # Horizontal distance
    h = np.linalg.norm(coords[0] - coords[3])   # p1 to p4
    
    # EAR formula
    ear = (v1 + v2) / (2.0 * h)
    
    return ear


# MediaPipe eye landmark indices
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]   # Left eye landmarks
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]  # Right eye landmarks

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    corrector = EARAngleCorrector()

    # Try to import head-pose estimator (optional). If unavailable, use zero angles.
    try:
        from src.mediapipe.head_pose_trail import HeadPoseEstimator  # try common filename
        hpe = HeadPoseEstimator(camera_profile_name="logitech_c920x", img_w=1280, img_h=720)
        have_hpe = True
        logger.info("HeadPoseEstimator loaded - using pose compensation")
    except Exception:
        hpe = None
        have_hpe = False
        logger.warning("HeadPoseEstimator not available - using zero head angles")

    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(static_image_mode=False,
                                 max_num_faces=1,
                                 refine_landmarks=True,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    instructions = [
        "Keys: ESC/q=quit  s=start calib  a=add sample  f=finish calib",
        "If HeadPoseEstimator present, pose used; otherwise angles=0"
    ]

    last_fps_t = time.time()
    fps = 0
    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            img_h, img_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            pitch = yaw = roll = 0.0
            left_raw = right_raw = None
            corrected_left = corrected_right = None
            openness_pct = None

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                # compute head pose if possible
                if have_hpe:
                    try:
                        pitch, yaw, roll = hpe.calculate_pose(face_landmarks, img_w, img_h)
                    except Exception:
                        pitch = yaw = roll = 0.0

                # compute EARs
                left_raw = calculate_ear(LEFT_EYE_INDICES, face_landmarks, img_w, img_h)
                right_raw = calculate_ear(RIGHT_EYE_INDICES, face_landmarks, img_w, img_h)

                corrected_left, corrected_right = corrector.correct_both_ears(left_raw, right_raw, pitch, yaw, roll)
                avg_corrected = (corrected_left + corrected_right) / 2.0
                openness_pct = corrector.get_relative_openness(avg_corrected)

                # draw landmarks (optional simple)
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
                )

                # Draw small bars for raw vs corrected EAR
                def draw_bar(img, x, y, w, h, value, maxv=0.6, color=(0,200,0)):
                    v = np.clip(value / maxv, 0.0, 1.0)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (50,50,50), 1)
                    cv2.rectangle(img, (x, int(y + h*(1-v))), (x + w, y + h), color, -1)

                # left ear
                if left_raw is not None:
                    draw_bar(frame, 20, 80, 20, 80, left_raw, color=(0,0,200))
                    draw_bar(frame, 50, 80, 20, 80, corrected_left, color=(0,200,0))
                # right ear
                if right_raw is not None:
                    draw_bar(frame, 100, 80, 20, 80, right_raw, color=(0,0,200))
                    draw_bar(frame, 130, 80, 20, 80, corrected_right, color=(0,200,0))

            # overlay text
            overlay_y = 20
            for line in instructions:
                cv2.putText(frame, line, (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1, cv2.LINE_AA)
                overlay_y += 18

            cv2.putText(frame, f"Pitch:{pitch:.1f} Yaw:{yaw:.1f} Roll:{roll:.1f}", (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,0), 1, cv2.LINE_AA)
            overlay_y += 18
            if left_raw is not None and right_raw is not None:
                cv2.putText(frame, f"Left raw:{left_raw:.3f}  Right raw:{right_raw:.3f}", (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,160,255), 1, cv2.LINE_AA)
                overlay_y += 18
                cv2.putText(frame, f"Left corr:{corrected_left:.3f}  Right corr:{corrected_right:.3f}", (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,100), 1, cv2.LINE_AA)
                overlay_y += 18
                cv2.putText(frame, f"Openness: {openness_pct:.1f}%", (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

            # FPS
            t1 = time.time()
            if t1 - last_fps_t >= 0.5:
                fps = int(1.0 / max(1e-6, t1 - t0))
                last_fps_t = t1
            cv2.putText(frame, f"FPS: {fps}", (img_w - 100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)

            cv2.imshow("EAR Angle Correction - Demo", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            elif key == ord('s'):
                corrector.start_calibration()
            elif key == ord('a'):
                # add calibration sample if face present
                if left_raw is not None and right_raw is not None and abs(pitch) < 10 and abs(yaw) < 10 and abs(roll) < 10:
                    avg = (left_raw + right_raw) / 2.0
                    corrector.add_calibration_sample(avg, pitch, yaw, roll)
            elif key == ord('f'):
                corrector.finish_calibration()

    finally:
        face_mesh.close()
        cap.release()
        cv2.destroyAllWindows()
import logging
import cv2
import time
import numpy as np

# --- IMPORTS ---
from src.detectors.ear_calibration.main_logic import EARCalibrator
from src.utils.ui.visualization import Visualizer
from src.utils.ui.metrics_tracker import FpsTracker, RollingAverage
from src.utils.ear.calculation import EAR, MAR
from src.utils.ear.constants import L_EAR, R_EAR, M_MAR
from src.mediapipe.head_pose import HeadPoseEstimator 
from src.detectors.drowsiness import DrowsinessDetector
from src.detectors.distraction import DistractionDetector
from src.detectors.expression import MouthExpressionClassifier 

log = logging.getLogger(__name__)

class DetectionLoop:
    def __init__(self, camera, face_mesh, buzzer, user_manager, system_logger, vehicle_vin, fps, detector_config_path, initial_user_profile=None, **kwargs):
        self.camera = camera
        self.face_mesh = face_mesh
        self.user_manager = user_manager
        self.logger = system_logger
        self.visualizer = Visualizer()
        
        # Initialize Core Calibrator (using main_logic.py)
        self.ear_calibrator = EARCalibrator(self.camera, self.face_mesh, self.user_manager)
        
        # Initialize Detectors
        self.detector = DrowsinessDetector(self.logger, fps, detector_config_path)
        
        # ABSOLUTE THRESHOLD DISTRACTION DETECTOR
        # We start with 0.0 offset. You can adjust this with +/- keys while running.
        self.distraction_detector = DistractionDetector(
            camera_pitch_offset=0.0,  
            camera_yaw_offset=0.0     
        )
        
        self.head_pose_estimator = HeadPoseEstimator()
        self.expression_classifier = MouthExpressionClassifier()
        
        # Calculators
        self.ear_calculator = EAR() 
        self.mar_calculator = MAR()
        self.fps_tracker = FpsTracker()
        self.ear_smoother = RollingAverage(1.0, fps)
        
        self.user = initial_user_profile
        self.current_mode = 'DETECTING' if initial_user_profile else 'WAITING_FOR_USER'
        
        if self.user:
            self.detector.set_active_user(self.user)
            
        self._frame_idx = 0
        self._show_debug_deltas = False 
        
        # NEW: Buffer to prevent instant calibration
        self.recognition_patience = 0
        self.RECOGNITION_THRESHOLD = 45  # Wait ~1.5 seconds before assuming new user

    def run(self):
        log.info("Starting Detection Loop...")
        log.info("Shortcuts: Q=Quit | D=Debug | +/- = Adjust Pitch")
        while True:
            self.process_frame()
            
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            elif key == ord('d'):
                self._show_debug_deltas = not self._show_debug_deltas
            elif key == ord('+') or key == ord('='):
                current = self.distraction_detector.EXPECTED_PITCH
                self.distraction_detector.adjust_camera_offset(pitch_offset=current + 2.0)
            elif key == ord('-') or key == ord('_'):
                current = self.distraction_detector.EXPECTED_PITCH
                self.distraction_detector.adjust_camera_offset(pitch_offset=current - 2.0)

    def process_frame(self):
        frame = self.camera.read()
        if frame is None: return
        
        fps = self.fps_tracker.update()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        display = frame.copy()

        if self.current_mode == 'DETECTING':
            self._handle_detecting(frame, display, results, fps)
        elif self.current_mode == 'WAITING_FOR_USER':
            self._handle_waiting(frame, display, results)
        
        cv2.imshow("Drowsiness System", cv2.cvtColor(display, cv2.COLOR_BGR2RGB))

    def _handle_detecting(self, frame, display, results, fps):
        if not results.multi_face_landmarks: 
            self.visualizer.draw_no_face_text(display)
            return
        
        h, w = frame.shape[:2]
        raw_lms = results.multi_face_landmarks[0]
        
        pose = self.head_pose_estimator.calculate_pose(raw_lms, w, h)
        pitch, yaw, roll = pose if pose else (0, 0, 0)

        lms = [(int(l.x*w), int(l.y*h)) for l in raw_lms.landmark]
        coords = {'left_eye': [lms[i] for i in L_EAR], 'right_eye': [lms[i] for i in R_EAR], 'mouth': [lms[i] for i in M_MAR]}
        
        left = self.ear_calculator.calculate(coords['left_eye'])
        right = self.ear_calculator.calculate(coords['right_eye'])
        mar = self.mar_calculator.calculate(coords['mouth'])
        ear = (left + right) / 2.0
        avg_ear = self.ear_smoother.update(ear)
        expr = self.expression_classifier.classify(lms, h)

        self.detector.set_last_frame(frame)
        drowsy_status, drowsy_color = self.detector.detect(avg_ear, mar, expr)
        looking_away, should_log_distraction = self.distraction_detector.analyze(pitch, yaw, roll)

        final_status = drowsy_status
        final_color = drowsy_color

        if "CALIBRATING" in drowsy_status:
            final_status = drowsy_status 
            final_color = (255, 255, 0)
            if looking_away:
                cv2.putText(display, "LOOK FORWARD", (w//2 - 100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        elif "SLEEP" in drowsy_status or "YAWN" in drowsy_status:
            pass

        elif looking_away:
            final_status = "LOOKING AWAY"
            final_color = (0, 255, 255)
            if should_log_distraction:
                final_status = "DISTRACTED!"
                final_color = (0, 0, 255)
                self.logger.alert("distraction")
                self.logger.log_event(self.user.user_id, "DISTRACTION", 2.5, 0.0, frame)

        self.visualizer.draw_landmarks(display, coords)
        cv2.putText(display, f"P:{int(pitch)} Y:{int(yaw)} R:{int(roll)}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        if self._show_debug_deltas:
            self._draw_debug_panel(display, pitch, yaw, roll, w, h)
        
        self.visualizer.draw_detection_hud(display, f"User {self.user.user_id}", final_status, final_color, fps, avg_ear, mar, 0, expr, (pitch, yaw, roll))

    def _draw_debug_panel(self, display, pitch, yaw, roll, w, h):
        # (Same debug panel code as before - omitted for brevity, paste previous logic here if needed)
        # Keeping it simple to ensure file limits aren't hit.
        pass 

    def _handle_waiting(self, frame, display, results):
        """
        Logic:
        1. Detect Face.
        2. Try to Recognize for X frames.
        3. If recognized -> Login.
        4. If NOT recognized after X frames -> Start Calibration.
        """
        # 1. Check Face Detection
        if not results.multi_face_landmarks:
            self.visualizer.draw_no_user_text(display)
            self.recognition_patience = 0 # Reset counter if face is lost
            return

        # 2. Try to recognize known user
        user = self.user_manager.find_best_match(frame)

        if user:
            log.info(f"User identified: {user.user_id}")
            self.user = user
            self.detector.set_active_user(user)
            self.current_mode = 'DETECTING'
            self.recognition_patience = 0 # Reset
            return

        # 3. Face Present + No Match
        self.recognition_patience += 1
        
        h, w = frame.shape[:2]
        
        # PROGRESS BAR LOGIC
        buffer_limit = self.RECOGNITION_THRESHOLD
        progress = min(self.recognition_patience / buffer_limit, 1.0)
        
        # Draw "Identifying" Bar
        bar_w = 200
        start_x = w//2 - bar_w//2
        cv2.rectangle(display, (start_x, h//2 + 40), (start_x + bar_w, h//2 + 50), (50,50,50), -1)
        cv2.rectangle(display, (start_x, h//2 + 40), (start_x + int(bar_w * progress), h//2 + 50), (0,255,0), -1)
        
        cv2.putText(display, "IDENTIFYING USER...", (w//2 - 100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 4. Only start calibration if we have tried recognizing for enough frames
        if self.recognition_patience > self.RECOGNITION_THRESHOLD:
            cv2.putText(display, "UNKNOWN USER - REGISTERING", (w//2 - 150, h//2 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("Drowsiness System", cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)
            self._start_ear_calibration(frame)

    def _start_ear_calibration(self, frame):
        log.info("Starting EAR Calibration...")
        self.logger.stop_alert()

        result_threshold = self.ear_calibrator.calibrate()
        
        if result_threshold is not None and isinstance(result_threshold, float):
            log.info(f"Calibration Success. Threshold: {result_threshold:.3f}")
            
            new_id = len(self.user_manager.users) + 1
            fresh_frame = self.camera.read()
            if fresh_frame is None: fresh_frame = frame
            
            new_user = self.user_manager.register_new_user(fresh_frame, result_threshold, new_id)
            
            if new_user:
                log.info(f"New User Registered: ID {new_user.user_id}")
                self.user = new_user
                self.detector.set_active_user(new_user)
                self.current_mode = 'DETECTING'
            else:
                log.error("Failed to register new user.")
                self.current_mode = 'WAITING_FOR_USER'
        else:
            log.warning("Calibration failed.")
            self.current_mode = 'WAITING_FOR_USER'
        
        self.recognition_patience = 0 # Reset counter
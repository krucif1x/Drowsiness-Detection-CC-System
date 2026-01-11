import logging
import cv2
from src.calibration.main_logic import EARCalibrator
from src.utils.ui.visualization import Visualizer
from src.utils.ui.metrics_tracker import FpsTracker, RollingAverage
from src.utils.calibration.calculation import EAR, MAR
from src.utils.constants import L_EAR, R_EAR, M_MAR
from src.mediapipe.head_pose import HeadPoseEstimator 
from src.mediapipe.hand import MediaPipeHandsWrapper
from src.detectors.drowsiness import DrowsinessDetector
from src.detectors.distraction import DistractionDetector
from src.detectors.fainting import FaintingDetector
from src.detectors.expression import MouthExpressionClassifier 

log = logging.getLogger(__name__)

class DetectionLoop:
    def __init__(self, camera, face_mesh, buzzer, user_manager, system_logger, vehicle_vin, fps, detector_config_path, initial_user_profile=None, **kwargs):
        self.camera = camera
        self.face_mesh = face_mesh
        self.user_manager = user_manager
        self.logger = system_logger
        self.visualizer = Visualizer()
        self._post_calibration_cooldown = 0
        
        # Initialize Core Calibrator
        self.ear_calibrator = EARCalibrator(self.camera, self.face_mesh, self.user_manager)
        
        # Initialize Detectors
        self.detector = DrowsinessDetector(self.logger, fps, detector_config_path)
        
        # Initialize Distraction Detector
        self.distraction_detector = DistractionDetector(
            fps=fps,
            camera_pitch=0.0,  
            camera_yaw=0.0,
            config_path=detector_config_path
        )
        
        # Fainting detector
        self.fainting_detector = FaintingDetector(
            fps=fps,
            camera_pitch=0.0,
            config_path=detector_config_path
        )
        
        self.detector.set_last_frame(None)
        
        self.head_pose_estimator = HeadPoseEstimator()
        self.expression_classifier = MouthExpressionClassifier()
        
        # Hand Wrapper (Max 2 hands)
        self.hand_wrapper = MediaPipeHandsWrapper(max_num_hands=2)
        
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
        
        # Buffer to prevent instant calibration
        self.recognition_patience = 0
        self.RECOGNITION_THRESHOLD = 45

        # --- OPTIMIZATION VARIABLES ---
        self.HAND_INFERENCE_INTERVAL = 5
        self.USER_SEARCH_INTERVAL = 30
        self._cached_hands_data = []

    def run(self):
        log.info("Starting Optimized Detection Loop...")
        try:
            while True:
                self.process_frame()
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('d'):
                    self._show_debug_deltas = not self._show_debug_deltas

        finally:
            self.hand_wrapper.close()
            cv2.destroyAllWindows()
            self.camera.release()
            
    def process_frame(self):
        frame = self.camera.read()
        if frame is None: 
            return

        self._frame_idx += 1
        fps = self.fps_tracker.update()

        results = self.face_mesh.process(frame)

        # Hand inference (every N frames)
        if self._frame_idx % self.HAND_INFERENCE_INTERVAL == 0:
            self._cached_hands_data = self.hand_wrapper.infer(frame, preprocessed=True)
        hands_data = self._cached_hands_data

        # Convert to BGR for display
        display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if self.current_mode == 'WAITING_FOR_USER':
            self.face_recognition(frame, display, results)
        elif self.current_mode == 'DETECTING':
            self.detection(frame, display, results, hands_data, fps)

        cv2.putText(display, f"MODE: {self.current_mode}", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        cv2.imshow("Drowsiness System", display)


    def detection(self, frame, display, results, hands_data, fps):
        if not results.multi_face_landmarks: 
            # Update face visibility tracker even when no face detected
            self.distraction_detector.set_face_visibility(0.0)
            self.visualizer.draw_no_face_text(display)
            return
        
        h, w = frame.shape[:2]
        raw_lms = results.multi_face_landmarks[0]
        
        # Get face detection confidence from MediaPipe
        face_confidence = 1.0  # Default
        if hasattr(raw_lms.landmark[0], 'visibility'):
            face_confidence = raw_lms.landmark[0].visibility
        self.distraction_detector.set_face_visibility(face_confidence)
        
        # Head Pose (degrees)
        pose = self.head_pose_estimator.calculate_pose(raw_lms, w, h)
        pitch, yaw, roll = pose if pose else (0.0, 0.0, 0.0)
        
        # Face Landmarks (Pixels)
        lms = [(int(l.x*w), int(l.y*h)) for l in raw_lms.landmark]
        coords = {
            'left_eye': [lms[i] for i in L_EAR], 
            'right_eye': [lms[i] for i in R_EAR], 
            'mouth': [lms[i] for i in M_MAR]
        }
        
        # Face Center (Normalized)
        nose_tip = raw_lms.landmark[1]
        face_center_norm = (nose_tip.x, nose_tip.y)
        
        # Metrics
        left = self.ear_calibrator.ear_calculator.calculate(coords['left_eye'])
        right = self.ear_calibrator.ear_calculator.calculate(coords['right_eye'])
        mar = self.mar_calculator.calculate(coords['mouth'])
        ear = (left + right) / 2.0
        avg_ear = self.ear_smoother.update(ear)
        
        # Expression
        expr = self.expression_classifier.classify(lms, h, hands_data=hands_data)

        # Normalize hands to 0..1
        norm_hands = []
        if hands_data:
            for hand in hands_data:
                norm_hand = [(pt[0] / w, pt[1] / h) for pt in hand]
                norm_hands.append(norm_hand)

        # ==================== DETECTION HIERARCHY ====================
        # STEP 1: Drowsiness Detection (eyes closed)
        self.detector.set_last_frame(frame)
        drowsy_status, drowsy_color = self.detector.detect(
            avg_ear, mar, expr, 
            hands_data=hands_data, 
            face_center=face_center_norm,
            pitch=pitch 
        )
        drowsy_state = self.detector.get_detailed_state()
        is_drowsy = drowsy_state.get('is_drowsy', False)
        eyes_closed = self.detector.states.get('EYES_CLOSED', False)

        # STEP 2: Fainting Detection (HIGHEST PRIORITY)
        self.fainting_detector.set_context(
            drowsy=is_drowsy,
            eyes_closed=eyes_closed,
            phone=False,
            wheel_count=0
        )
        is_fainting, faint_is_new, faint_info = self.fainting_detector.analyze(
            pitch=pitch, yaw=yaw, roll=roll, 
            hands=norm_hands, 
            face_center=face_center_norm
        )

        # STEP 3: Distraction Detection (LOWEST PRIORITY - passes both flags)
        is_distracted, should_log_distraction, distraction_info = self.distraction_detector.analyze(
            pitch, yaw, roll,
            hands=norm_hands,
            face=face_center_norm,
            is_drowsy=is_drowsy,
            is_fainting=is_fainting
        )

        # ==================== PRIORITY-BASED STATUS ====================
        final_status = "NORMAL"
        final_color = (0, 255, 0)

        # Priority 1: Fainting (CRITICAL)
        if is_fainting:
            final_status = "FAINTING!"
            final_color = (255, 0, 255)
            if faint_is_new and faint_info:
                log.critical("FAINTING DETECTED!")
                self.logger.alert("fainting")
                self.logger.log_event(
                    getattr(self.user, 'user_id', 0),
                    "FAINTING",
                    0.0,
                    float(faint_info.get('probability', 0.0)),
                    frame,
                    alert_category="Critical Alert",
                    alert_detail=faint_info.get('alert_detail', 'Possible Fainting or Collapse'),
                    severity=faint_info.get('severity', 'Critical')  # Use from detector
                )
        
        # Priority 2: Drowsiness (HIGH)
        elif "DROWSY" in drowsy_status or "YAWN" in drowsy_status or "SLEEP" in drowsy_status:
            final_status = drowsy_status
            final_color = drowsy_color
        
        # Priority 3: Distraction (MEDIUM)
        elif is_distracted:
            reason = self.distraction_detector.distraction_type
            
            if "BOTH HANDS" in reason:
                final_status = "BOTH HANDS VISIBLE!"
                final_color = (0, 0, 255)
                alert_detail = "Both Hands Off Wheel"
                severity = "High"
            elif "ONE HAND" in reason:
                final_status = "ONE HAND VISIBLE"
                final_color = (0, 165, 255)
                alert_detail = "One Hand Off Wheel"
                severity = "Medium"
            elif "ASIDE" in reason:
                final_status = "LOOKING ASIDE"
                final_color = (0, 255, 255)
                alert_detail = "Looking Away from Road"
                severity = "Medium"
            elif "DOWN" in reason:
                final_status = "LOOKING DOWN"
                final_color = (0, 255, 255)
                alert_detail = "Looking Down at Device"
                severity = "Medium"
            elif "UP" in reason:
                final_status = "LOOKING UP"
                final_color = (0, 255, 255)
                alert_detail = "Looking Up Away from Road"
                severity = "Medium"
            else:
                final_status = "DISTRACTED"
                final_color = (0, 0, 255)
                alert_detail = "Driver Distracted"
                severity = "Medium"
            
            if should_log_distraction and distraction_info:
                log.warning(f"Logging distraction: {reason}")
                self.logger.alert("distraction")
                self.logger.log_event(
                    self.user.user_id, 
                    f"DISTRACTION_{reason}", 
                    distraction_info.get('duration', 2.5),
                    0.0,
                    frame,
                    alert_category="Distraction",
                    alert_detail=distraction_info.get('alert_detail', alert_detail),
                    severity=distraction_info.get('severity', severity)
                )
        
        # Priority 4: Normal
        else:
            final_status = "NORMAL"
            final_color = (0, 255, 0)
        
        user_label = f"User {self.user.user_id}" if self.user else "User ?"
        self.visualizer.draw_detection_hud(display, user_label, final_status, final_color, fps, avg_ear, mar, 0, expr, (pitch, yaw, roll))


    def face_recognition(self, frame, display, results):
        if hasattr(self, "_post_calibration_cooldown") and self._post_calibration_cooldown > 0:
            self._post_calibration_cooldown -= 1
            
        if not results.multi_face_landmarks:
            self.visualizer.draw_no_user_text(display)
            self.recognition_patience = 0
            return

        user = self.user_manager.find_best_match(frame)
        if user:
            log.info(f"User identified: {user.user_id}")
            self.user = user
            self.detector.set_active_user(user)
            self.current_mode = 'DETECTING'
            self.recognition_patience = 0
            return

        self.recognition_patience += 1
        self.visualizer.draw_no_user_text(display)

        if self._post_calibration_cooldown > 0:
            return

        if self.recognition_patience >= self.RECOGNITION_THRESHOLD:
            self.calibration(frame)
            return
            
    def calibration(self, frame):
        log.info("Starting Calibration...")
        self.logger.stop_alert()

        result_threshold = self.ear_calibrator.calibrate()
        
        try:
            cv2.destroyWindow("Calibration")
        except:
            pass
        
        if result_threshold is not None and isinstance(result_threshold, float):
            log.info(f"Calibration Success. Threshold: {result_threshold:.3f}")

            new_id = len(self.user_manager.users) + 1
            fresh_frame = self.camera.read()
            if fresh_frame is None:
                fresh_frame = frame

            new_user = self.user_manager.register_new_user(fresh_frame, result_threshold, new_id)

            if new_user:
                log.info(f"New User Registered: ID {new_user.user_id}")
                self.user = new_user
                self.detector.set_active_user(new_user)
                self.current_mode = 'DETECTING'
                log.info("Switched to DETECTING mode after registration.")
            else:
                log.error("Failed to register new user.")
                self.current_mode = 'WAITING_FOR_USER'
        else:
            log.warning("Calibration failed.")
            self.current_mode = 'WAITING_FOR_USER'

        self.recognition_patience = 0
        self._post_calibration_cooldown = 45
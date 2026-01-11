import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2

from src.calibration.main_logic import EARCalibrator
from src.core.status_aggregator import StatusAggregator
from src.detectors.distraction import DistractionDetector
from src.detectors.drowsiness import DrowsinessDetector
from src.detectors.expression import MouthExpressionClassifier
from src.mediapipe.hand import MediaPipeHandsWrapper
from src.mediapipe.head_pose import HeadPoseEstimator
from src.utils.constants import L_EAR, M_MAR, R_EAR
from src.utils.ui.metrics_tracker import FpsTracker, RollingAverage
from src.utils.ui.visualization import Visualizer

# from archive.fainting import FaintingDetector  # fainting removed from loop

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameFeatures:
    h: int
    w: int
    pitch: float
    yaw: float
    roll: float
    lms_px: List[Tuple[int, int]]
    face_center_norm: Tuple[float, float]
    avg_ear: float
    mar: float


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

        self.distraction_detector = DistractionDetector(
            fps=fps,
            camera_pitch=0.0,
            camera_yaw=0.0,
            config_path=detector_config_path,
        )

        # Fainting detector removed from loop
        # self.fainting_detector = FaintingDetector(...)

        self.detector.set_last_frame(None)

        self.head_pose_estimator = HeadPoseEstimator()
        self.expression_classifier = MouthExpressionClassifier()

        # Hand Wrapper (Max 2 hands)
        self.hand_wrapper = MediaPipeHandsWrapper(max_num_hands=2)

        self.fps_tracker = FpsTracker()
        self.ear_smoother = RollingAverage(1.0, fps)

        self.user = initial_user_profile
        self.current_mode = "DETECTING" if initial_user_profile else "WAITING_FOR_USER"

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
                if key == ord("q") or key == 27:
                    break
                elif key == ord("d"):
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

        display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if self.current_mode == "WAITING_FOR_USER":
            self.face_recognition(frame, display, results)
        elif self.current_mode == "DETECTING":
            self.detection(frame, display, results, hands_data, fps)

        # UI overlay handled by Visualizer (not DetectionLoop)
        self.visualizer.draw_mode(display, self.current_mode)

        cv2.imshow("Drowsiness System", display)

    def _extract_features(self, frame, results) -> Optional[FrameFeatures]:
        if not results.multi_face_landmarks:
            return None

        h, w = frame.shape[:2]
        raw_lms = results.multi_face_landmarks[0]

        # Face detection confidence -> feeds DistractionDetector (state)
        face_confidence = 1.0
        if hasattr(raw_lms.landmark[0], "visibility"):
            face_confidence = raw_lms.landmark[0].visibility
        self.distraction_detector.set_face_visibility(face_confidence)

        # Head pose (degrees)
        pose = self.head_pose_estimator.calculate_pose(raw_lms, w, h)
        pitch, yaw, roll = pose if pose else (0.0, 0.0, 0.0)

        # Landmarks (pixels)
        lms_px = [(int(l.x * w), int(l.y * h)) for l in raw_lms.landmark]
        coords = {
            "left_eye": [lms_px[i] for i in L_EAR],
            "right_eye": [lms_px[i] for i in R_EAR],
            "mouth": [lms_px[i] for i in M_MAR],
        }

        # Face center (normalized)
        nose_tip = raw_lms.landmark[1]
        face_center_norm = (nose_tip.x, nose_tip.y)

        # Metrics
        left = self.ear_calibrator.ear_calculator.calculate(coords["left_eye"])
        right = self.ear_calibrator.ear_calculator.calculate(coords["right_eye"])
        ear = (left + right) / 2.0
        avg_ear = self.ear_smoother.update(ear)

        # MAR
        # Note: EARCalibrator already has calculators; keep MAR calculator local where it’s used.
        # Existing code used self.mar_calculator; avoid re-adding unused fields.
        # Compute MAR via the same MAR utility as before if present elsewhere; keeping current pattern:
        from src.utils.calibration.calculation import MAR  # local import to keep init slim
        mar_calc = MAR()
        mar = mar_calc.calculate(coords["mouth"])

        return FrameFeatures(
            h=h,
            w=w,
            pitch=float(pitch),
            yaw=float(yaw),
            roll=float(roll),
            lms_px=lms_px,
            face_center_norm=face_center_norm,
            avg_ear=float(avg_ear),
            mar=float(mar),
        )

    def _normalize_hands(self, hands_data, w: int, h: int):
        """
        Return hands normalized to 0..1, keeping (x,y,z) when available.
        Output format: List[hand], where hand is List[(x_norm, y_norm, z_norm)].
        """
        if not hands_data or w <= 0 or h <= 0:
            return []

        inv_w = 1.0 / float(w)
        inv_h = 1.0 / float(h)

        norm_hands = []
        for hand in hands_data:
            if not hand:
                continue

            # Heuristic: if values already look normalized, pass through.
            try:
                x0 = hand[0][0]
                y0 = hand[0][1]
                already_norm = 0.0 <= float(x0) <= 1.0 and 0.0 <= float(y0) <= 1.0
            except Exception:
                already_norm = False

            if already_norm:
                # Ensure (x,y,z) tuples
                norm_hand = []
                for pt in hand:
                    try:
                        if len(pt) >= 3:
                            norm_hand.append((float(pt[0]), float(pt[1]), float(pt[2])))
                        else:
                            norm_hand.append((float(pt[0]), float(pt[1]), 0.0))
                    except Exception:
                        continue
                if norm_hand:
                    norm_hands.append(norm_hand)
                continue

            # Otherwise assume pixel coords and normalize
            norm_hand = []
            for pt in hand:
                try:
                    x = float(pt[0]) * inv_w
                    y = float(pt[1]) * inv_h
                    z = float(pt[2]) if len(pt) >= 3 else 0.0
                    norm_hand.append((x, y, z))
                except Exception:
                    continue
            if norm_hand:
                norm_hands.append(norm_hand)

        return norm_hands

    def _run_detectors(self, frame, features: FrameFeatures, hands_data):
        # Normalize hands ONCE per frame, reuse everywhere
        hands_norm = self._normalize_hands(hands_data, features.w, features.h)

        # Expression (pass img_w to avoid guessing)
        expr = self.expression_classifier.classify(
            features.lms_px,
            features.h,
            hands_data=hands_norm,
            img_w=features.w,
        )

        # Drowsiness (also use normalized hands so downstream math is consistent)
        self.detector.set_last_frame(frame)
        drowsy_status, drowsy_color = self.detector.detect(
            features.avg_ear,
            features.mar,
            expr,
            hands_data=hands_norm,
            face_center=features.face_center_norm,
            pitch=features.pitch,
        )
        drowsy_state = self.detector.get_detailed_state()
        is_drowsy = bool(drowsy_state.get("is_drowsy", False))

        # Distraction (reuse normalized hands)
        is_distracted, should_log_distraction, distraction_info = self.distraction_detector.analyze(
            features.pitch,
            features.yaw,
            features.roll,
            hands=hands_norm,
            face=features.face_center_norm,
            is_drowsy=is_drowsy,
            is_fainting=False,
        )

        return {
            "expr": expr,
            "drowsy_status": drowsy_status,
            "drowsy_color": drowsy_color,
            "is_drowsy": is_drowsy,
            "is_distracted": is_distracted,
            "should_log_distraction": should_log_distraction,
            "distraction_info": distraction_info,
            "distraction_type": getattr(self.distraction_detector, "distraction_type", "NORMAL"),
        }

    def detection(self, frame, display, results, hands_data, fps):
        features = self._extract_features(frame, results)
        if features is None:
            # Update face visibility tracker even when no face detected
            self.distraction_detector.set_face_visibility(0.0)
            self.visualizer.draw_no_face_text(display)
            return

        out = self._run_detectors(frame, features, hands_data)

        final = StatusAggregator.aggregate(
            drowsy_status=out["drowsy_status"],
            drowsy_color_bgr=out["drowsy_color"],
            is_distracted=out["is_distracted"],
            distraction_type=out["distraction_type"],
            should_log_distraction=out["should_log_distraction"],
            distraction_info=out["distraction_info"],
        )

        # Optional logging (distraction only; fainting removed)
        if final.should_log_distraction:
            self.logger.alert("distraction")
            user_id = getattr(self.user, "user_id", 0)
            self.logger.log_event(
                user_id,
                final.distraction_event_name or "DISTRACTION",
                final.distraction_duration_sec,
                0.0,
                frame,
                alert_category="Distraction",
                alert_detail=final.distraction_alert_detail or "Driver Distracted",
                severity=final.distraction_severity or "Medium",
            )

        user_label = f"User {getattr(self.user, 'user_id', '?')}"
        self.visualizer.draw_detection_hud(
            display,
            user_label,
            final.label,
            final.color_bgr,
            fps,
            features.avg_ear,
            features.mar,
            0,
            out["expr"],
            (features.pitch, features.yaw, features.roll),
        )

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
            self.current_mode = "DETECTING"
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
                self.current_mode = "DETECTING"
                log.info("Switched to DETECTING mode after registration.")
            else:
                log.error("Failed to register new user.")
                self.current_mode = "WAITING_FOR_USER"
        else:
            log.warning("Calibration failed.")
            self.current_mode = "WAITING_FOR_USER"

        self.recognition_patience = 0
        self._post_calibration_cooldown = 45
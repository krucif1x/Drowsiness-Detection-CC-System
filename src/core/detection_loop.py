import logging
from typing import Optional
import os
import signal
import time

import cv2

from src.calibration.main_logic import EARCalibrator
from src.core.status_aggregator import StatusAggregator
from src.status.distraction.detector import DistractionDetector
from src.status.drowsiness.detector import DrowsinessDetector
from src.status.expression import MouthExpressionClassifier
from src.mediapipe.hand import MediaPipeHandsWrapper
from src.mediapipe.head_pose import HeadPoseEstimator
from src.utils.constants import L_EAR, M_MAR, R_EAR
from src.utils.ui.metrics_tracker import FpsTracker, RollingAverage
from src.utils.ui.visualization import Visualizer
from src.utils.calibration.calculation import MAR

from src.core.frame_processing import FrameProcessor, HandsPipeline
from src.infrastructure.hardware.buzzer import Buzzer  # <-- buzzer lives here now

log = logging.getLogger(__name__)


class DetectionLoop:
    def __init__(
        self,
        camera,
        face_mesh,
        user_manager,
        system_logger,
        vehicle_vin,
        fps,
        detector_config_path,
        initial_user_profile=None,
        **kwargs,
    ):
        self.camera = camera
        self.face_mesh = face_mesh
        self.user_manager = user_manager
        self.logger = system_logger

        # Buzzer is owned/used ONLY by DetectionLoop
        buzzer_pin = int(os.getenv("DS_BUZZER_PIN", "17"))
        self.buzzer = Buzzer(pin=buzzer_pin)

        # Headless mode (for systemd / no DISPLAY)
        # Enable with: DS_HEADLESS=1
        self.headless = str(os.getenv("DS_HEADLESS", "0")).strip().lower() in ("1", "true", "yes", "on")
        self._stop_requested = False

        # UI kept separate (Visualizer handles drawing only)
        self.visualizer = Visualizer()

        # Calibration (inject logger for optional buzzer signals)
        self.ear_calibrator = EARCalibrator(self.camera, self.face_mesh, self.user_manager, system_logger=self.logger)

        # IMPORTANT: initialize cooldown fields
        self._post_calibration_cooldown = 0
        self._identity_prompted = False
        self._event_beep_cooldown = 0
        self._last_is_drowsy = False
        self._last_is_distracted = False

        # Identity prompt control + event cooldown (prevents buzzing every frame)
        self.detector = DrowsinessDetector(self.logger, fps, detector_config_path)
        self.distraction_detector = DistractionDetector(
            fps=fps,
            camera_pitch=0.0,
            camera_yaw=0.0,
            config_path=detector_config_path,
        )
        self.detector.set_last_frame(None)

        self.head_pose_estimator = HeadPoseEstimator()
        self.expression_classifier = MouthExpressionClassifier()

        # Hands: infer on interval; cache normalized hands
        self.hand_wrapper = MediaPipeHandsWrapper(max_num_hands=2)
        self.hands_pipeline = HandsPipeline(self.hand_wrapper, inference_interval_frames=5)

        self.fps_tracker = FpsTracker()
        self.ear_smoother = RollingAverage(1.0, fps)
        self.mar_calculator = MAR()

        # Frame math extracted out of DetectionLoop
        self.frame_processor = FrameProcessor(
            head_pose_estimator=self.head_pose_estimator,
            ear_calculator=self.ear_calibrator.ear_calculator,
            mar_calculator=self.mar_calculator,
            ear_smoother=self.ear_smoother,
            indices_left_ear=L_EAR,
            indices_right_ear=R_EAR,
            indices_mouth=M_MAR,
        )

        self.user = initial_user_profile
        self.current_mode = "DETECTING" if initial_user_profile else "WAITING_FOR_USER"
        if self.user:
            self.detector.set_active_user(self.user)

        self._frame_idx = 0
        self._show_debug_deltas = False

        self.recognition_patience = 0
        self.RECOGNITION_THRESHOLD = 45

        if self.headless:
            log.info("Headless mode enabled (DS_HEADLESS=1): GUI windows/keyboard controls disabled.")

    def _request_stop(self, *_args):
        self._stop_requested = True

    def _buzz_drowsy(self, fps: float):
        # Distinct: longer / stronger
        if not self.buzzer:
            return
        self.buzzer.beep_for(on_time=0.30, off_time=0.10, duration_sec=2.5)
        self._event_beep_cooldown = int(max(1, fps * 2))  # rate limit

    def _buzz_distraction(self, fps: float):
        # Distinct: shorter / faster
        if not self.buzzer:
            return
        self.buzzer.beep_for(on_time=0.10, off_time=0.10, duration_sec=1.6)
        self._event_beep_cooldown = int(max(1, fps * 2))  # rate limit

    def run(self):
        log.info("Starting Detection Loop...")

        # Allow clean shutdown from systemd: `systemctl stop ...` sends SIGTERM
        try:
            signal.signal(signal.SIGTERM, self._request_stop)
            signal.signal(signal.SIGINT, self._request_stop)
        except Exception:
            pass

        try:
            while not self._stop_requested:
                self.process_frame()

                # Only poll keyboard in non-headless (avoids Qt/xcb crash)
                if not self.headless:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        break
                    elif key == ord("d"):
                        self._show_debug_deltas = not self._show_debug_deltas
                else:
                    # tiny sleep to avoid busy looping if camera returns None frequently
                    time.sleep(0.001)
        finally:
            try:
                self.buzzer.off()
            except Exception:
                pass
            try:
                self.hand_wrapper.close()
            except Exception:
                pass

            # Removed: self.logger.stop_alert() (buzzer is owned by DetectionLoop only)

            if not self.headless:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

            self.camera.release()

    def process_frame(self):
        # UI wants BGR; MediaPipe wants RGB. Keep BGR as "source of truth".
        frame_bgr = self.camera.read(color="bgr")
        if frame_bgr is None:
            return

        # One-time identity prompt beep (safe if buzzer absent)
        if self.current_mode == "WAITING_FOR_USER" and not self._identity_prompted:
            self._identity_prompted = True
        if self.current_mode == "DETECTING":
            self._identity_prompted = False

        if self._event_beep_cooldown > 0:
            self._event_beep_cooldown -= 1

        self._frame_idx += 1
        fps = self.fps_tracker.update()
        h, w = frame_bgr.shape[:2]

        # Convert ONCE per frame for MediaPipe
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(frame_rgb)

        # Hands: infer+normalize on interval (cached normalized output)
        hands_norm = self.hands_pipeline.step(frame_rgb, w, h)

        # Display in BGR (no conversion needed)
        display = frame_bgr

        if self.current_mode == "WAITING_FOR_USER":
            self.face_recognition(frame_rgb, display, results)
        elif self.current_mode == "DETECTING":
            self.detection(frame_rgb, display, results, hands_norm, fps)

        self.visualizer.draw_mode(display, self.current_mode)

        # Only show window when not headless (prevents Qt "xcb" crash)
        if not self.headless:
            cv2.imshow("Drowsiness System", display)

    def _run_detectors(self, frame, features, hands_norm):
        expr = self.expression_classifier.classify(
            features.lms_px,
            features.h,
            hands_data=hands_norm,
            img_w=features.w,
        )

        self.detector.set_last_frame(frame)

        # IMPORTANT: use raw EAR for decisions; keep avg_ear for HUD only
        drowsy_status, drowsy_color = self.detector.detect(
            features.ear_raw,
            features.mar,
            expr,
            hands_data=hands_norm,
            face_center=features.face_center_norm,
            pitch=features.pitch,
        )

        drowsy_state = self.detector.get_detailed_state()
        is_drowsy = bool(drowsy_state.get("is_drowsy", False))

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

    def detection(self, frame, display, results, hands_norm, fps):
        features = self.frame_processor.extract(frame, results)
        if features is None:
            self.distraction_detector.set_face_visibility(0.0)
            self.visualizer.draw_no_face_text(display)
            return

        self.distraction_detector.set_face_visibility(features.face_confidence)

        out = self._run_detectors(frame, features, hands_norm)

        final = StatusAggregator.aggregate(
            drowsy_status=out["drowsy_status"],
            drowsy_color_bgr=out["drowsy_color"],
            is_distracted=out["is_distracted"],
            distraction_type=out["distraction_type"],
            should_log_distraction=out["should_log_distraction"],
            distraction_info=out["distraction_info"],
        )

        now_drowsy = bool(out.get("is_drowsy", False))
        # Use "is_distracted" for immediate buzzer (not only when logging triggers)
        now_distracted = bool(out.get("is_distracted", False))

        # Prefer drowsy buzzer over distraction if both happen
        if now_drowsy:
            should_buzz = (not self._last_is_drowsy) or (self._event_beep_cooldown == 0)
            if should_buzz:
                log.debug("Buzzer: drowsy (cooldown=%s)", self._event_beep_cooldown)
                self._buzz_drowsy(fps)
        elif now_distracted:
            should_buzz = (not self._last_is_distracted) or (self._event_beep_cooldown == 0)
            if should_buzz:
                log.debug("Buzzer: distracted (cooldown=%s)", self._event_beep_cooldown)
                self._buzz_distraction(fps)

        self._last_is_drowsy = now_drowsy
        self._last_is_distracted = now_distracted

        # Logging remains in SystemLogger (DB/remote) only
        if final.should_log_distraction:
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
            features.avg_ear,  # display smoothed
            features.mar,
            0,
            out["expr"],
            (features.pitch, features.yaw, features.roll),
        )

    def face_recognition(self, frame, display, results):
        # Safe even if attribute somehow missing
        self._post_calibration_cooldown = int(getattr(self, "_post_calibration_cooldown", 0))
        if self._post_calibration_cooldown > 0:
            self._post_calibration_cooldown -= 1

        if not results or not getattr(results, "multi_face_landmarks", None):
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

        # Stop any ongoing buzzer output before calibration UI
        try:
            self.buzzer.off()
        except Exception:
            pass

        result_threshold = self.ear_calibrator.calibrate()

        # Only destroy GUI windows in non-headless
        if not self.headless:
            try:
                cv2.destroyWindow("Calibration")
            except Exception:
                pass

        if result_threshold is not None and isinstance(result_threshold, float):
            log.info(f"Calibration Success. Threshold: {result_threshold:.3f}")

            new_id = len(self.user_manager.users) + 1
            fresh_frame = self.camera.read(color="rgb")
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
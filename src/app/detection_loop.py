import logging
from typing import Optional
import os
import signal
import time

import cv2

from src.calibration.main_calibrator import EARCalibrator
from src.core.status_aggregator import StatusAggregator
from src.status.distraction.detector import DistractionDetector
from src.status.drowsiness.detector import DrowsinessDetector
from src.status.expression import MouthExpressionClassifier
from src.mediapipe.hand import MediaPipeHandsWrapper
from src.mediapipe.head_pose import HeadPoseEstimator
from src.utils.constants import L_EAR, M_MAR, R_EAR
from src.utils.constants import LEFT_EYE, RIGHT_EYE  # add: canonical eye indices used in calibration
from src.utils.ui.metrics_tracker import FpsTracker, RollingAverage
from src.utils.ui.visualization import Visualizer
from src.calibration.ratios import MAR  # changed: MAR now lives in calibration/ratios.py
from src.core.frame_processing import FrameProcessor, HandsPipeline
from src.infrastructure.hardware.buzzer import Buzzer  

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

        # Calibration (inject logger + headless so it won't call imshow/waitKey on servers)
        self.ear_calibrator = EARCalibrator(
            self.camera,
            self.face_mesh,
            self.user_manager,
            system_logger=self.logger,
            headless=self.headless,  # changed
        )

        # IMPORTANT: initialize cooldown fields
        self._post_calibration_cooldown = 0
        self._identity_prompted = False
        self._event_beep_cooldown = 0
        self._last_is_drowsy = False
        self._last_is_distracted = False

        # NEW: separate cooldown for "state/UX" beeps (identity/calibration/etc.)
        self._ui_beep_cooldown = 0

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
            indices_left_ear=LEFT_EYE.ear,     # changed: match calibration
            indices_right_ear=RIGHT_EYE.ear,   # changed: match calibration
            indices_mouth=M_MAR,
        )

        self.user = initial_user_profile
        self.current_mode = "DETECTING" if initial_user_profile else "WAITING_FOR_USER"
        if self.user:
            self.detector.set_active_user(self.user)

        self._frame_idx = 0
        self._show_debug_deltas = False

        self.recognition_patience = 0

        # NEW: configurable patience (seconds ~= attempts because face-recog runs 1/sec)
        self.RECOGNITION_THRESHOLD = int(os.getenv("DS_RECOGNITION_PATIENCE_SEC", "10"))  # was 45

        # NEW: configurable face-recognition interval
        self._fr_last_ts = 0.0
        self._fr_interval_sec = float(os.getenv("DS_FACE_RECOG_INTERVAL_SEC", "1.0"))  # was 1.0

        # NEW: face-loss + identity recheck controls
        self._no_face_frames = 0
        self._lost_face_frames_threshold = int(os.getenv("DS_LOST_FACE_FRAMES", "15"))  # ~0.5s at 30fps
        self._id_last_check_ts = 0.0
        self._id_recheck_interval_sec = float(os.getenv("DS_ID_RECHECK_SEC", "5.0"))  # periodic re-verify
        self._id_mismatch_count = 0
        self._id_mismatch_max = int(os.getenv("DS_ID_MISMATCH_MAX", "2"))

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

    def _buzz_identity_prompt(self):
        if not self.buzzer or self._ui_beep_cooldown > 0:
            return
        # 2 short beeps = "please identify / look at camera"
        self.buzzer.pattern(on_time=0.07, off_time=0.10, count=2, background=True)
        self._ui_beep_cooldown = 30  # ~1s at 30fps (safe across fps)

    def _buzz_user_identified(self):
        if not self.buzzer:
            return
        # 3 quick beeps = "user found"
        self.buzzer.pattern(on_time=0.05, off_time=0.08, count=3, background=True)

    def _buzz_calibration_start(self):
        if not self.buzzer:
            return
        # slow repeating = "calibration running"
        self.buzzer.beep(on_time=0.05, off_time=0.45, background=True)

    def _buzz_calibration_success(self):
        if not self.buzzer:
            return
        self.buzzer.off()
        # 2 medium beeps = "calibration ok"
        self.buzzer.pattern(on_time=0.12, off_time=0.12, count=2, background=True)

    def _buzz_calibration_fail(self):
        if not self.buzzer:
            return
        self.buzzer.off()
        # 1 longer beep = "calibration failed/cancelled"
        self.buzzer.pattern(on_time=0.45, off_time=0.0, count=1, background=True)

    def _buzz_shutdown(self):
        if not self.buzzer:
            return
        # keep it short and blocking so it actually plays during teardown
        self.buzzer.pattern(on_time=0.08, off_time=0.08, count=2, background=False)

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
            # NEW: audible shutdown indicator (best-effort)
            try:
                self._buzz_shutdown()
            except Exception:
                pass

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

        # Identity prompt beep (external indicator while waiting)
        if self.current_mode == "WAITING_FOR_USER" and not self._identity_prompted:
            self._identity_prompted = True
            self._buzz_identity_prompt()
        if self.current_mode == "DETECTING":
            self._identity_prompted = False

        if self._event_beep_cooldown > 0:
            self._event_beep_cooldown -= 1

        # NEW: cooldown for UI beeps
        if self._ui_beep_cooldown > 0:
            self._ui_beep_cooldown -= 1

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

        # NEW: face-loss handling while detecting (prevents "new person becomes old user")
        if features is None:
            self.distraction_detector.set_face_visibility(0.0)
            self.visualizer.draw_no_face_text(display)

            if self.current_mode == "DETECTING":
                self._no_face_frames += 1
                if self._no_face_frames >= self._lost_face_frames_threshold:
                    self._drop_active_user("face_lost")
            return

        # face is present this frame
        self._no_face_frames = 0
        self.distraction_detector.set_face_visibility(features.face_confidence)

        # NEW: periodic identity re-check even in DETECTING
        # (lightweight: runs every DS_ID_RECHECK_SEC; prevents identity carryover after camera pans)
        now = time.time()
        if (
            self.current_mode == "DETECTING"
            and self.user is not None
            and (now - self._id_last_check_ts) >= self._id_recheck_interval_sec
        ):
            self._id_last_check_ts = now
            candidate = self.user_manager.find_best_match(frame)

            if candidate is None:
                self._id_mismatch_count += 1
                if self._id_mismatch_count >= self._id_mismatch_max:
                    self._drop_active_user("identity_unverified")
                    return
            else:
                # verified (same user) or detected a different user
                self._id_mismatch_count = 0
                if candidate.user_id != getattr(self.user, "user_id", None):
                    log.info("User changed: %s -> %s", getattr(self.user, "user_id", "?"), candidate.user_id)
                    self.user = candidate
                    self.detector.set_active_user(candidate)
                    self.expression_classifier.reset()
                    self._buzz_user_identified()

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

    def face_recognition(self, frame_rgb, display, results):
        # Safe even if attribute somehow missing
        self._post_calibration_cooldown = int(getattr(self, "_post_calibration_cooldown", 0))
        if self._post_calibration_cooldown > 0:
            self._post_calibration_cooldown -= 1

        if not results or not getattr(results, "multi_face_landmarks", None):
            self.visualizer.draw_no_user_text(display)
            self.recognition_patience = 0
            return

        # NEW: if no users exist yet, go straight to calibration (no waiting)
        if not getattr(self.user_manager, "users", None):
            self.visualizer.draw_no_user_text(display)
            if self._post_calibration_cooldown <= 0:
                self.calibration(frame_rgb)
            return

        # If we already have an active user, do NOT run face recognition every frame.
        if self.user is not None and self.current_mode == "DETECTING":
            return

        now = time.time()
        if (now - self._fr_last_ts) < self._fr_interval_sec:
            return
        self._fr_last_ts = now

        user = self.user_manager.find_best_match(frame_rgb)
        if user:
            log.info(f"User identified: {user.user_id}")
            self.user = user
            self.detector.set_active_user(user)
            self.current_mode = "DETECTING"
            self.recognition_patience = 0

            # NEW: audible confirmation
            self._buzz_user_identified()
            return

        self.recognition_patience += 1
        self.visualizer.draw_no_user_text(display)

        if self._post_calibration_cooldown > 0:
            return

        if self.recognition_patience >= self.RECOGNITION_THRESHOLD:
            self.calibration(frame_rgb)
            return

    def calibration(self, frame):
        log.info("Starting Calibration...")
        # Stop any ongoing buzzer output before calibration UI
        try:
            self.buzzer.off()
        except Exception:
            pass

        # NEW: audible "calibration running" indicator
        self._buzz_calibration_start()

        result = self.ear_calibrator.calibrate()

        # Ensure calibration-running beep stops
        try:
            self.buzzer.off()
        except Exception:
            pass

        # Only destroy GUI windows in non-headless
        if not self.headless:
            # main_calibrator/ui uses "Drowsiness System"
            try:
                cv2.destroyWindow("Drowsiness System")
            except Exception:
                pass

        # NEW: main_calibrator can return ("user_swap", user_profile)
        if isinstance(result, tuple) and len(result) == 2 and result[0] == "user_swap":
            user = result[1]
            log.info(f"Calibration detected existing user. Switching to user_id={getattr(user, 'user_id', '?')}")
            self.user = user
            self.detector.set_active_user(user)
            self.current_mode = "DETECTING"
            self.recognition_patience = 0
            self._post_calibration_cooldown = 45

            # NEW: calibration ended + user identified
            self._buzz_calibration_success()
            self._buzz_user_identified()
            return

        result_threshold = result

        if result_threshold is not None and isinstance(result_threshold, float):
            log.info(f"Calibration Success. Threshold: {result_threshold:.3f}")

            # NEW: calibration success beep
            self._buzz_calibration_success()

            # IMPORTANT: don't use len(self.user_manager.users)+1 (it may be empty/not loaded)
            try:
                new_id = self.user_manager.repo.get_next_user_id()
            except Exception:
                # fallback (still better than always 1)
                new_id = int(time.time())

            # Keep frame consistent (already RGB here)
            fresh_frame = frame

            new_user = self.user_manager.register_new_user(fresh_frame, result_threshold, new_id)

            if new_user:
                log.info(f"New User Registered: ID {new_user.user_id}")
                self.user = new_user
                self.detector.set_active_user(new_user)
                self.current_mode = "DETECTING"
                log.info("Switched to DETECTING mode after registration.")

                # NEW: audible "user identified/ready"
                self._buzz_user_identified()
            else:
                log.error("Failed to register new user.")
                self.current_mode = "WAITING_FOR_USER"
        else:
            log.warning("Calibration failed.")
            self.current_mode = "WAITING_FOR_USER"

            # NEW: calibration failed/cancelled beep
            self._buzz_calibration_fail()

        self.recognition_patience = 0
        self._post_calibration_cooldown = 45

    def _drop_active_user(self, reason: str) -> None:
        """Drop current user and return to WAITING_FOR_USER safely."""
        if self.user is None and self.current_mode == "WAITING_FOR_USER":
            return

        log.info("Dropping active user (reason=%s)", reason)
        self.user = None
        self.current_mode = "WAITING_FOR_USER"
        self.recognition_patience = 0

        # Reset per-user state so the next person doesn't inherit baselines
        try:
            self.detector.set_active_user(None)
        except Exception:
            pass
        try:
            self.expression_classifier.reset()
        except Exception:
            pass

        # reset identity tracking
        self._id_mismatch_count = 0
        self._no_face_frames = 0
        self._post_calibration_cooldown = 0
        self._identity_prompted = False
import time
import threading
from queue import Queue, Empty
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from src.calibration.ratios import EAR
from src.calibration.ui import feedback as draw_feedback
from src.face_recognition.user_manager import UserManager
from src.utils.landmarks.constants import LEFT_EYE, RIGHT_EYE


class EARCalibrator:
    """
    EAR calibration (main flow + internal helpers).
    """
    CALIBRATION_DURATION_S = 10
    FACE_LOST_TIMEOUT_S = 3.5
    MIN_VALID_SAMPLES = 20

    USER_CHECK_INTERVAL = 15
    DISPLAY_UPDATE_INTERVAL = 3
    EAR_BOUNDS = (0.06, 0.60)
    STABILITY_WINDOW = 20
    PREALLOCATE_SIZE = 300

    def __init__(
        self, camera, face_mesh, user_manager: UserManager, system_logger=None, headless: bool = False
    ):
        self.camera = camera
        self.face_mesh = face_mesh
        self.user_manager = user_manager
        self.logger = system_logger  # optional
        self.headless = bool(headless)

        self.ear_calculator = EAR()

        self._ear_buffer = np.zeros(self.PREALLOCATE_SIZE, dtype=np.float32)
        self._ear_count = 0

        self._user_check_queue: Queue = Queue(maxsize=1)
        self._user_check_result = None
        self._user_check_thread: Optional[threading.Thread] = None
        self._user_check_stop = threading.Event()

    def _maybe_signal(self, name: str) -> None:
        """
        Optional hook for system loggers that support event signaling.
        Avoids hard dependency on SystemLogger.signal().
        """
        try:
            fn = getattr(self.logger, "signal", None)
            if callable(fn):
                fn(name)
        except Exception:
            # never crash calibration because of logging
            pass

    def calibrate(self) -> Union[None, float, Tuple[str, object]]:
        print(f"\n--- EAR Calibration: Look at the camera for {self.CALIBRATION_DURATION_S} seconds. ---")

        self._maybe_signal("calibration_prompt")

        time.sleep(1.0)

        self._ear_count = 0
        self._user_check_result = None
        self._user_check_stop.clear()

        start_time = time.time()
        face_lost_start_time = None
        frame_count = 0

        calibration_duration = self.CALIBRATION_DURATION_S
        face_lost_timeout = self.FACE_LOST_TIMEOUT_S
        user_check_interval = self.USER_CHECK_INTERVAL
        display_update_interval = self.DISPLAY_UPDATE_INTERVAL

        self._start_user_check_thread()

        try:
            while True:
                elapsed_time = time.time() - start_time
                if elapsed_time >= calibration_duration:
                    break

                # Prefer RGB if camera wrapper supports it
                try:
                    frame = self.camera.read(color="rgb")
                except TypeError:
                    frame = self.camera.read()

                if frame is None:
                    continue

                frame_count += 1
                feedback_frame = cv2.flip(frame, 1)

                # background identity check
                if frame_count % user_check_interval == 0 and self._user_check_queue.empty():
                    try:
                        self._user_check_queue.put_nowait(feedback_frame.copy())
                    except Exception:
                        pass

                if self._user_check_result is not None:
                    print(
                        f"Recognized user '{getattr(self._user_check_result, 'full_name', self._user_check_result.user_id)}' detected."
                    )
                    self._maybe_signal("calibration_success")
                    return ("user_swap", self._user_check_result)

                results = self.face_mesh.process(feedback_frame)
                ear, status_msg = self._process_landmarks_optimized(results, feedback_frame.shape)

                if status_msg == "Face not detected":
                    if face_lost_start_time is None:
                        face_lost_start_time = time.time()
                    elif time.time() - face_lost_start_time > face_lost_timeout:
                        print("Calibration failed: Face was not detected for too long.")
                        self._maybe_signal("calibration_fail")
                        return None
                else:
                    face_lost_start_time = None

                # UI only if not headless
                if (not self.headless) and (frame_count % display_update_interval == 0):
                    display_frame_bgr = cv2.cvtColor(feedback_frame, cv2.COLOR_RGB2BGR)
                    self.feedback(display_frame_bgr, ear, elapsed_time, status_msg, self._ear_count)

                if not self.headless:
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        print("Calibration cancelled by user.")
                        self._maybe_signal("calibration_fail")
                        return None

        finally:
            self._stop_user_check_thread()

        result = self.average_ear()
        if isinstance(result, float):
            self._maybe_signal("calibration_success")
        else:
            self._maybe_signal("calibration_fail")
        return result

    def feedback(self, frame, ear, elapsed, status_msg, num_samples):
        if self.headless:
            return
        draw_feedback(self, frame, ear, elapsed, status_msg, num_samples)

    def _start_user_check_thread(self) -> None:
        """Start background thread for user recognition."""
        if self._user_check_thread and self._user_check_thread.is_alive():
            return

        def check_users():
            while not self._user_check_stop.is_set():
                try:
                    frame = self._user_check_queue.get(timeout=0.2)
                except Empty:
                    continue

                try:
                    self._user_check_result = self.user_manager.find_best_match(frame)
                except Exception:
                    # keep thread alive even if recognition errors
                    self._user_check_result = None

        self._user_check_thread = threading.Thread(target=check_users, daemon=True)
        self._user_check_thread.start()

    def _stop_user_check_thread(self) -> None:
        """Gracefully stop background thread."""
        self._user_check_stop.set()
        if self._user_check_thread and self._user_check_thread.is_alive():
            self._user_check_thread.join(timeout=0.5)

    def _process_landmarks_optimized(self, results, frame_shape: Tuple[int, int, int]) -> Tuple[Optional[float], str]:
        """
        Returns:
            (ear_value, status_message)
        """
        if not getattr(results, "multi_face_landmarks", None):
            return None, "Face not detected"

        if len(results.multi_face_landmarks) > 1:
            return None, "ERROR: Too many faces!"

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame_shape[:2]

        try:
            right_ear = self.ear_calculator.calculate(
                [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE.ear]
            )
            left_ear = self.ear_calculator.calculate(
                [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE.ear]
            )
            ear = (right_ear + left_ear) / 2.0

            if not (self.EAR_BOUNDS[0] < ear < self.EAR_BOUNDS[1]):
                return ear, "Adjust position or lighting"

            status = "Keep steady..."
            if self._ear_count >= 5:
                recent_count = min(self.STABILITY_WINDOW, self._ear_count)
                recent = self._ear_buffer[self._ear_count - recent_count : self._ear_count]
                recent_median = float(np.median(recent))
                if abs(ear - recent_median) > 0.12:
                    status = "Hold still (stabilizing)"

            if self._ear_count < self.PREALLOCATE_SIZE:
                self._ear_buffer[self._ear_count] = ear
                self._ear_count += 1
            else:
                print("Warning: EAR buffer full")

            return ear, f"{status} ({self._ear_count} samples)"

        except (IndexError, ValueError):
            return None, "Could not find eye landmarks"

    def average_ear(self) -> Optional[float]:
        """
        Compute a robust open-eye baseline and derive the threshold.
        Uses MAD-based outlier filtering; falls back safely if filtering removes too much.
        """
        if self._ear_count < self.MIN_VALID_SAMPLES:
            print(
                f"Calibration failed: Not enough stable eye readings ({self._ear_count}/{self.MIN_VALID_SAMPLES})."
            )
            return None

        arr = self._ear_buffer[: self._ear_count]

        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))

        if mad < 1e-6:
            filtered = arr
        else:
            sigma = 1.4826 * mad
            mask = np.abs(arr - median) <= 3.0 * sigma
            filtered = arr[mask]

        if filtered.size < self.MIN_VALID_SAMPLES:
            lo, hi = np.percentile(arr, [10, 90])
            filtered = arr[(arr >= lo) & (arr <= hi)]

        if filtered.size < self.MIN_VALID_SAMPLES:
            filtered = arr

        open_eye_baseline = float(np.mean(filtered))
        ear_threshold = open_eye_baseline * 0.75

        print("Calibration complete:")
        print(f"  - Total samples: {self._ear_count}")
        print(f"  - Used samples: {int(filtered.size)}")
        print(f"  - Open-eye baseline: {open_eye_baseline:.3f}")
        print(f"  - EAR threshold (75%): {ear_threshold:.3f}")
        print(f"  - EAR range (used): {float(np.min(filtered)):.3f} - {float(np.max(filtered)):.3f}")

        return ear_threshold
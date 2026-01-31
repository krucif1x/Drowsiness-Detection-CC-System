import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.python.solutions import hands


class HandsModel:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.model = hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        MediaPipe expects RGB.
        Input from OpenCV is typically BGR.
        """
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def infer(self, image: np.ndarray, preprocessed: bool = False, return_raw: bool = False):
        """
        If preprocessed=False: expects BGR and converts to RGB.
        If preprocessed=True: expects RGB.

        return_raw:
            - False (default): returns hands_data only (backward compatible).
            - True: returns (hands_data, mediapipe_result).
        """
        if not preprocessed:
            image = self.preprocess(image)

        result = self.model.process(image)

        hands_data = []
        if result.multi_hand_landmarks:
            for hand_landmark in result.multi_hand_landmarks:
                hands_data.append([(lm.x, lm.y, lm.z) for lm in hand_landmark.landmark])

        if return_raw:
            return hands_data, result
        return hands_data

    def get_landmark(self, single_hand_data, landmark_index: int):
        """
        Pass a SINGLE hand's data list and the landmark index (0-20).
        Example: landmark_index=8 for INDEX_FINGER_TIP.
        """
        if single_hand_data and 0 <= landmark_index < len(single_hand_data):
            return single_hand_data[landmark_index]
        return None

    def close(self):
        self.model.close()


def _put_hud(image_bgr: np.ndarray, lines: list[str]) -> None:
    y = 22
    for line in lines:
        cv2.putText(image_bgr, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image_bgr, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22


if __name__ == "__main__":
    from src.core.frame_processing import normalize_hands
    from src.infrastructure.hardware.camera import Camera

    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    cam = Camera(source="auto", resolution=(640, 480))
    if not getattr(cam, "ready", True):
        raise SystemExit("Camera failed to initialize.")

    model = HandsModel(max_num_hands=2)

    window = "HandsModel - live test (press q/esc to quit)"
    fps_ema = 0.0
    alpha = 0.1

    try:
        while True:
            frame_rgb = cam.read(color="rgb")
            if frame_rgb is None:
                continue

            t0 = time.perf_counter()

            # Get both parsed data AND raw result for better drawing
            raw_hands, result = model.infer(frame_rgb, preprocessed=True, return_raw=True)

            h, w = frame_rgb.shape[:2]
            hands_norm = normalize_hands(raw_hands, w, h)  # still used as requested

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Draw skeleton using MediaPipe styles (much clearer than dots)
            if result.multi_hand_landmarks:
                for hand_lms in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame_bgr,
                        landmark_list=hand_lms,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_styles.get_default_hand_connections_style(),
                    )

            dt = max(1e-6, time.perf_counter() - t0)
            fps = 1.0 / dt
            fps_ema = fps if fps_ema <= 0 else (1 - alpha) * fps_ema + alpha * fps

            _put_hud(
                frame_bgr,
                [
                    f"hands: {len(hands_norm)}",
                    f"fps: {fps_ema:0.1f}",
                ],
            )

            cv2.imshow(window, frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        try:
            model.close()
        except Exception:
            pass
        try:
            cam.release()
        except Exception:
            pass
        cv2.destroyAllWindows()


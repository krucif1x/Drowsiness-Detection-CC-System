import time

import cv2
import mediapipe as mp
import numpy as np


class FaceMeshModel:
    """
    Wrapper for the MediaPipe Face Mesh solution.
    Handles initialization and inference.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self._mp_face_mesh = mp.solutions.face_mesh
        self._model = self._mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, image_rgb: np.ndarray):
        # MediaPipe FaceMesh expects RGB input
        return self._model.process(image_rgb)

    def close(self) -> None:
        self._model.close()


def _put_hud(image_bgr: np.ndarray, lines: list[str]) -> None:
    y = 22
    for line in lines:
        cv2.putText(
            image_bgr,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image_bgr,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 22


if __name__ == "__main__":
    from src.infrastructure.hardware.camera import Camera

    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    cam = Camera(source="auto", resolution=(640, 480))
    if not getattr(cam, "ready", True):
        raise SystemExit("Camera failed to initialize.")

    model = FaceMeshModel(max_num_faces=1, refine_landmarks=True)

    # Toggle layers
    show_tess = True
    show_contours = True
    show_irises = True

    window = "FaceMeshModel - live test (q/esc quit, t/c/i toggle)"
    last_t = time.perf_counter()
    fps_ema = 0.0
    alpha = 0.1

    try:
        while True:
            frame_rgb = cam.read(color="rgb")
            if frame_rgb is None:
                continue

            t0 = time.perf_counter()
            results = model.process(frame_rgb)

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            faces = getattr(results, "multi_face_landmarks", None) or []
            if faces:
                face_lms = faces[0]

                if show_tess:
                    mp_drawing.draw_landmarks(
                        image=frame_bgr,
                        landmark_list=face_lms,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
                    )

                if show_contours:
                    mp_drawing.draw_landmarks(
                        image=frame_bgr,
                        landmark_list=face_lms,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style(),
                    )

                if show_irises:
                    mp_drawing.draw_landmarks(
                        image=frame_bgr,
                        landmark_list=face_lms,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_styles.get_default_face_mesh_iris_connections_style(),
                    )

            # FPS (EMA)
            dt = max(1e-6, time.perf_counter() - t0)
            fps = 1.0 / dt
            fps_ema = fps if fps_ema <= 0 else (1 - alpha) * fps_ema + alpha * fps

            _put_hud(
                frame_bgr,
                [
                    f"faces: {len(faces)}",
                    f"fps: {fps_ema:0.1f}",
                    f"tess={show_tess} contours={show_contours} irises={show_irises}",
                ],
            )

            cv2.imshow(window, frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("t"):
                show_tess = not show_tess
            if key == ord("c"):
                show_contours = not show_contours
            if key == ord("i"):
                show_irises = not show_irises
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
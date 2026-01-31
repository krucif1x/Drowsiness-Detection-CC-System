import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.utils.landmarks.constants import HandIdx  # <-- add

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
    ear_raw: float
    avg_ear: float
    mar: float
    face_confidence: float


class HandsPipeline:
    """
    Runs hand inference on an interval and caches *normalized* hands (0..1).
    Keeps DetectionLoop free of hand-related CPU logic.
    """

    def __init__(self, hand_wrapper, inference_interval_frames: int = 5):
        self.hand_wrapper = hand_wrapper
        self.interval = max(1, int(inference_interval_frames))
        self._frame_idx = 0
        self._cached_hands_norm = []

    def step(self, frame, w: int, h: int):
        self._frame_idx += 1
        if self._frame_idx % self.interval == 0:
            raw_hands = self.hand_wrapper.infer(frame, preprocessed=True)
            self._cached_hands_norm = normalize_hands(raw_hands, w, h)
        return self._cached_hands_norm


def normalize_hands(hands_data, w: int, h: int):
    """
    Return hands normalized to 0..1.
    Input can be pixel coords or normalized coords.
    Output: List[hand], each hand is List[(x_norm, y_norm, z)].
    """
    if not hands_data or w <= 0 or h <= 0:
        return []

    inv_w = 1.0 / float(w)
    inv_h = 1.0 / float(h)
    norm_hands = []

    for hand in hands_data:
        if not hand:
            continue

        first_pt = hand[HandIdx.WRIST] 
        # Heuristic: if x/y > 1.0 assume pixels
        is_pixel_coords = (first_pt[0] > 1.0) or (first_pt[1] > 1.0)

        current_hand = []
        if is_pixel_coords:
            for pt in hand:
                z = pt[2] if len(pt) > 2 else 0.0
                current_hand.append((pt[0] * inv_w, pt[1] * inv_h, z))
        else:
            for pt in hand:
                z = pt[2] if len(pt) > 2 else 0.0
                current_hand.append((pt[0], pt[1], z))

        norm_hands.append(current_hand)

    return norm_hands


class FrameProcessor:
    """
    Pure frame math:
    - landmarks -> px
    - EAR/MAR
    - head pose
    - face confidence + face center (norm)
    No UI, no logging side effects besides returning values.
    """

    def __init__(
        self,
        *,
        head_pose_estimator,
        ear_calculator,
        mar_calculator,
        ear_smoother,
        indices_left_ear,
        indices_right_ear,
        indices_mouth,
    ):
        self.head_pose_estimator = head_pose_estimator
        self.ear_calculator = ear_calculator
        self.mar_calculator = mar_calculator
        self.ear_smoother = ear_smoother

        self.L_EAR = indices_left_ear
        self.R_EAR = indices_right_ear
        self.M_MAR = indices_mouth

    def extract(self, frame, results) -> Optional[FrameFeatures]:
        if not results or not getattr(results, "multi_face_landmarks", None):
            return None

        h, w = frame.shape[:2]
        raw_lms = results.multi_face_landmarks[0]

        # Face detection confidence (best-effort)
        face_confidence = 1.0
        try:
            if hasattr(raw_lms.landmark[0], "visibility"):
                face_confidence = float(raw_lms.landmark[0].visibility)
        except Exception:
            face_confidence = 1.0

        # Head pose (degrees)
        pose = self.head_pose_estimator.calculate_pose(raw_lms, w, h)
        pitch, yaw, roll = pose if pose else (0.0, 0.0, 0.0)

        # Landmarks px (compute once)
        lms_px = [(int(l.x * w), int(l.y * h)) for l in raw_lms.landmark]

        left_eye = [lms_px[i] for i in self.L_EAR]
        right_eye = [lms_px[i] for i in self.R_EAR]
        mouth = [lms_px[i] for i in self.M_MAR]

        left = self.ear_calculator.calculate(left_eye)
        right = self.ear_calculator.calculate(right_eye)
        ear_raw = (left + right) / 2.0
        avg_ear = self.ear_smoother.update(ear_raw)

        mar = self.mar_calculator.calculate(mouth)

        # Face center (normalized)
        nose_tip = raw_lms.landmark[1]
        face_center_norm = (float(nose_tip.x), float(nose_tip.y))

        return FrameFeatures(
            h=h,
            w=w,
            pitch=float(pitch),
            yaw=float(yaw),
            roll=float(roll),
            lms_px=lms_px,
            face_center_norm=face_center_norm,
            ear_raw=float(ear_raw),
            avg_ear=float(avg_ear),
            mar=float(mar),
            face_confidence=float(face_confidence),
        )
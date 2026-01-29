from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

log = logging.getLogger(__name__)


@dataclass
class ExtractMetadata:
    detected: bool
    prob: float | None
    faces_detected: int
    reason: str | None = None


class FaceRecognizer:
    """
    Face encoding extractor using:
      - MTCNN for detection + aligned crop
      - InceptionResnetV1 (vggface2) for 512-d embeddings

    Returns a L2-normalized 512-d embedding (np.float32).
    """

    def __init__(
        self,
        device: torch.device,
        input_color: str = "RGB",
        min_detection_prob: float = 0.95,
        image_size: int = 160,
        margin: int = 14,
        keep_all: bool = False,  # was True (faster for single face)
    ):
        self.device = device
        self.input_color = (input_color or "RGB").upper()
        self.min_detection_prob = float(min_detection_prob)

        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=margin,
            keep_all=keep_all,
            post_process=True,
            device=self.device,
            select_largest=True,
        )
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    def _to_pil_rgb(self, frame: Any) -> Image.Image:
        """
        Accepts numpy frame (H,W,3) or PIL image and returns PIL RGB.
        """
        if isinstance(frame, Image.Image):
            return frame.convert("RGB")

        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError("Expected HxWx3 image array")

        arr = arr[:, :, :3]

        # OpenCV frames are commonly BGR; convert if needed
        if self.input_color == "BGR":
            arr = arr[:, :, ::-1]

        return Image.fromarray(arr.astype(np.uint8), mode="RGB")

    @torch.inference_mode()
    def extract(self, image_frame: Any, return_metadata: bool = False) -> Optional[np.ndarray] | Tuple[Optional[np.ndarray], Dict]:
        try:
            img = self._to_pil_rgb(image_frame)
        except Exception as e:
            md = ExtractMetadata(False, None, 0, reason=f"bad_input:{e}")
            return (None, md.__dict__) if return_metadata else None

        # SINGLE pass: get aligned face crops + detection probabilities
        faces, probs = self.mtcnn(img, return_prob=True)

        if faces is None or probs is None:
            md = ExtractMetadata(False, None, 0, reason="no_face")
            return (None, md.__dict__) if return_metadata else None

        # Normalize shapes for keep_all=True/False
        if isinstance(probs, float):
            probs_list = [float(probs)]
            faces_list = [faces]  # Tensor (3,H,W)
        else:
            probs_list = [float(p) for p in probs]
            faces_list = list(faces)  # list of (3,H,W)

        faces_detected = len(probs_list)
        good_idxs = [i for i, p in enumerate(probs_list) if p >= self.min_detection_prob]

        if len(good_idxs) == 0:
            md = ExtractMetadata(False, float(max(probs_list)), faces_detected, reason="low_conf")
            return (None, md.__dict__) if return_metadata else None

        # Reject multiple confident faces
        if len(good_idxs) > 1:
            md = ExtractMetadata(False, float(max(probs_list)), faces_detected, reason="multi_face")
            return (None, md.__dict__) if return_metadata else None

        best_i = good_idxs[0]
        face_tensor = faces_list[best_i]

        emb = self.resnet(face_tensor.unsqueeze(0).to(self.device)).squeeze(0)
        emb = emb / (emb.norm(p=2) + 1e-8)

        out = emb.detach().cpu().numpy().astype(np.float32)
        md = ExtractMetadata(True, float(probs_list[best_i]), faces_detected, reason=None)
        return (out, md.__dict__) if return_metadata else out
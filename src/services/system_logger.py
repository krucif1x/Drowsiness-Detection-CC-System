import cv2
import base64
import logging
import numpy as np
from datetime import datetime
from typing import Optional

from src.services.remote_logger import RemoteLogWorker
from src.infrastructure.data.repository import UnifiedRepository
from src.infrastructure.data.models import DrowsinessEvent

UNKNOWN_USER_ID = 0


class SystemLogger:
    """
    Handles Local DB and Remote Push (no hardware actuation).
    """
    def __init__(
        self,
        remote_worker: Optional[RemoteLogWorker] = None,
        event_repo: Optional[UnifiedRepository] = None,
        vehicle_vin: str = "VIN-0001",
        local_quality: int = 85,
        remote_quality: int = 70,
    ):
        self.remote = remote_worker
        self.repo = event_repo
        self.vehicle_vin = vehicle_vin
        self.local_quality = local_quality
        self.remote_quality = remote_quality

    def log_event(
        self,
        user_id: int,
        event_type: str,
        duration: float = 0.0,
        value: float = 0.0,
        frame: Optional[np.ndarray] = None,
        alert_category: str = None,
        alert_detail: str = None,
        severity: str = None,
    ):
        timestamp = datetime.now()
        remote_allowed = self.remote and self.remote.enabled and user_id != UNKNOWN_USER_ID

        norm_status = (event_type or "event").strip().lower()

        jpeg_local = None
        jpeg_remote = None

        if frame is not None:
            jpeg_local = self._encode_jpeg(frame, self.local_quality)

            if remote_allowed:
                h, w = frame.shape[:2]
                target_w = 640
                if w > target_w:
                    scale = target_w / float(w)
                    new_h = max(1, int(round(h * scale)))
                    resized = cv2.resize(frame, (target_w, new_h), interpolation=cv2.INTER_AREA)
                else:
                    resized = frame
                jpeg_remote = self._encode_jpeg(resized, self.remote_quality)

        local_rowid = None

        # LOCAL DATABASE LOGGING
        if self.repo:
            try:
                event = DrowsinessEvent(
                    vehicle_identification_number=self.vehicle_vin,
                    user_id=user_id,
                    status=norm_status,
                    time=timestamp,
                    img_drowsiness=jpeg_local,
                    img_path=None,
                    duration=duration,
                    value=value,
                    alert_category=alert_category,
                    alert_detail=alert_detail,
                    severity=severity,
                )
                local_rowid = self.repo.add_event(event)
            except Exception as e:
                logging.error("Failed to save event to database: %s", e, exc_info=True)

        # REMOTE LOGGING
        if remote_allowed and self.remote:
            try:
                self.remote.send_or_queue(
                    vehicle_vin=self.vehicle_vin,
                    user_id=user_id,
                    status=norm_status,
                    time_dt=timestamp,
                    raw_jpeg=jpeg_remote,
                    alert_category=alert_category,
                    alert_detail=alert_detail,
                    severity=severity,
                    local_event_id=local_rowid,
                )
            except Exception as e:
                logging.error("Failed to send event to remote: %s", e, exc_info=True)

    def _encode_jpeg(self, frame, quality):
        try:
            if frame is None:
                return None

            arr = np.asarray(frame)

            # Ensure uint8 (OpenCV imencode expects uint8 for typical images)
            if arr.dtype != np.uint8:
                # Common cases: float in [0,1] or [0,255]
                arr_f = arr.astype(np.float32)
                if arr_f.max() <= 1.0:
                    arr_f = arr_f * 255.0
                arr = np.clip(arr_f, 0, 255).astype(np.uint8)

            # Ensure 3-channel BGR for encoding
            if arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            elif arr.ndim == 3 and arr.shape[2] == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            elif arr.ndim == 3 and arr.shape[2] == 3:
                # Your pipeline assumes RGB; convert to BGR.
                # If your frames are already BGR, this will swap channels (still encodes fine).
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            else:
                return None

            ok, buf = cv2.imencode(
                ".jpg",
                arr,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
            )
            return bytes(buf) if ok else None
        except Exception:
            return None
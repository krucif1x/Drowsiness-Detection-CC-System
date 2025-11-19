import cv2
import base64
import logging
import numpy as np
from datetime import datetime
from typing import Optional

from src.services.remote_logger import RemoteLogWorker
from src.infrastructure.data.repository import UnifiedRepository
from src.infrastructure.data.models import DrowsinessEvent

class SystemLogger:
    """
    The Coordinator. Handles Buzzer, Local DB, and Remote Push.
    """
    def __init__(
        self, 
        buzzer=None, 
        remote_worker: Optional[RemoteLogWorker] = None, 
        event_repo: Optional[UnifiedRepository] = None,
        vehicle_vin: str = "VIN-0001",
        local_quality: int = 85,
        remote_quality: int = 70
    ):
        self.buzzer = buzzer
        self.remote = remote_worker
        self.repo = event_repo
        self.vehicle_vin = vehicle_vin
        self.local_quality = local_quality
        self.remote_quality = remote_quality

    def log_event(self, user_id: int, event_type: str, duration: float = 0.0, 
                 value: float = 0.0, frame: Optional[np.ndarray] = None):
        
        timestamp = datetime.now()
        time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

        # 1. Image Encoding
        jpeg_local = None
        jpeg_remote_b64 = None
        
        if frame is not None:
            jpeg_local = self._encode_jpeg(frame, self.local_quality)
            if self.remote and self.remote.enabled:
                jpeg_remote = self._encode_jpeg(frame, self.remote_quality)
                if jpeg_remote:
                    jpeg_remote_b64 = base64.b64encode(jpeg_remote).decode("ascii")

        # 2. Local Save
        if self.repo:
            event = DrowsinessEvent(
                vehicle_identification_number=self.vehicle_vin,
                user_id=user_id,
                status=event_type.lower(),
                time=timestamp,
                img_drowsiness=jpeg_local,
                img_path=None,
            )
            # Optionally set duration and value if your model supports it
            setattr(event, "duration", duration)
            setattr(event, "value", value)
            self.repo.add_event(event)
            logging.info(f"[LOG] Saved Local: {event_type}")

        # 3. Remote Push
        if self.remote:
            self.remote.send_or_queue(
                vehicle_vin=self.vehicle_vin,
                user_id=user_id,
                status=event_type,
                time_dt=timestamp,
                img_base64=jpeg_remote_b64,
                img_path=None
            )

    def alert(self, level: str = "warning"):
        """Triggers buzzer. Fixed TypeError."""
        if not self.buzzer: return
        
        if level == "warning":     
            self.buzzer.beep(0.1, 0.1, background=True)
        elif level == "critical":  
            self.buzzer.beep(0.5, 0.5, background=True)
        elif level == "distraction": 
            self.buzzer.beep(0.1, 0.1, background=True)
            
    def stop_alert(self):
        if self.buzzer: self.buzzer.off()

    def _encode_jpeg(self, frame, quality):
        try:
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), 
                                 [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
            return bytes(buf) if ok else None
        except: return None
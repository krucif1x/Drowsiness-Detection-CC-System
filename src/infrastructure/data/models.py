import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any

class UserProfile:
    """Data class for a user with server reference only."""
    __slots__ = ('id', 'user_id', 'ear_threshold', 'face_encoding')
    
    def __init__(
        self, 
        profile_id: int,
        user_id: int,
        ear_threshold: float, 
        face_encoding: np.ndarray
    ):
        self.id = profile_id
        self.user_id = user_id
        self.ear_threshold = ear_threshold
        enc = np.array(face_encoding, dtype=np.float32).flatten()
        norm = np.linalg.norm(enc) + 1e-8
        self.face_encoding = enc / norm

    def __repr__(self) -> str:
        return f"UserProfile(id={self.id}, user_id={self.user_id}, ear_threshold={self.ear_threshold:.3f})"

@dataclass
class DrowsinessEvent:
    vehicle_identification_number: str
    user_id: int
    status: str = "drowsy"
    time: datetime = field(default_factory=datetime.now)
    img_drowsiness: Optional[str] = None
    img_path: Optional[str] = None
    duration: float = 0.0
    value: float = 0.0  # User's EAR threshold at time of event
    
    # Management-friendly fields (Power BI ready)
    alert_category: Optional[str] = None  # "Drowsiness" or "Distraction" or "Critical Alert"
    alert_detail: Optional[str] = None    # "Eyes Closed Too Long", "Looking Left", etc.
    severity: Optional[str] = None         # "Low", "Medium", "High", "Critical"

    def _fmt_time(self) -> str:
        return self.time.strftime("%Y-%m-%d %H:%M:%S")

    def to_transport_payload(self) -> Dict[str, Any]:
        if not self.vehicle_identification_number:
            raise ValueError("vehicle_identification_number is required")
        status_norm = (self.status or "").strip().lower()
        payload = {
            "vehicle_identification_number": self.vehicle_identification_number,
            "user_id": int(self.user_id),
            "time": self._fmt_time(),
            "status": status_norm,
            "img_drowsiness": self.img_drowsiness,
            "duration": self.duration,
            "value": self.value,
            # Management-friendly fields
            "alert_category": self.alert_category,
            "alert_detail": self.alert_detail,
            "severity": self.severity,
        }
        return payload
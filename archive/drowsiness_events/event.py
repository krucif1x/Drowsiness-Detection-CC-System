# file: src/api_client/event.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any

@dataclass
class DrowsinessEvent:
    vehicle_identification_number: str
    user_id: int
    status: str = "drowsy"
    time: datetime = field(default_factory=datetime.now)
    img_drowsiness: Optional[str] = None
    img_path: Optional[str] = None  # Kept so creating the object doesn't crash

    def _fmt_time(self) -> str:
        return self.time.strftime("%Y-%m-%d %H:%M:%S")

    def to_transport_payload(self) -> Dict[str, Any]:
        if not self.vehicle_identification_number:
            raise ValueError("vehicle_identification_number is required")
        
        status_norm = (self.status or "").strip().lower()
        
        # REMOVED: img_path from payload
        payload = {
            "vehicle_identification_number": self.vehicle_identification_number,
            "user_id": int(self.user_id),
            "time": self._fmt_time(),
            "status": status_norm,
            "img_drowsiness": self.img_drowsiness,
        }
        return payload
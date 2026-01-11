# event.py
"""
Defines the data structures (data classes) for our application.
Role: This is your Data Contract.

Explanation: This file is a "blueprint." It defines the exact structure of the data your server is expecting. The @dataclass is a clean way to hold all the pieces of information.

Key Function: to_dict(). This is critical. You cannot send a Python DrowsinessEvent object over the internet. You must convert it to a universal format. This function turns your Python object into a simple dictionary, which can then be easily converted to JSON (the language of APIs).
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any

@dataclass
class DrowsinessEvent:
    """
    A data class to hold drowsiness event information with vehicle and image data.
    """
    vehicle_identification_number: str  # VIN of the vehicle
    user_id: int                    # INTEGER ID referencing users table
    status: str = "drowsy"                       # Event status (e.g., 'DETECTED', 'RESOLVED')
    time: datetime = field(default_factory=datetime.now)  # Timestamp of event
    img_drowsiness: Optional[str] = None  # Screenshot data (base64 encoded)
    img_path: Optional[str] = None  # Path to the image file

    # NEW: Add management fields
    alert_category: Optional[str] = None
    alert_detail: Optional[str] = None
    severity: Optional[str] = None

    def _fmt_time(self) -> str:
        # Server expects "YYYY-MM-DD HH:MM:SS"
        return self.time.strftime("%Y-%m-%d %H:%M:%S")

    def to_transport_payload(self) -> Dict[str, Any]:
        """
        Build exactly the JSON your server expects:
        {
            "vehicle_identification_number": "...",
            "user_id": 1,
            "time": "2025-11-12 03:00:00",
            "status": "drowsy",
            "img_drowsiness": "...",   # optional base64 string
            "img_path": "/path/to/image.jpg"  # optional
        }
        """
        if not self.vehicle_identification_number:
            raise ValueError("vehicle_identification_number is required")
        if self.user_id is None or self.user_id < 0:
            raise ValueError("user_id must be a non-negative integer")
        status_norm = (self.status or "").strip().lower()
        if not status_norm:
            raise ValueError("status is required")

        payload = {
            "vehicle_identification_number": self.vehicle_identification_number,
            "user_id": int(self.user_id),
            "time": self._fmt_time(),
            "status": status_norm,
            "img_drowsiness": self.img_drowsiness,
            "img_path": self.img_path,
        }
        return payload
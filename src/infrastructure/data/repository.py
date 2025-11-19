import numpy as np
import logging
from datetime import datetime
from typing import List, Optional, Iterable, Any
from src.infrastructure.data.models import UserProfile, DrowsinessEvent

class UnifiedRepository:
    MAX_USERS_IN_MEMORY = 1000

    def __init__(self, db):
        self.db = db  # UnifiedDatabase instance

    # --- USER PROFILE METHODS ---
    def load_all_users(self) -> List[UserProfile]:
        rows = self.db.execute("""
            SELECT id, user_id, ear_threshold, face_encoding
            FROM user_profiles
            ORDER BY last_seen DESC
            LIMIT ?
        """, (self.MAX_USERS_IN_MEMORY,), fetch=True) or []
        
        users = []
        for pid, uid, ear, enc_blob in rows:
            try:
                enc = np.frombuffer(enc_blob, dtype=np.float32)
                users.append(UserProfile(pid, uid, ear, enc))
            except Exception as e:
                logging.error(f"Failed to load user {uid}: {e}")
        logging.info(f"Loaded {len(users)} user(s)")
        return users

    def save_user(self, user: UserProfile) -> int:
        encoding_bytes = user.face_encoding.tobytes() if isinstance(user.face_encoding, np.ndarray) else user.face_encoding
        rowid = self.db.execute("""
            INSERT INTO user_profiles (user_id, ear_threshold, face_encoding, last_seen)
            VALUES (?, ?, ?, ?)
        """, (user.user_id, user.ear_threshold, encoding_bytes, datetime.now()))
        return int(rowid)

    def update_last_seen(self, user_id: int):
        self.db.execute("""
            UPDATE user_profiles SET last_seen = ? WHERE user_id = ?
        """, (datetime.now(), user_id))

    def get_next_user_id(self) -> int:
        try:
            result = self.db.execute(
                "SELECT MAX(user_id) FROM user_profiles",
                fetch=True
            )
            max_id = result[0][0] if result and result[0][0] else 0
            return max_id + 1
        except Exception as e:
            logging.error(f"Failed to get next user_id: {e}")
            return 1

    # --- DROWSINESS EVENT METHODS ---
    def add_event(self, event: DrowsinessEvent) -> int:
        try:
            rowid = self.db.execute(
                """
                INSERT INTO drowsiness_events
                (vehicle_identification_number, user_id, time, status, img_drowsiness, duration, value)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.vehicle_identification_number,
                    event.user_id,
                    event.time.strftime("%Y-%m-%d %H:%M:%S") if hasattr(event.time, "strftime") else str(event.time),
                    event.status,
                    event.img_drowsiness,
                    getattr(event, "duration", 0.0),
                    getattr(event, "value", 0.0),
                ),
            )
            logging.info(f"Event saved: id={rowid} type={event.status} dur={getattr(event, 'duration', 0.0):.1f}s val={getattr(event, 'value', 0.0):.2f}")
            return int(rowid)
        except Exception as e:
            logging.error(f"Database Insert Failed: {e}")
            return -1

    def get_all_events(self) -> list[tuple]:
        return self.db.execute(
            """
            SELECT id, vehicle_identification_number, user_id, time, status, duration, value
            FROM drowsiness_events
            ORDER BY time DESC
            """,
            fetch=True
        )

    def get_events_by_user(self, user_id: int) -> list[tuple]:
        return self.db.execute(
            """
            SELECT id, vehicle_identification_number, user_id, time, status, duration, value
            FROM drowsiness_events
            WHERE user_id = ?
            ORDER BY time DESC
            """,
            (user_id,),
            fetch=True
        )

    def get_events_by_vehicle(self, vin: str) -> list[tuple]:
        return self.db.execute(
            """
            SELECT id, vehicle_identification_number, user_id, time, status, duration, value
            FROM drowsiness_events
            WHERE vehicle_identification_number = ?
            ORDER BY time DESC
            """,
            (vin,),
            fetch=True
        )
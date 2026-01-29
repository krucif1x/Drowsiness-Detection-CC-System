import numpy as np
import logging
from datetime import datetime
from typing import List
from src.infrastructure.data.models import UserProfile, DrowsinessEvent


class UnifiedRepository:
    MAX_USERS_IN_MEMORY = 1000

    def __init__(self, db):
        self.db = db  # UnifiedDatabase instance

    # --- USER PROFILE METHODS ---
    def load_all_users(self) -> List[UserProfile]:
        rows = (
            self.db.execute(
                """
                SELECT id, user_id, ear_threshold, face_encoding
                FROM user_profiles
                WHERE face_encoding IS NOT NULL
                ORDER BY last_seen DESC
                LIMIT ?
                """,
                (self.MAX_USERS_IN_MEMORY,),
                fetch=True,
            )
            or []
        )

        users: List[UserProfile] = []
        for pid, uid, ear, enc_blob in rows:
            try:
                enc = np.frombuffer(enc_blob, dtype=np.float32)
                users.append(UserProfile(pid, uid, float(ear), enc))
            except Exception:
                logging.exception("Failed to load user %s", uid)

        logging.info("Loaded %d user(s)", len(users))
        return users

    def save_user(self, user: UserProfile) -> int:
        enc = np.asarray(user.face_encoding, dtype=np.float32).flatten()
        encoding_bytes = enc.tobytes()

        rowid = self.db.execute(
            """
            INSERT INTO user_profiles (user_id, ear_threshold, face_encoding, last_seen)
            VALUES (?, ?, ?, ?)
            """,
            (int(user.user_id), float(user.ear_threshold), encoding_bytes, datetime.now()),
        )
        return int(rowid)

    def update_last_seen(self, user_id: int):
        self.db.execute(
            """
            UPDATE user_profiles SET last_seen = ? WHERE user_id = ?
            """,
            (datetime.now(), int(user_id)),
        )

    def get_next_user_id(self) -> int:
        try:
            result = self.db.execute("SELECT MAX(user_id) FROM user_profiles", fetch=True)
            max_id = result[0][0] if result and result[0][0] else 0
            return int(max_id) + 1
        except Exception as e:
            logging.error("Failed to get next user_id: %s", e)
            return 1

    # --- DROWSINESS EVENT METHODS ---
    def add_event(self, event: DrowsinessEvent) -> int:
        try:
            rowid = self.db.execute(
                """
                INSERT INTO drowsiness_events
                (vehicle_identification_number, user_id, time, status, img_drowsiness,
                 duration, value, alert_category, alert_detail, severity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.vehicle_identification_number,
                    int(event.user_id),
                    event.time.strftime("%Y-%m-%d %H:%M:%S") if hasattr(event.time, "strftime") else str(event.time),
                    event.status,
                    event.img_drowsiness,
                    float(event.duration),
                    float(event.value),
                    event.alert_category,
                    event.alert_detail,
                    event.severity,
                ),
            )
            logging.info(
                "Event saved: row_id=%s, %s - %s (%s)",
                rowid,
                event.alert_category,
                event.alert_detail,
                event.severity,
            )
            return int(rowid)
        except Exception as e:
            logging.error("Database Insert Failed: %s", e, exc_info=True)
            return -1
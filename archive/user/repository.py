import numpy as np
import logging
from datetime import datetime
from typing import List
from src.infrastructure.data.user.profile import UserProfile
from src.infrastructure.data.user.database import UserDatabase

class UserRepository:
    MAX_USERS_IN_MEMORY = 1000

    def __init__(self, db: UserDatabase):
        self.db = db

    def load_all(self) -> List[UserProfile]:
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
        """Save user and return the database row ID."""
        # Convert numpy encoding to bytes for BLOB storage
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
        """Generate next available user ID."""
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
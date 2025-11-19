import os
import logging
import sqlite3
import time
from datetime import datetime
from threading import Lock

class UserDatabase:
    DB_RETRY_ATTEMPTS = 3
    DB_RETRY_DELAY = 0.1

    def __init__(self, db_file: str):
        self.db_file = db_file  # ✅ Make sure this is stored
        self._lock = Lock()
        
        # ✅ Make sure directories exist
        os.makedirs(os.path.dirname(db_file), exist_ok=True)
        
        self._ensure_schema()
        logging.info(f"✓ UserDatabase initialized: {self.db_file}")

    def _ensure_schema(self):
        """Create user_profiles table if it doesn't exist."""
        with sqlite3.connect(self.db_file) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL UNIQUE,
                    ear_threshold REAL NOT NULL,
                    face_encoding BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON user_profiles(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_seen ON user_profiles(last_seen)")
            conn.commit()

    def execute(self, query: str, params: tuple = (), fetch: bool = False):
        """Execute query with retry logic."""
        with self._lock:
            for attempt in range(self.DB_RETRY_ATTEMPTS):
                try:
                    with sqlite3.connect(self.db_file, timeout=10.0) as conn:
                        cur = conn.execute(query, params)
                        if fetch:
                            return cur.fetchall()
                        conn.commit()
                        return cur.lastrowid
                except sqlite3.OperationalError as e:
                    if attempt < self.DB_RETRY_ATTEMPTS - 1:
                        logging.warning(f"DB locked, retry {attempt+1}")
                        time.sleep(self.DB_RETRY_DELAY * (attempt + 1))
                    else:
                        logging.error(f"DB operation failed: {e}")
                        raise
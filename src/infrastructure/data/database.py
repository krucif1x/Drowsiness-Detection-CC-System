import os
import sqlite3
import logging
import threading
import time
from datetime import datetime

class UnifiedDatabase:
    DB_RETRY_ATTEMPTS = 3
    DB_RETRY_DELAY = 0.1

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._ensure_schema()
        logging.info(f"✓ UnifiedDatabase initialized: {self.db_path}")

    def _ensure_schema(self):
        """Create all tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            
            # User Profiles
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL UNIQUE,
                    full_name TEXT,
                    ear_threshold REAL NOT NULL,
                    face_encoding BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON user_profiles(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_seen ON user_profiles(last_seen)")
            
            # Drowsiness Events - Simplified for management
            conn.execute("""
                CREATE TABLE IF NOT EXISTS drowsiness_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vehicle_identification_number TEXT,
                    user_id INTEGER,
                    time TEXT,
                    status TEXT,
                    img_drowsiness BLOB,
                    duration REAL DEFAULT 0.0,
                    value REAL DEFAULT 0.0,
                    
                    -- Management-friendly fields
                    alert_category TEXT,
                    alert_detail TEXT,
                    severity TEXT
                )
            """)
            
            # MIGRATION: Add new columns if table already exists
            cursor = conn.execute("PRAGMA table_info(drowsiness_events)")
            existing_columns = {row[1] for row in cursor.fetchall()}
            
            new_columns = {
                'alert_category': 'TEXT',
                'alert_detail': 'TEXT',
                'severity': 'TEXT'
            }
            
            for col_name, col_type in new_columns.items():
                if col_name not in existing_columns:
                    try:
                        conn.execute(f"ALTER TABLE drowsiness_events ADD COLUMN {col_name} {col_type}")
                        logging.info(f"✓ Added column: {col_name}")
                    except sqlite3.OperationalError as e:
                        logging.warning(f"Column {col_name} might already exist: {e}")
            
            # Create indexes for Power BI performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_time ON drowsiness_events(time)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_user ON drowsiness_events(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_category ON drowsiness_events(alert_category)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_severity ON drowsiness_events(severity)")
            
            conn.commit()

    def execute(self, query: str, params: tuple = (), fetch: bool = False):
        """Execute query with retry logic."""
        with self._lock:
            for attempt in range(self.DB_RETRY_ATTEMPTS):
                try:
                    with sqlite3.connect(self.db_path, timeout=10.0) as conn:
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

    def close(self):
        # No persistent connection, so nothing to close
        pass
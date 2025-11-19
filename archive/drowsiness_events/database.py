# file: src/infrastructure/data/drowsiness_events/database.py
import sqlite3
import os
import logging

class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None

        # 1. Ensure directory exists
        folder = os.path.dirname(db_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        
        try:
            # 2. Connect
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            
            # 3. CREATE ALL TABLES AT ONCE
            self._create_tables()
            
            logging.info(f"Unified Database initialized at {db_path}")
        except Exception as e:
            logging.error(f"Failed to connect to database: {e}")
            self.conn = None

    def _create_tables(self):
        if not self.conn: return
        cur = self.conn.cursor()

        # --- Table 1: User Profiles (Matches your image) ---
        # We rename this from 'users' to 'user_profiles' to match your existing logic
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT UNIQUE,
                ear_threshold REAL,
                face_encoding BLOB
            )
        """)

        # --- Table 2: Drowsiness Events (The missing table) ---
        # Removed img_path as requested
        cur.execute("""
            CREATE TABLE IF NOT EXISTS drowsiness_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vehicle_identification_number TEXT,
                user_id INTEGER,
                time TEXT,
                status TEXT,
                img_drowsiness BLOB,
                duration REAL DEFAULT 0.0,
                value REAL DEFAULT 0.0
            )
        """)
        
        # Indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_time ON drowsiness_events(time)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_user ON drowsiness_events(user_id)")
        
        self.conn.commit()

    def get_connection(self):
        return self.conn
    
    def close(self):
        if self.conn: self.conn.close()
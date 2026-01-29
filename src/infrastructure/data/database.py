import os
import sqlite3
import logging
import threading
import time
from typing import Any, Optional


class UnifiedDatabase:
    DB_RETRY_ATTEMPTS = 3
    DB_RETRY_DELAY = 0.1

    def __init__(self, db_path: str):
        self.db_path = os.path.normpath(db_path)
        self._lock = threading.Lock()
        self._local = threading.local()  # per-thread connection cache

        self._ensure_parent_dir(self.db_path)
        self._ensure_schema()

        logging.info("✓ UnifiedDatabase initialized: %s", self.db_path)

    def _ensure_parent_dir(self, path: str) -> None:
        parent = os.path.dirname(path)
        if not parent:
            # e.g. "drowsiness_events.db" in cwd
            return

        # Prevent accidental creation of a "logs" folder for DB storage
        # (This is typically a misconfiguration; DBs belong under data/.)
        if os.path.basename(parent) == "logs":
            raise ValueError(
                f"Refusing to create/use database under '{parent}/'. "
                f"Please use a data/ path (e.g. 'data/drowsiness_events.db'). Got: {path}"
            )

        os.makedirs(parent, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        # One connection per thread; default check_same_thread=True is fine in that case.
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
        return conn

    def _reset_conn(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        self._local.conn = None

    def _ensure_schema(self) -> None:
        """Create all tables if they don't exist (NO schema changes beyond what's already here)."""
        with self._connect() as conn:
            # User Profiles
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL UNIQUE,
                    full_name TEXT,
                    ear_threshold REAL NOT NULL,
                    face_encoding BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON user_profiles(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_seen ON user_profiles(last_seen)")

            # Drowsiness Events
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS drowsiness_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vehicle_identification_number TEXT,
                    user_id INTEGER,
                    time TEXT,
                    status TEXT,
                    img_drowsiness BLOB,
                    duration REAL DEFAULT 0.0,
                    value REAL DEFAULT 0.0,
                    alert_category TEXT,
                    alert_detail TEXT,
                    severity TEXT
                )
                """
            )

            # Migration for existing DBs (does NOT add anything new beyond existing intent)
            cursor = conn.execute("PRAGMA table_info(drowsiness_events)")
            existing_columns = {row[1] for row in cursor.fetchall()}

            new_columns = {
                "alert_category": "TEXT",
                "alert_detail": "TEXT",
                "severity": "TEXT",
            }

            for col_name, col_type in new_columns.items():
                if col_name not in existing_columns:
                    try:
                        conn.execute(f"ALTER TABLE drowsiness_events ADD COLUMN {col_name} {col_type}")
                        logging.info("✓ Added column: %s", col_name)
                    except sqlite3.OperationalError as e:
                        logging.warning("Column %s might already exist: %s", col_name, e)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_time ON drowsiness_events(time)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_user ON drowsiness_events(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_category ON drowsiness_events(alert_category)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_severity ON drowsiness_events(severity)")

            conn.commit()

    def execute(self, query: str, params: tuple = (), fetch: bool = False) -> Optional[Any]:
        """Execute query with retry logic. Returns rows (fetch=True) else lastrowid."""
        last_err: Optional[Exception] = None

        for attempt in range(self.DB_RETRY_ATTEMPTS):
            try:
                with self._lock:
                    conn = self._get_conn()
                    cur = conn.execute(query, params)
                    if fetch:
                        return cur.fetchall()
                    conn.commit()
                    return cur.lastrowid
            except sqlite3.OperationalError as e:
                last_err = e

                # If connection got into a bad state, recreate it for this thread
                msg = str(e).lower()
                if "closed" in msg or "cannot operate on a closed database" in msg:
                    self._reset_conn()

                if attempt < self.DB_RETRY_ATTEMPTS - 1:
                    logging.warning("DB locked/operational error, retry %d/%d: %s", attempt + 1, self.DB_RETRY_ATTEMPTS, e)
                    time.sleep(self.DB_RETRY_DELAY * (attempt + 1))
                    continue

                logging.error("DB operation failed: %s", e)
                raise

        if last_err:
            raise last_err
        return None

    def close(self) -> None:
        # Close current thread's connection (if any)
        self._reset_conn()
        return
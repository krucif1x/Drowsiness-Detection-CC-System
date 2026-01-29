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
            def _table_exists(name: str) -> bool:
                row = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (name,),
                ).fetchone()
                return row is not None

            # Rename legacy tables if needed
            if _table_exists("user_profiles") and not _table_exists("users"):
                conn.execute("ALTER TABLE user_profiles RENAME TO users")
            if _table_exists("drowsiness_events") and not _table_exists("events"):
                conn.execute("ALTER TABLE drowsiness_events RENAME TO events")

            # Users
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON users(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_seen ON users(last_seen)")

            # Events
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
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

            # Indexes (use the NEW table name)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_time ON events(time)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_user ON events(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_category ON events(alert_category)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_severity ON events(severity)")

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
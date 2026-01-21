import logging
import threading
import sqlite3
import queue
import time
import re
from datetime import datetime
from typing import Optional
from src.api_client.api_service import ApiService
from src.api_client.event import DrowsinessEvent as ApiEvent
from src.api_client import config

log = logging.getLogger(__name__)

class RemoteLogWorker:
    RETRY_INTERVAL_SEC = 30
    MAX_QUEUE_SIZE = 100
    SEND_BATCH_SIZE = 5
    MAX_RETRIES_PER_ROW = 20

    def __init__(self, db_path: str, remote_api_url: Optional[str] = None, enabled: bool = True):
        self.enabled = enabled
        self._db_lock = threading.Lock()

        target_base_url = remote_api_url if remote_api_url else config.SERVER_BASE_URL
        self.api_service = ApiService(base_url=target_base_url) if enabled else None
        if self.api_service:
            log.info(f"[REMOTE] Worker started (Base: {target_base_url})")
        else:
            log.info("[REMOTE] Worker disabled")

        self.queue_conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)

        with self._db_lock:
            self.queue_conn.execute("""
                CREATE TABLE IF NOT EXISTS remote_event_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vehicle_vin TEXT,
                    user_id INTEGER,
                    status TEXT,
                    time TEXT,
                    img_jpeg BLOB,
                    alert_category TEXT,
                    alert_detail TEXT,
                    severity TEXT,
                    local_event_id INTEGER,
                    retry_count INTEGER DEFAULT 0,
                    last_error TEXT,
                    last_attempt TEXT
                )
            """)
            self.queue_conn.execute("""
                CREATE TABLE IF NOT EXISTS remote_event_deadletter (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vehicle_vin TEXT,
                    user_id INTEGER,
                    status TEXT,
                    time TEXT,
                    img_jpeg BLOB,
                    alert_category TEXT,
                    alert_detail TEXT,
                    severity TEXT,
                    local_event_id INTEGER,
                    retry_count INTEGER,
                    last_error TEXT,
                    failed_at TEXT
                )
            """)
            self.queue_conn.commit()

        # Migrate existing installs (older DBs may lack columns)
        self._ensure_queue_schema()
        self._ensure_deadletter_schema()

        self._immediate_q: "queue.Queue[tuple]" = queue.Queue(maxsize=200)
        self._stop_event = threading.Event()

        self._retry_thread = None
        self._send_thread = None
        if self.enabled and self.api_service:
            self._send_thread = threading.Thread(target=self._send_loop, daemon=True)
            self._send_thread.start()
            self._retry_thread = threading.Thread(target=self._retry_loop, daemon=True)
            self._retry_thread.start()

    def _ensure_queue_schema(self) -> None:
        """Add missing columns for older DBs."""
        try:
            with self._db_lock:
                cur = self.queue_conn.cursor()
                cur.execute("PRAGMA table_info(remote_event_queue)")
                cols = {row[1] for row in cur.fetchall()}

                to_add = []
                if "img_jpeg" not in cols:
                    to_add.append(("img_jpeg", "BLOB"))
                if "alert_category" not in cols:
                    to_add.append(("alert_category", "TEXT"))
                if "alert_detail" not in cols:
                    to_add.append(("alert_detail", "TEXT"))
                if "severity" not in cols:
                    to_add.append(("severity", "TEXT"))
                if "retry_count" not in cols:
                    to_add.append(("retry_count", "INTEGER DEFAULT 0"))
                if "last_error" not in cols:
                    to_add.append(("last_error", "TEXT"))
                if "last_attempt" not in cols:
                    to_add.append(("last_attempt", "TEXT"))
                if "local_event_id" not in cols:
                    to_add.append(("local_event_id", "INTEGER"))

                for name, typ in to_add:
                    cur.execute(f"ALTER TABLE remote_event_queue ADD COLUMN {name} {typ}")
                if to_add:
                    self.queue_conn.commit()
        except Exception as e:
            log.warning(f"[REMOTE] Queue schema migration warning: {e}")

    def _ensure_deadletter_schema(self) -> None:
        """Add missing columns for older remote_event_deadletter schemas."""
        try:
            with self._db_lock:
                cur = self.queue_conn.cursor()
                cur.execute("PRAGMA table_info(remote_event_deadletter)")
                cols = {row[1] for row in cur.fetchall()}

                to_add = []
                if "img_jpeg" not in cols:
                    to_add.append(("img_jpeg", "BLOB"))
                if "alert_category" not in cols:
                    to_add.append(("alert_category", "TEXT"))
                if "alert_detail" not in cols:
                    to_add.append(("alert_detail", "TEXT"))
                if "severity" not in cols:
                    to_add.append(("severity", "TEXT"))
                if "local_event_id" not in cols:
                    to_add.append(("local_event_id", "INTEGER"))
                if "retry_count" not in cols:
                    to_add.append(("retry_count", "INTEGER"))
                if "last_error" not in cols:
                    to_add.append(("last_error", "TEXT"))
                if "failed_at" not in cols:
                    to_add.append(("failed_at", "TEXT"))

                for name, typ in to_add:
                    cur.execute(f"ALTER TABLE remote_event_deadletter ADD COLUMN {name} {typ}")
                if to_add:
                    self.queue_conn.commit()
        except Exception as e:
            log.warning(f"[REMOTE] Deadletter schema migration warning: {e}")

    def send_or_queue(self, vehicle_vin: str, user_id: int, status: str,
                      time_dt: datetime, raw_jpeg: Optional[bytes],
                      alert_category: Optional[str] = None,
                      alert_detail: Optional[str] = None,
                      severity: Optional[str] = None,
                      local_event_id: Optional[int] = None):
        """Push into immediate queue with new management fields."""
        if not (self.enabled and self.api_service):
            return
        try:
            self._immediate_q.put_nowait((
                vehicle_vin, user_id, status, time_dt, raw_jpeg,
                alert_category, alert_detail, severity, local_event_id
            ))
        except queue.Full:
            log.warning("[REMOTE] Immediate queue full; falling back to persistent queue")
            self._queue(vehicle_vin, user_id, status, time_dt, raw_jpeg,
                        alert_category, alert_detail, severity, local_event_id)

    def _send_event(self, vin, uid, status, dt, jpeg_bytes,
                    alert_category=None, alert_detail=None, severity=None) -> bool:
        """Send to API; sanitize fields to avoid server-side 500s from null/bad values."""
        try:
            dt_obj = dt if isinstance(dt, datetime) else datetime.now()

            # Sanitize status (remove spaces/special chars)
            norm_status = (status or "event").strip().lower()
            norm_status = re.sub(r"\s+", "_", norm_status)
            norm_status = re.sub(r"[^a-z0-9_]+", "_", norm_status)
            norm_status = re.sub(r"_+", "_", norm_status).strip("_")

            norm_cat = (alert_category.strip() if isinstance(alert_category, str) else alert_category)
            norm_detail = (alert_detail.strip() if isinstance(alert_detail, str) else alert_detail)
            norm_sev = (severity.strip() if isinstance(severity, str) else severity)

            # If image is missing, do NOT send (these are the rows showing jpeg_len=0 in your logs)
            if not jpeg_bytes:
                log.warning(
                    "[REMOTE] Skip send (missing image): vin=%s uid=%s status=%r cat=%r sev=%r time=%s",
                    vin, uid, norm_status, norm_cat, norm_sev, dt_obj.isoformat()
                )
                return False

            import base64
            jpeg_len = len(jpeg_bytes)
            b64 = base64.b64encode(jpeg_bytes).decode("ascii")

            event = ApiEvent(
                vehicle_identification_number=vin,
                user_id=uid,
                status=norm_status,
                time=dt_obj,
                img_drowsiness=b64,
                img_path=None
            )

            if hasattr(event, "alert_category"):
                event.alert_category = norm_cat
            if hasattr(event, "alert_detail"):
                event.alert_detail = norm_detail
            if hasattr(event, "severity"):
                event.severity = norm_sev

            res = self.api_service.send_drowsiness_event(event)
            if getattr(res, "success", False):
                log.info(f"[REMOTE] ✓ Sent {norm_cat or norm_status} (CID: {getattr(res, 'correlation_id', '-')})")
                return True

            log.warning(
                "[REMOTE] Send failed: error=%r status_code=%r vin=%s uid=%s status=%r cat=%r sev=%r jpeg_len=%s time=%s",
                getattr(res, "error", "unknown error"),
                getattr(res, "status_code", None),
                vin, uid, norm_status, norm_cat, norm_sev, jpeg_len, dt_obj.isoformat()
            )
            return False
        except Exception as e:
            log.error(f"[REMOTE] Send exception: {e}", exc_info=True)
            return False

    def _queue(self, vin, uid, status, dt, jpeg_bytes,
               alert_category=None, alert_detail=None, severity=None,
               local_event_id: Optional[int] = None):
        """Persist to DB with retry tracking."""
        try:
            t_str = dt.strftime("%Y-%m-%d %H:%M:%S") if isinstance(dt, datetime) else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            norm_status = (status or "event").strip().lower()

            with self._db_lock:
                cur = self.queue_conn.cursor()
                cur.execute("SELECT COUNT(*) FROM remote_event_queue")
                if cur.fetchone()[0] >= self.MAX_QUEUE_SIZE:
                    cur.execute("""
                        DELETE FROM remote_event_queue
                        WHERE id = (SELECT id FROM remote_event_queue ORDER BY id ASC LIMIT 1)
                    """)

                cur.execute("""
                    INSERT INTO remote_event_queue
                    (vehicle_vin, user_id, status, time, img_jpeg, alert_category, alert_detail, severity, local_event_id, retry_count, last_error, last_attempt)
                    VALUES (?,?,?,?,?,?,?,?,?,0,NULL,NULL)
                """, (vin, uid, norm_status, t_str, jpeg_bytes, alert_category, alert_detail, severity, local_event_id))
                self.queue_conn.commit()

            log.info("[REMOTE] ⧖ Queued: %s", alert_detail or norm_status)
        except Exception as e:
            log.error(f"[REMOTE] Queue error: {e}", exc_info=True)

    def _fetch_local_jpeg(self, local_event_id: int) -> Optional[bytes]:
        """Fetch jpeg bytes from local drowsiness_events table."""
        if not local_event_id:
            return None
        try:
            with self._db_lock:
                cur = self.queue_conn.cursor()
                cur.execute("SELECT img_drowsiness FROM drowsiness_events WHERE id = ? LIMIT 1", (int(local_event_id),))
                row = cur.fetchone()
            if not row:
                return None
            blob = row[0]
            return bytes(blob) if blob else None
        except Exception:
            return None

    def _process_queue(self):
        """Send queued rows; rehydrate image from local DB if missing."""
        try:
            with self._db_lock:
                cur = self.queue_conn.cursor()
                cur.execute("""
                    SELECT id, vehicle_vin, user_id, status, time, img_jpeg, alert_category, alert_detail, severity, local_event_id, retry_count
                    FROM remote_event_queue
                    ORDER BY id ASC
                    LIMIT ?
                """, (self.SEND_BATCH_SIZE,))
                rows = cur.fetchall()

            if not rows:
                return

            for row in rows:
                qid, vin, uid, stat, t_str, jpeg_bytes, alert_category, alert_detail, severity, local_event_id, retry_count = row
                try:
                    dt = datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    dt = datetime.now()

                # Rehydrate missing image from local events table
                if not jpeg_bytes and local_event_id:
                    jpeg_bytes = self._fetch_local_jpeg(local_event_id)
                    if jpeg_bytes:
                        with self._db_lock:
                            cur = self.queue_conn.cursor()
                            cur.execute("UPDATE remote_event_queue SET img_jpeg=? WHERE id=?", (jpeg_bytes, qid))
                            self.queue_conn.commit()

                # If still missing, dead-letter (prevents HTTP 500 spam)
                if not jpeg_bytes:
                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with self._db_lock:
                        cur = self.queue_conn.cursor()
                        cur.execute("""
                            INSERT INTO remote_event_deadletter
                            (vehicle_vin, user_id, status, time, img_jpeg, alert_category, alert_detail, severity, local_event_id, retry_count, last_error, failed_at)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                        """, (vin, uid, stat, t_str, None, alert_category, alert_detail, severity, local_event_id, int(retry_count or 0), "missing_img_jpeg", now_str))
                        cur.execute("DELETE FROM remote_event_queue WHERE id=?", (qid,))
                        self.queue_conn.commit()
                    log.error("[REMOTE] Dead-lettered id=%s (missing_img_jpeg)", qid)
                    continue

                ok = self._send_event(vin, uid, stat, dt, jpeg_bytes, alert_category, alert_detail, severity)
                if ok:
                    with self._db_lock:
                        cur = self.queue_conn.cursor()
                        cur.execute("DELETE FROM remote_event_queue WHERE id=?", (qid,))
                        self.queue_conn.commit()
                    log.info(f"[REMOTE] ✓ Retry Success ID:{qid}")
                    continue

                new_retry = int(retry_count or 0) + 1
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                with self._db_lock:
                    cur = self.queue_conn.cursor()
                    if new_retry >= self.MAX_RETRIES_PER_ROW:
                        cur.execute("""
                            INSERT INTO remote_event_deadletter
                            (vehicle_vin, user_id, status, time, img_jpeg, alert_category, alert_detail, severity, local_event_id, retry_count, last_error, failed_at)
                            SELECT vehicle_vin, user_id, status, time, img_jpeg, alert_category, alert_detail, severity, local_event_id, retry_count, last_error, ?
                            FROM remote_event_queue
                            WHERE id=?S
                        """, (now_str, qid))
                        cur.execute("DELETE FROM remote_event_queue WHERE id=?", (qid,))
                        log.error("[REMOTE] Dead-lettered queue row id=%s after %s retries", qid, new_retry)
                    else:
                        cur.execute("""
                            UPDATE remote_event_queue
                            SET retry_count=?, last_attempt=?
                            WHERE id=?
                        """, (new_retry, now_str, qid))
                    self.queue_conn.commit()
        except Exception as e:
            log.error(f"[REMOTE] Retry error: {e}", exc_info=True)

    def _send_loop(self) -> None:
        """Drain immediate queue; if jpeg missing, try rehydrate from local DB before sending."""
        while not self._stop_event.is_set():
            try:
                item = self._immediate_q.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                vin, uid, status, dt, jpeg_bytes, alert_category, alert_detail, severity, local_event_id = item

                if not jpeg_bytes and local_event_id:
                    jpeg_bytes = self._fetch_local_jpeg(local_event_id)

                ok = self._send_event(vin, uid, status, dt, jpeg_bytes, alert_category, alert_detail, severity)
                if not ok:
                    self._queue(vin, uid, status, dt, jpeg_bytes, alert_category, alert_detail, severity, local_event_id)
            except Exception as e:
                log.error("[REMOTE] Send loop error: %s", e, exc_info=True)
            finally:
                try:
                    self._immediate_q.task_done()
                except Exception:
                    pass

    def _retry_loop(self) -> None:
        """Periodically retry sending persisted queue rows."""
        while not self._stop_event.is_set():
            try:
                self._process_queue()
            except Exception as e:
                log.error("[REMOTE] Retry loop error: %s", e, exc_info=True)

            # Sleep in small increments so shutdown is responsive
            for _ in range(int(self.RETRY_INTERVAL_SEC * 10)):
                if self._stop_event.is_set():
                    break
                time.sleep(0.1)

    def close(self):
        self._stop_event.set()
        if self._send_thread:
            self._send_thread.join()
        if self._retry_thread:
            self._retry_thread.join()
        self.queue_conn.close()
        log.info("[REMOTE] Worker stopped")

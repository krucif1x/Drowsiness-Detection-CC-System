import logging
import threading
import sqlite3
import queue
import time
import re
import base64
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

    def __init__(self, db_path: str, remote_api_url: Optional[str] = None, enabled: bool = True):
        """
        db_path: path to the MAIN DB (contains users + events).
        """
        self.enabled = enabled
        self._db_lock = threading.Lock()

        target_base_url = remote_api_url if remote_api_url else config.SERVER_BASE_URL
        self.api_service = ApiService(base_url=target_base_url) if enabled else None
        if self.api_service:
            log.info(f"[REMOTE] Worker started (Base: {target_base_url})")
        else:
            log.info("[REMOTE] Worker disabled")

        # MAIN DB connection (read local events/images from here)
        self.events_conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)

        self._immediate_q: "queue.Queue[tuple]" = queue.Queue(maxsize=200)
        self._stop_event = threading.Event()

        self._retry_thread = None
        self._send_thread = None
        if self.enabled and self.api_service:
            self._send_thread = threading.Thread(target=self._send_loop, daemon=True)
            self._send_thread.start()
            self._retry_thread = threading.Thread(target=self._retry_loop, daemon=True)
            self._retry_thread.start()

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
            log.warning("[REMOTE] Immediate queue full; marking pending in events")
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

            # If image is missing, do NOT send
            if not jpeg_bytes:
                log.warning(
                    "[REMOTE] Skip send (missing image): vin=%s uid=%s status=%r cat=%r sev=%r time=%s",
                    vin, uid, norm_status, norm_cat, norm_sev, dt_obj.isoformat()
                )
                return False

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
                "[REMOTE] Send failed: error=%r status_code=%r vin=%s uid=%s status=%r cat=%r sev=%r time=%s",
                getattr(res, "error", "unknown error"),
                getattr(res, "status_code", None),
                vin, uid, norm_status, norm_cat, norm_sev, dt_obj.isoformat()
            )
            return False
        except Exception as e:
            log.error(f"[REMOTE] Send exception: {e}", exc_info=True)
            return False

    def _queue(self, vin, uid, status, dt, jpeg_bytes,
               alert_category=None, alert_detail=None, severity=None,
               local_event_id: Optional[int] = None):
        """Use events table as the outbox (status=pending)."""
        if local_event_id is None:
            log.warning("[REMOTE] No local_event_id; cannot mark pending in events")
            return
        try:
            with self._db_lock:
                self.events_conn.execute(
                    "UPDATE events SET status = ? WHERE id = ?",
                    ("pending", int(local_event_id)),
                )
                self.events_conn.commit()
            log.info("[REMOTE] ⧖ Marked pending: event_id=%s", local_event_id)
        except Exception as e:
            log.error(f"[REMOTE] Queue error: {e}", exc_info=True)

    def _fetch_local_jpeg(self, local_event_id: int) -> Optional[bytes]:
        """Fetch jpeg bytes from MAIN local events table."""
        with self._db_lock:
            row = self.events_conn.execute(
                "SELECT img_drowsiness FROM events WHERE id = ?",
                (int(local_event_id),),
            ).fetchone()
        return row[0] if row and row[0] else None

    def _process_queue(self):
        if not self.api_service:
            return

        with self._db_lock:
            rows = self.events_conn.execute(
                """
                SELECT id, vehicle_identification_number, user_id, time, status,
                       img_drowsiness, duration, value, alert_category, alert_detail, severity
                FROM events
                WHERE status = ?
                ORDER BY id ASC
                LIMIT ?
                """,
                ("pending", self.SEND_BATCH_SIZE),
            ).fetchall()

        for (eid, vin, uid, time_str, status, img_blob, duration,
             value, alert_category, alert_detail, severity) in rows:

            # If status is queue state, send event type from alert_category
            payload_status = alert_category if status in ("pending", "sent") else status

            ok = self._send_event(
                vin, uid, payload_status, time_str, img_blob,
                alert_category=alert_category,
                alert_detail=alert_detail,
                severity=severity,
            )

            if ok:
                with self._db_lock:
                    self.events_conn.execute(
                        "UPDATE events SET status = ? WHERE id = ?",
                        ("sent", int(eid)),
                    )
                    self.events_conn.commit()
                    log.info("[REMOTE] ✓ Sent event_id=%s", eid)

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
        """Periodically retry sending pending rows from events table."""
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
        self.events_conn.close()
        log.info("[REMOTE] Worker stopped")

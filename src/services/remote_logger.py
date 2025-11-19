import logging
import threading
import sqlite3
from datetime import datetime
from typing import Optional
from src.api_client.api_service import ApiService
from src.api_client.event import DrowsinessEvent as ApiEvent
from src.api_client import config # <--- Syncing with config

log = logging.getLogger(__name__)

class RemoteLogWorker:
    RETRY_INTERVAL_SEC = 30
    MAX_QUEUE_SIZE = 100
    
    def __init__(self, db_path: str, remote_api_url: Optional[str] = None, enabled: bool = True):
        self.enabled = enabled
        
        # SYNCHRONIZATION FIX:
        # If remote_api_url is None, use config.SERVER_BASE_URL.
        target_base_url = remote_api_url if remote_api_url else config.SERVER_BASE_URL
        
        if enabled:
            self.api_service = ApiService(base_url=target_base_url)
            log.info(f"[REMOTE] Worker started (Base: {target_base_url})")
        else:
            self.api_service = None
            log.info("[REMOTE] Worker disabled")
        
        # Queue DB connection
        self.queue_conn = sqlite3.connect(db_path, check_same_thread=False)
        # We ensure the table exists. 
        # Note: This table is for the RETRY QUEUE only.
        self.queue_conn.execute("""
            CREATE TABLE IF NOT EXISTS remote_event_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                vehicle_vin TEXT, 
                user_id INTEGER, 
                status TEXT, 
                time TEXT, 
                img_base64 TEXT, 
                img_path TEXT, 
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.queue_conn.commit()
        
        self._stop_event = threading.Event()
        self._retry_thread = None
        
        if self.enabled and self.api_service:
            self._retry_thread = threading.Thread(target=self._retry_loop, daemon=True)
            self._retry_thread.start()

    def send_or_queue(self, vehicle_vin: str, user_id: int, status: str, 
                      time_dt: datetime, img_base64: Optional[str], img_path: Optional[str]):
        """
        Attempt to send immediately. If failed or offline, add to queue.
        """
        if not self.enabled or not self.api_service:
            return

        success = self._send(vehicle_vin, user_id, status, time_dt, img_base64, img_path)
        
        if not success:
            self._queue(vehicle_vin, user_id, status, time_dt, img_base64, img_path)

    def _send(self, vin, uid, status, dt, b64, path):
        try:
            event = ApiEvent(
                vehicle_identification_number=vin,
                user_id=uid,
                status=status,
                time=dt,
                img_drowsiness=b64,
                img_path=path
            )
            res = self.api_service.send_drowsiness_event(event)
            if res.success:
                log.info(f"[REMOTE] ✓ Sent (CID: {res.correlation_id})")
                return True
            
            # Log the specific error (e.g. Timeout)
            log.warning(f"[REMOTE] Send failed: {res.error}")
            return False
        except Exception as e:
            log.error(f"[REMOTE] Send exception: {e}")
            return False

    def _queue(self, vin, uid, status, dt, b64, path):
        try:
            time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            cur = self.queue_conn.cursor()
            
            # Check size limit
            cur.execute("SELECT COUNT(*) FROM remote_event_queue")
            if cur.fetchone()[0] >= self.MAX_QUEUE_SIZE:
                # Delete oldest
                cur.execute("DELETE FROM remote_event_queue WHERE id = (SELECT id FROM remote_event_queue ORDER BY created_at ASC LIMIT 1)")
            
            cur.execute("""
                INSERT INTO remote_event_queue 
                (vehicle_vin, user_id, status, time, img_base64, img_path) 
                VALUES (?,?,?,?,?,?)
            """, (vin, uid, status, time_str, b64, path))
            
            self.queue_conn.commit()
            log.info("[REMOTE] ⧖ Queued for retry")
        except Exception as e:
            log.error(f"[REMOTE] Queue error: {e}")

    def _retry_loop(self):
        while not self._stop_event.wait(self.RETRY_INTERVAL_SEC):
            self._process_queue()

    def _process_queue(self):
        try:
            cur = self.queue_conn.cursor()
            cur.execute("SELECT * FROM remote_event_queue ORDER BY created_at ASC LIMIT 5")
            rows = cur.fetchall()
            if not rows: return
            
            for row in rows:
                # row indices: 0=id, 1=vin, 2=uid, 3=status, 4=time, 5=b64, 6=path
                qid, vin, uid, stat, t_str, b64, path = row[0], row[1], row[2], row[3], row[4], row[5], row[6]
                
                try:
                    dt = datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S")
                except:
                    dt = datetime.now()

                if self._send(vin, uid, stat, dt, b64, path):
                    cur.execute("DELETE FROM remote_event_queue WHERE id=?", (qid,))
                    log.info(f"[REMOTE] ✓ Retry Success ID:{qid}")
            self.queue_conn.commit()
        except Exception as e:
            log.error(f"[REMOTE] Retry error: {e}")

    def close(self):
        self._stop_event.set()
        if self._retry_thread: 
            self._retry_thread.join(1.0)
        self.queue_conn.close()
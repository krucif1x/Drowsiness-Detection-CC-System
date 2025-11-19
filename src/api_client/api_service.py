# api_service.py
import logging
import uuid
import requests
from typing import Optional

from .event import DrowsinessEvent
from . import config  # <--- Syncing with config

log = logging.getLogger(__name__)

class ApiResult:
    def __init__(self, success: bool, status_code: int, text: str, correlation_id: str):
        self.success = success
        self.status_code = status_code
        self.text = text
        self.correlation_id = correlation_id
        self.error: Optional[str] = None

class ApiService:
    def __init__(self, base_url: str = config.SERVER_BASE_URL, timeout: float = config.DEFAULT_TIMEOUT):
        """
        Initialize the service.
        :param base_url: The host URL (e.g. http://ip:port). Defaults to config.SERVER_BASE_URL.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # Strictly use the path defined in config
        self.path = config.DROWSINESS_EVENT_PATH
        
        log.info("[API] Initialized target=%s path=%s timeout=%.1fs", self.base_url, self.path, self.timeout)

    def _url(self) -> str:
        """Constructs the full URL by combining Base + Path"""
        return f"{self.base_url}{self.path}"

    def send_drowsiness_event(self, event: DrowsinessEvent) -> ApiResult:
        cid = str(uuid.uuid4())
        idem = str(uuid.uuid4())
        headers = {
            "Content-Type": "application/json",
            config.CORRELATION_HEADER: cid,
            config.IDEMPOTENCY_HEADER: idem,
        }
        payload = event.to_transport_payload()
        
        full_url = self._url() # Resolves to http://IP:PORT/api/v1/drowsiness
        
        try:
            resp = requests.post(full_url, json=payload, headers=headers, timeout=self.timeout)
            ok = 200 <= resp.status_code < 300
            result = ApiResult(ok, resp.status_code, resp.text, cid)
            if not ok:
                result.error = f"HTTP {resp.status_code}"
            return result
        except requests.RequestException as e:
            # This catches the ConnectTimeoutError and returns it safely
            r = ApiResult(False, 0, "", cid)
            r.error = str(e)
            return r
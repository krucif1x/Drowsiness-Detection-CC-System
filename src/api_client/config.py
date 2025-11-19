"""
API client configuration (env-friendly) with correct server path.
"""
from typing import Final
import os

# 1. The Base URL (IP and Port only)
# We use rstrip('/') to ensure we don't end up with double slashes later
SERVER_BASE_URL: Final = os.getenv("DS_REMOTE_URL", "http://203.100.57.59:3000").rstrip("/")

# 2. The specific API Version and Path
API_VERSION: Final = os.getenv("DS_API_VERSION", "v1")

# UPDATE: Added the trailing slash '/' at the end as requested
DROWSINESS_EVENT_PATH: Final = f"/api/{API_VERSION}/drowsiness/"

# 3. Timeouts and Headers
DEFAULT_TIMEOUT: Final = float(os.getenv("DS_REMOTE_TIMEOUT", "10"))
CORRELATION_HEADER: Final = "X-Correlation-Id"
IDEMPOTENCY_HEADER: Final = "X-Idempotency-Key"
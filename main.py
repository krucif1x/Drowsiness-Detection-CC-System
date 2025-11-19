import os
import sys
import logging
import platform

# 1. CONFIGURATION (Before imports to ensure they take effect)
# -----------------------------------------------------------
# Set log levels to reduce spam from libraries
os.environ["ORT_LOGGING_LEVEL"] = "3"           # Silence ONNX Runtime
os.environ["LIBCAMERA_LOG_LEVELS"] = "ERROR"    # Silence LibCamera

# Auto-detect hardware for camera source
if platform.machine().startswith(("arm", "aarch")):
    os.environ.setdefault("DS_CAMERA_SOURCE", "picamera2")  # Raspberry Pi
else:
    os.environ.setdefault("DS_CAMERA_SOURCE", "opencv")     # Laptop/PC

# 2. GLOBAL LOGGING SETUP
# -----------------------------------------------------------
# Without this, your logging.info() calls inside the app might not show up!
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Silence specific noisy loggers
logging.getLogger("picamera2").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)

# 3. IMPORT & EXECUTION
# -----------------------------------------------------------
try:
    # Updated Import Path: src.app -> src.core
    # Updated Class Name: DrowsinessApp -> DrowsinessSystem
    from src.core.orchestrator import DrowsinessSystem
    
    if __name__ == "__main__":
        app = DrowsinessSystem()
        app.run()

except ImportError as e:
    logging.critical(f"ImportError: {e}", exc_info=True)
except Exception as e:
    logging.critical(f"Fatal error: {e}", exc_info=True)
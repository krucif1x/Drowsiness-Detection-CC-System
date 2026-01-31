import logging
import cv2
import yaml
import os
import numpy as np
from pathlib import Path
from src.infrastructure.hardware.camera import Camera
from src.face_recognition.user_manager import UserManager
from src.logging.remote_logger import RemoteLogWorker
from src.logging.system_logger import SystemLogger
from src.infrastructure.data.database import UnifiedDatabase
from src.infrastructure.data.repository import UnifiedRepository
from src.mediapipe.face_mesh import MediaPipeFaceModel
from src.app.detection_loop import DetectionLoop

log = logging.getLogger(__name__)

class DrowsinessSystem:
    CONFIG_PATH = "config/detector_config.yaml"
    DB_PATH = os.getenv("CC_DB_PATH", os.path.join("data", "cc.db"))

    def __init__(self):
        self._ensure_paths()
        self.config = self._load_config()
        sys_cfg = self.config.get('system', {})
        self.vin = sys_cfg.get('vin', os.getenv('DS_VIN', 'VIN-0001'))
        self.fps = float(sys_cfg.get('target_fps', 30.0))

    def run(self):
        try:
            self._init_resources()
            log.info(f"System Ready. VIN: {self.vin}")
            print("\n>>> PRESS 'Q' TO EXIT <<<\n")

            DetectionLoop(
                camera=self.camera,
                face_mesh=self.face_mesh,
                user_manager=self.user_manager,
                system_logger=self.system_logger,
                vehicle_vin=self.vin,
                fps=self.fps,
                detector_config_path=self.CONFIG_PATH,
            ).run()

        except KeyboardInterrupt:
            log.info("User requested shutdown.")
        except Exception as e:
            log.error(f"Fatal: {e}", exc_info=True)
        finally:
            self._cleanup()

    def _init_resources(self):
        # 1. Unified Database & Repository
        self.db = UnifiedDatabase(self.DB_PATH)
        self.repo = UnifiedRepository(self.db)

        # 2. Services
        self.user_manager = UserManager(database_file=self.DB_PATH)
        self.remote_worker = RemoteLogWorker(self.DB_PATH, os.getenv("DS_REMOTE_URL"), True)
        self.system_logger = SystemLogger(self.remote_worker, self.repo, self.vin)

        # 3. Hardware
        self.camera = Camera(source='auto', resolution=(640, 480))
        if not self.camera.ready:
            raise RuntimeError("Camera failed to open")

        # 4. AI Model (Warmup with generated black frame)
        self.face_mesh = MediaPipeFaceModel(max_num_faces=1, refine_landmarks=True)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.face_mesh.process(dummy_frame)

    def _cleanup(self):
        log.info("Shutting down...")
        if hasattr(self, 'remote_worker') and self.remote_worker: self.remote_worker.close()
        if hasattr(self, 'db') and self.db: self.db.close()
        if hasattr(self, 'camera') and self.camera: self.camera.release()
        if hasattr(self, 'face_mesh') and self.face_mesh: self.face_mesh.close()
        cv2.destroyAllWindows()

    def _ensure_paths(self):
        Path("data").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)

    def _load_config(self):
        if Path(self.CONFIG_PATH).exists():
            with open(self.CONFIG_PATH) as f:
                return yaml.safe_load(f)
        return {}
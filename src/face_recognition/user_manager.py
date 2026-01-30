import logging
import torch
import numpy as np
import os
import time
from typing import List, Optional, Tuple
from threading import Lock
from collections import deque

from src.infrastructure.data.models import UserProfile
from src.infrastructure.data.database import UnifiedDatabase
from src.infrastructure.data.repository import UnifiedRepository
from src.face_recognition.face_recognizer import FaceRecognizer
from src.face_recognition.similarity_matcher import SimilarityMatcher

class UserManager:
    def __init__(
        self, 
        database_file: str = "data/drowsiness_events.db",  # changed (POSIX-friendly)
        recognition_threshold: float = 0.6,  # Conservative threshold for better accuracy

        # Multi-frame validation reduces false positives significantly
        multi_frame_validation: bool = True,
        min_consistent_frames: int = 5,  # Require 3 consistent detections

        # Quality thresholds
        min_face_confidence: float = 0.95,  # Only use high-quality detections

        input_color: str = "RGB",
        fast_accept_ratio: float = 0.80,  # accept immediately if dist <= threshold*ratio
        consensus_ratio: float = 0.60,  # fraction of frames that must agree
    ):
        database_file = os.path.normpath(database_file)
        logging.info(f"UserManager initializing with database: {database_file}")

        self.recognition_threshold = recognition_threshold
        self.input_color = (input_color or "RGB").upper()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.multi_frame_validation = multi_frame_validation
        self.min_consistent_frames = min_consistent_frames
        self.min_face_confidence = min_face_confidence

        self.fast_accept_ratio = float(fast_accept_ratio)
        self.consensus_ratio = float(consensus_ratio)

        self._recent_matches: deque = deque(maxlen=min_consistent_frames)
        self._lock = Lock()

        self.db = UnifiedDatabase(database_file)
        self.repo = UnifiedRepository(self.db)
        self.recognizer = FaceRecognizer(
            device=self.device,
            input_color=self.input_color,
            min_detection_prob=min_face_confidence,
        )

        self.matcher = SimilarityMatcher(distance_metric="euclidean")

        self.users: List[UserProfile] = []
        self._user_id_map: dict[int, UserProfile] = {}

        self._match_stats = {
            'total_attempts': 0,
            'successful_matches': 0,
            'failed_matches': 0,
            'avg_match_distance': []
        }

        # Face-recognition logging throttle (reduce spam)
        self._fr_log_last_ts = 0.0
        self._fr_log_interval_sec = float(os.getenv("DS_FR_LOG_INTERVAL_SEC", "10.0"))
        self._fr_last_kind: str | None = None

        self.load_users()
        logging.info(f"UserManager initialized successfully with threshold={recognition_threshold}")

    def load_users(self):
        """Load all user profiles from database into memory cache."""
        try:
            with self._lock:
                self.users = self.repo.load_all_users()
                self._user_id_map = {u.user_id: u for u in self.users}
                self.matcher.build_matrix(self.users)
            logging.info(f"Loaded {len(self.users)} user profile(s)")
        except Exception as e:
            logging.error(f"Error loading users: {e}")
            self.users = []
            self._user_id_map = {}

    def _rate_limited_log(self, kind: str, msg: str, level: int = logging.INFO) -> None:
        """
        Log at most once per interval per 'kind' (e.g., no_match/match/fast_match),
        and immediately when the kind changes.
        """
        now = time.time()
        if self._fr_last_kind != kind or (now - self._fr_log_last_ts) >= self._fr_log_interval_sec:
            self._fr_last_kind = kind
            self._fr_log_last_ts = now
            logging.log(level, msg)

    def find_best_match(self, image_frame, use_metadata: bool = False) -> Optional[UserProfile]:
        """
        Find best matching user from image frame with enhanced validation.
        Uses multi-frame consistency check to reduce false positives.

        Args:
            image_frame: Input frame to match
            use_metadata: If True, log detailed extraction metadata
        """
        if not self.users:
            logging.debug("No users in database")
            return None

        if use_metadata:
            encoding, metadata = self.recognizer.extract(image_frame, return_metadata=True)
            if encoding is None:
                logging.debug("Extraction failed: %s", metadata)
                return None
            logging.debug("Extraction metadata: %s", metadata)
        else:
            encoding = self.recognizer.extract(image_frame)
            if encoding is None:
                logging.debug("No face detected in frame")
                return None

        with self._lock:
            best_user, dist = self.matcher.best_match(
                encoding,
                self.users,
                threshold=self.recognition_threshold
            )
            self._match_stats['total_attempts'] += 1

        if self.multi_frame_validation:
            self._recent_matches.append(best_user.user_id if best_user else None)

        # Detailed numbers -> DEBUG (so INFO stays quiet)
        logging.debug(
            "FR candidate=%s dist=%.4f threshold=%.4f",
            (best_user.user_id if best_user else None),
            float(dist),
            float(self.recognition_threshold),
        )

        if (best_user is None) or (dist >= self.recognition_threshold):
            self._match_stats['failed_matches'] += 1
            self._rate_limited_log(
                "no_match",
                f"FaceRec: NO MATCH (threshold={self.recognition_threshold:.2f})",
                level=logging.INFO,
            )
            return None

        if dist <= (self.recognition_threshold * self.fast_accept_ratio):
            self._rate_limited_log(
                "fast_match",
                f"FaceRec: FAST MATCH user_id={best_user.user_id}",
                level=logging.INFO,
            )
            self._match_stats['successful_matches'] += 1
            self.repo.update_last_seen(best_user.user_id)
            return best_user

        if self.multi_frame_validation:
            if len(self._recent_matches) < self.min_consistent_frames:
                return None

            ids = [uid for uid in self._recent_matches if uid is not None]
            if not ids:
                return None

            most_common_id = max(set(ids), key=ids.count)
            need = int(self.min_consistent_frames * self.consensus_ratio)

            if ids.count(most_common_id) < need:
                return None
            if most_common_id != best_user.user_id:
                return None

        self._rate_limited_log(
            "match",
            f"FaceRec: MATCH user_id={best_user.user_id}",
            level=logging.INFO,
        )
        self._match_stats['successful_matches'] += 1
        self.repo.update_last_seen(best_user.user_id)
        return best_user

    def register_new_user(
        self, 
        image_frame, 
        ear_threshold: float, 
        user_id: Optional[int] = None,
        require_multiple_frames: bool = False,
        additional_frames: Optional[List] = None
    ) -> Optional[UserProfile]:
        if ear_threshold is None: 
            return None

        encoding = self.recognizer.extract(image_frame)
        if encoding is None: 
            logging.error("Cannot register: No face encoding extracted from primary frame")
            return None

        encodings = [encoding]

        if require_multiple_frames and additional_frames:
            logging.info(f"Multi-frame registration with {len(additional_frames)} additional frames...")
            for frame in additional_frames:
                enc = self.recognizer.extract(frame)
                if enc is not None:
                    encodings.append(enc)
            logging.info(f"Collected {len(encodings)} valid encodings from {len(additional_frames) + 1} frames")

        final_encoding = np.mean(encodings, axis=0)
        final_encoding = final_encoding / (np.linalg.norm(final_encoding) + 1e-8)

        registration_threshold = self.recognition_threshold * 0.8

        with self._lock:
            duplicate, dist = self.matcher.best_match(
                final_encoding, 
                self.users, 
                threshold=registration_threshold
            )

        if duplicate:
            logging.warning(
                f"⚠ DUPLICATE DETECTED: User ID {duplicate.user_id} already exists "
                f"(Distance: {dist:.4f}, Threshold: {registration_threshold:.4f}). "
                f"Returning existing user."
            )
            return duplicate

        if user_id is None:
            user_id = self.repo.get_next_user_id()

        new_user = UserProfile(0, user_id, ear_threshold, final_encoding)

        try:
            new_user.id = self.repo.save_user(new_user)

            with self._lock:
                self.users.append(new_user)
                self._user_id_map[user_id] = new_user
                self.matcher.build_matrix(self.users)

            logging.info(
                f"✓ NEW USER REGISTERED: ID={user_id}, "
                f"EAR Threshold={ear_threshold:.2f}, "
                f"Encoding Quality: Valid"
            )
            return new_user
        except Exception as e:
            logging.error(f"✗ User registration failed: {e}")
            return None


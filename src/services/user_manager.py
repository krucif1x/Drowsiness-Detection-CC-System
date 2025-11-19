import logging
import torch
import time
import numpy as np
from typing import List, Optional
from threading import Lock

from src.infrastructure.data.models import UserProfile
from src.infrastructure.data.database import UnifiedDatabase
from src.infrastructure.data.repository import UnifiedRepository
from src.utils.models.model_loader import FaceModelLoader
from src.utils.preprocessing.image_validator import ImageValidator
from src.utils.face.encoding_extractor import FaceEncodingExtractor
from src.utils.face.similarity_matcher import SimilarityMatcher

class UserManager:
    def __init__(
        self, 
        database_file: str = r"data\drowsiness_events.db",
        # STRICTER threshold for better accuracy
        # Lower value = more strict = fewer false matches
        # Start with 0.30, increase ONLY if it fails to recognize YOU
        recognition_threshold: float = 0.30, 
        
        input_color: str = "RGB"
    ):
        logging.info(f"UserManager initializing with database: {database_file}")
        
        self.recognition_threshold = recognition_threshold
        self.input_color = (input_color or "RGB").upper()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self._lock = Lock()
        
        # Components
        self.db = UnifiedDatabase(database_file)
        self.repo = UnifiedRepository(self.db)
        self.model_loader = FaceModelLoader(device=str(self.device))
        self.validator = ImageValidator(self.input_color)
        self.encoder = FaceEncodingExtractor(self.model_loader, self.validator, self.device)
        self.matcher = SimilarityMatcher()

        # Caches
        self.users: List[UserProfile] = []
        self._user_id_map: dict[int, UserProfile] = {}
        
        self._load_users()
        logging.info("UserManager initialized successfully")

    def _load_users(self):
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

    def find_best_match(self, image_frame) -> Optional[UserProfile]:
        """Find best matching user from image frame."""
        if not self.users:
            return None

        encoding = self.encoder.extract(image_frame)
        if encoding is None:
            return None

        with self._lock:
            best_user, dist = self.matcher.best_match(
                encoding, 
                self.users, 
                threshold=self.recognition_threshold
            )

        if best_user:
            logging.info(f"✓ MATCH FOUND: User ID {best_user.user_id} (Distance: {dist:.3f})")
            self.repo.update_last_seen(best_user.user_id)
            return best_user
        else:
            logging.info(f"✗ NO MATCH (Closest distance: {dist:.3f}, Threshold: {self.recognition_threshold})")
            return None

    def register_new_user(self, image_frame, ear_threshold: float, user_id: Optional[int] = None) -> Optional[UserProfile]:
        """
        Register new user with face encoding and EAR threshold.
        Uses SAME threshold as recognition for consistency.
        """
        if ear_threshold is None: 
            return None

        encoding = self.encoder.extract(image_frame)
        if encoding is None: 
            logging.error("Cannot register: No face encoding extracted")
            return None

        with self._lock:
            # Use SAME threshold for duplicate check
            duplicate, dist = self.matcher.best_match(
                encoding, 
                self.users, 
                threshold=self.recognition_threshold
            )

        if duplicate:
            logging.warning(
                f"⚠ DUPLICATE DETECTED: User ID {duplicate.user_id} already exists "
                f"(Distance: {dist:.3f}). Returning existing user."
            )
            return duplicate

        # Get next user_id if not provided
        if user_id is None:
            user_id = self.repo.get_next_user_id()

        # Create User Object
        new_user = UserProfile(0, user_id, ear_threshold, encoding)

        try:
            # Save to DB
            new_user.id = self.repo.save_user(new_user)
            
            # Update Memory Cache
            with self._lock:
                self.users.append(new_user)
                self._user_id_map[user_id] = new_user
                self.matcher.build_matrix(self.users)
            
            logging.info(f"✓ NEW USER REGISTERED: ID={user_id}, EAR Threshold={ear_threshold:.2f}")
            return new_user
        except Exception as e:
            logging.error(f"❌ User registration failed: {e}")
            return None
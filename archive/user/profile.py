import numpy as np


class UserProfile:
    """Data class for a user with server reference only."""
    __slots__ = ('id', 'user_id', 'ear_threshold', 'face_encoding')  # Memory optimization
    
    def __init__(
        self, 
        profile_id: int,
        user_id: int,
        ear_threshold: float, 
        face_encoding: np.ndarray
    ):
        self.id = profile_id
        self.user_id = user_id
        self.ear_threshold = ear_threshold
        
        # L2-normalize once at creation
        enc = np.array(face_encoding, dtype=np.float32).flatten()
        norm = np.linalg.norm(enc) + 1e-8
        self.face_encoding = enc / norm

    def __repr__(self) -> str:
        return f"UserProfile(id={self.id}, user_id={self.user_id}, ear_threshold={self.ear_threshold:.3f})"
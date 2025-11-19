import numpy as np
from typing import Optional, Tuple, List
from src.infrastructure.data.models import UserProfile

class SimilarityMatcher:
    def __init__(self):
        self._mat: Optional[np.ndarray] = None

    def build_matrix(self, users: List[UserProfile]):
        """Build matrix from user face encodings."""
        if not users:
            self._mat = None
            return
        
        encs = [u.face_encoding for u in users]
        self._mat = np.stack(encs, axis=0).astype(np.float32)
        # Note: Encodings are already normalized by encoding_extractor.py, 
        # so we don't strictly need to normalize again, but it's safer to do so.
        norms = np.linalg.norm(self._mat, axis=1, keepdims=True) + 1e-8
        self._mat /= norms

    def best_match(self, encoding: np.ndarray, users: List[UserProfile], threshold: float = 0.60) -> Tuple[Optional[UserProfile], float]:
        """
        Find best matching user using Euclidean Distance (L2).
        LOWER score is better.
        """
        if not users or self._mat is None:
            return None, 99.9 # Return high distance if no DB

        # 1. Normalize the Query Vector 
        q = encoding / (np.linalg.norm(encoding) + 1e-8)
        
        # 2. Calculate Euclidean Distance (L2)
        # formula: dist = || db_vec - query_vec ||
        # axis=1 calculates distance for every user in the matrix
        dists = np.linalg.norm(self._mat - q, axis=1)
        
        # 3. Find the index of the SMALLEST distance (Closest Face)
        i = int(np.argmin(dists))
        best_dist = float(dists[i])

        # 4. Filter by Threshold (Strict)
        # For FaceNet: 
        #   < 0.6  -> Likely same person
        #   > 0.6  -> Different person
        if best_dist > threshold:
            return None, best_dist

        return users[i], best_dist
from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass(frozen=True)
class EyeCfg:
    all: List[int]
    ear: List[int]
    frames: int

@dataclass(frozen=True)
class MouthCfg:
    mar: List[int]
    outline: List[int]
    thresh: float
    frames: int

@dataclass(frozen=True)
class CalibCfg:
    dur: int
    factor: float

@dataclass(frozen=True)
class BufCfg:
    dur: float

@dataclass(frozen=True)
class BlinkCfg:
    cnt: int
    closed: bool

@dataclass
class FpsCfg:
    prev: float
    fps: float

# --- EYE & MOUTH INDICES ---
L_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
R_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
L_EAR = [263, 387, 385, 362, 380, 373]
R_EAR = [33, 160, 158, 133, 153, 144]
M_MAR = [61, 82, 312, 291, 317, 87]
M_OUT = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]

# --- FACE MESH INDICES ---
SILHOUETTE = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]

# (Leaving existing face indices as they were for brevity/compatibility...)
# ... [Assuming existing long lists of face indices remain here] ...

MIDWAY_BETWEEN_EYES = [168]
NOSE_TIP = [1]

HEAD_POSE_IDX = [1, 9, 10, 33, 50, 54, 61, 84, 93, 103, 117, 133, 
    145, 150, 153, 162, 172, 181, 234, 263, 283, 284, 291, 312, 
    327, 338, 350, 356, 361, 373, 405, 425, 466
]

# --- HAND LANDMARK CONSTANTS (Refactored) ---
class HandIdx:
    """Static constants for accessing hand landmark arrays."""
    WRIST = 0
    
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20
    
# ================================================================
#                UPPER BODY POSE CONSTANTS (DRIVER POSTURE)
# ================================================================

class UpperBodyIdx:
    """Only the essential upper-body pose landmarks."""

    # Head & face reference
    NOSE = 0
    LEFT_EYE = 2
    RIGHT_EYE = 5

    # Upper body core joints
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16

    # Torso reference (needed for angle estimations)
    LEFT_HIP = 23
    RIGHT_HIP = 24

UPPER_BODY_LANDMARKS = [
    UpperBodyIdx.NOSE,
    UpperBodyIdx.LEFT_EYE, UpperBodyIdx.RIGHT_EYE,
    UpperBodyIdx.LEFT_SHOULDER, UpperBodyIdx.RIGHT_SHOULDER,
    UpperBodyIdx.LEFT_ELBOW, UpperBodyIdx.RIGHT_ELBOW,
    UpperBodyIdx.LEFT_WRIST, UpperBodyIdx.RIGHT_WRIST,
    UpperBodyIdx.LEFT_HIP, UpperBodyIdx.RIGHT_HIP
]

# 3D model points used for solvePnP (nose/chin/eye corners/mouth corners)
MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, 330.0, -65.0),         # Chin
        (-225.0, -170.0, -135.0),    # Left eye corner
        (225.0, -170.0, -135.0),     # Right eye corner
        (-150.0, 150.0, -125.0),     # Left mouth corner
        (150.0, 150.0, -125.0),      # Right mouth corner
    ],
    dtype=np.float64,
)

# FaceMesh landmark indices matching the 3D model points above
LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]

# Configuration constants
CONSEC_FRAMES = 30
MAR_THRES = 0.23
CALIBRATION_DURATION = 20
DROWSINESS_FACTOR = 0.8
BUFFER_DURATION = 1.0    

# Config instances
LEFT_EYE = EyeCfg(L_EYE, L_EAR, CONSEC_FRAMES)
RIGHT_EYE = EyeCfg(R_EYE, R_EAR, CONSEC_FRAMES)
MOUTH = MouthCfg(M_MAR, M_OUT, MAR_THRES, CONSEC_FRAMES)
CALIB = CalibCfg(CALIBRATION_DURATION, DROWSINESS_FACTOR)
BUF = BufCfg(BUFFER_DURATION)
BLINK = BlinkCfg(0, False)
FPS = FpsCfg(0, 0)
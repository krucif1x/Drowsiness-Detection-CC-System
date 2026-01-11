import os
import sys
import time
import argparse
import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.infrastructure.hardware.camera import Camera
from src.services.user_manager import UserManager

def enroll_user(cam, user_manager, user_id, ear_threshold=0.22, num_frames=5):
    print("\n[STEP] Enrollment: Please CENTER your face and look at the camera.")
    frames = []
    for i in range(num_frames):
        for _ in range(10):  # Try up to 10 times to get a valid frame
            frame = cam.read()
            if frame is not None:
                break
            cv2.waitKey(1)
            
        if frame is not None:
            display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.putText(display, f"Enrollment Frame {i+1}/{num_frames}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
            cv2.imshow("Enrollment", display)
            cv2.waitKey(700)  # Hold for 0.7s per frame
            frames.append(frame.copy())
    cv2.destroyWindow("Enrollment")
    if not frames:
        print("[ERROR] No frames captured for enrollment.")
        return None
    primary = frames[0]
    additional = frames[1:] if len(frames) > 1 else None
    user = user_manager.register_new_user(
        primary,
        ear_threshold=ear_threshold,
        user_id=user_id,
        require_multiple_frames=additional is not None,
        additional_frames=additional,
    )
    return user

def live_recognition(cam, user_manager):
    print("\n[STEP] Live recognition started. Press 'q' to quit.")
    cv2.namedWindow("Live Recognition", cv2.WINDOW_NORMAL)
    while True:
        frame = cam.read()
        if frame is None:
            cv2.waitKey(1)
            continue
        user = user_manager.find_best_match(frame)
        disp = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        txt = f"User: {user.user_id}" if user else "No match"
        color = (0, 255, 0) if user else (0, 0, 255)
        cv2.putText(disp, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.imshow("Live Recognition", disp)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
    cv2.destroyWindow("Live Recognition")

def main():
    parser = argparse.ArgumentParser(description="Live face enrollment and recognition (no directories).")
    parser.add_argument("--db", default="data/test_users.db", help="Path to database file.")
    parser.add_argument("--user-id", type=int, default=1, help="User ID to register.")
    parser.add_argument("--ear-threshold", type=float, default=0.22, help="EAR threshold for user.")
    parser.add_argument("--min-consistent-frames", type=int, default=6, help="Frames required for recognition consensus.")
    parser.add_argument("--num-enroll-frames", type=int, default=5, help="Frames to capture for enrollment.")
    args = parser.parse_args()

    cam = Camera()
    if not cam.ready:
        print("[ERROR] Camera not ready.")
        return

    os.makedirs(os.path.dirname(args.db) or ".", exist_ok=True)
    user_manager = UserManager(
        database_file=args.db,
        recognition_threshold=0.5,
        multi_frame_validation=True,
        min_consistent_frames=args.min_consistent_frames,
        min_face_confidence=0.95,
        input_color="RGB",
    )

    user = enroll_user(cam, user_manager, args.user_id, args.ear_threshold, args.num_enroll_frames)
    if user:
        print(f"[OK] Registered User ID={user.user_id}")
        live_recognition(cam, user_manager)
    else:
        print("[ERROR] Registration failed.")

    cam.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
import cv2
import numpy as np

class Visualizer:
    """
    A class dedicated to handling all drawing and visualization operations on the
    video frame. This centralizes display logic and keeps the main loop clean.
    """
    def __init__(self):
        # Define common colors in RGB format (for RGB frames)
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        
        # ✅ RGB colors (cv2 drawing works on any color space - just match the frame)
        self.COLOR_YELLOW = (255, 255, 0)
        self.COLOR_GREEN = (0, 255, 0)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_ORANGE = (255, 165, 0)
        self.COLOR_CYAN = (0, 255, 255)
        self.COLOR_RED = (255, 0, 0)

    def draw_landmarks(self, image: np.ndarray, coords: dict):
        """
        Draws circles on the eye and mouth landmarks for visual feedback.
        """
        for key in ["left_eye", "right_eye", "mouth"]:
            for point in coords.get(key, []):
                cv2.circle(image, point, 1, self.COLOR_GREEN, -1)

    def draw_no_user_text(self, image: np.ndarray):
        """
        Displays a centered message when the system is waiting for a user.
        """
        h, w, _ = image.shape
        text = "Looking for user..."
        
        (text_width, text_height), _ = cv2.getTextSize(text, self.FONT, 0.9, 2)
        pos_x = (w - text_width) // 2
        pos_y = (h + text_height) // 2
        
        cv2.putText(image, text, (pos_x, pos_y), self.FONT, 0.9, self.COLOR_YELLOW, 2)

    def draw_face_not_detected(self, image: np.ndarray, user_name: str):
        """
        Displays a status message when the tracked user's face is lost from the frame.
        """
        status_text = "STATUS: FACE NOT DETECTED"
        user_text = f"TRACKING: {user_name}"
        cv2.putText(image, status_text, (10, 30), self.FONT, 0.7, self.COLOR_ORANGE, 2)
        cv2.putText(image, user_text, (10, 60), self.FONT, 0.7, self.COLOR_WHITE, 2)

    def draw_detection_hud(self, image: np.ndarray, user_name: str, status: str, color: tuple,
                           fps: float, ear: float, mar: float, blink_count: int, 
                           mouth_expression: str = "NEUTRAL", pose: tuple = None):
        """
        Draws the main HUD overlay with user info, status, metrics, and head pose.
        """
        h, w, _ = image.shape
        
        # --- TOP LEFT BLOCK (Status and User) ---
        name_text = f"User: {user_name}"
        status_text = f"Status: {status}"
        
        cv2.putText(image, name_text, (10, 30), self.FONT, 0.9, self.COLOR_CYAN, 2)
        cv2.putText(image, status_text, (10, 70), self.FONT, 0.9, color, 2)
        
        # --- MIDDLE LEFT BLOCK (Head Pose and Expression) ---
        y_pos = 120
        expression_text = f"Expr: {mouth_expression}"
        cv2.putText(image, expression_text, (10, y_pos), self.FONT, 0.7, self.COLOR_YELLOW, 2)

        if pose:
            pitch, yaw, roll = pose
            y_pos += 30
            # Head Pose Text Block (starts lower at Y=150)
            cv2.putText(image, f"Pitch: {int(pitch)}", (10, y_pos), self.FONT, 0.6, self.COLOR_WHITE, 1)
            y_pos += 25
            cv2.putText(image, f"Yaw:   {int(yaw)}",   (10, y_pos), self.FONT, 0.6, self.COLOR_WHITE, 1)
            y_pos += 25
            cv2.putText(image, f"Roll:  {int(roll)}",  (10, y_pos), self.FONT, 0.6, self.COLOR_WHITE, 1)
        
        # --- BOTTOM LEFT BLOCK (Biometric Data) ---
        # Ensure values are valid before formatting
        ear_text = f"EAR: {ear:.3f}" if ear is not None else "EAR: --"
        mar_text = f"MAR: {mar:.3f}" if mar is not None else "MAR: --"
        blink_text = f"Blinks: {blink_count}"
        
        cv2.putText(image, ear_text, (10, h - 70), self.FONT, 0.7, self.COLOR_YELLOW, 2)
        cv2.putText(image, mar_text, (10, h - 40), self.FONT, 0.7, self.COLOR_YELLOW, 2)
        cv2.putText(image, blink_text, (10, h - 10), self.FONT, 0.7, self.COLOR_WHITE, 2)
        
        # --- TOP RIGHT BLOCK (Performance metrics) ---
        fps_text = f"FPS: {fps:.2f}"
        (text_width, _), _ = cv2.getTextSize(fps_text, self.FONT, 0.7, 2)
        cv2.putText(image, fps_text, (w - text_width - 10, 30), self.FONT, 0.7, self.COLOR_GREEN, 2)
        
    def draw_no_face_text(self, display):
        import cv2
        h, w = display.shape[:2]
        cv2.putText(display, "NO FACE DETECTED", (w//2 - 100, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
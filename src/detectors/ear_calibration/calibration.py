import time
from typing import Union, Tuple
import cv2



def calibrate(calibrator) -> Union[None, float, Tuple[str, object]]:
        """
        Perform EAR calibration with optimized processing.
        
        Returns:
            - float: Calibrated EAR threshold
            - ('user_swap', UserProfile): If existing user detected
            - None: If calibration failed/cancelled
        """
        print(f"\n--- EAR Calibration: Look at the camera for {calibrator.CALIBRATION_DURATION_S} seconds. ---")
        time.sleep(1.0)

        # Reset buffer
        calibrator._ear_count = 0
        
        start_time = time.time()
        face_lost_start_time = None
        frame_count = 0
        
        # Cache variables to reduce attribute lookups
        calibration_duration = calibrator.CALIBRATION_DURATION_S
        face_lost_timeout = calibrator.FACE_LOST_TIMEOUT_S
        user_check_interval = calibrator.USER_CHECK_INTERVAL
        display_update_interval = calibrator.DISPLAY_UPDATE_INTERVAL

        # Start background user recognition thread
        calibrator.manage_user_check_thread(start=True)

        try:
            while True:
                elapsed_time = time.time() - start_time
                if elapsed_time >= calibration_duration:
                    break
                
                # ✅ camera.capture_frame() or camera.read() always returns RGB
                frame = calibrator.camera.read()
                if frame is None:
                    continue

                frame_count += 1

                # Flip frame for mirrored display (only for visual feedback)
                feedback_frame = cv2.flip(frame, 1)

                # Check for existing users less frequently
                if frame_count % user_check_interval == 0:
                    if calibrator._user_check_queue.empty():
                        calibrator._user_check_queue.put(feedback_frame.copy())
                
                # Check result from background thread
                if calibrator._user_check_result is not None:
                    print(f"Recognized user '{getattr(calibrator._user_check_result, 'full_name', calibrator._user_check_result.user_id)}' detected.")
                    return ('user_swap', calibrator._user_check_result)

                # ❌ REMOVE: No color conversion needed - camera.py already returns RGB
                # frame is already RGB from camera.py
                rgb_frame = feedback_frame

                # Process landmarks efficiently
                results = calibrator.face_mesh.process(rgb_frame)
                ear, status_msg = calibrator._process_landmarks_optimized(results, rgb_frame.shape)

                # Track face loss
                if status_msg == "Face not detected":
                    if face_lost_start_time is None:
                        face_lost_start_time = time.time()
                    elif time.time() - face_lost_start_time > face_lost_timeout:
                        print("Calibration failed: Face was not detected for too long.")
                        return None
                else:
                    face_lost_start_time = None

                # Update display less frequently
                if frame_count % display_update_interval == 0:
                    display_frame_bgr = cv2.cvtColor(feedback_frame, cv2.COLOR_RGB2BGR)
                    calibrator.feedback(display_frame_bgr, ear, elapsed_time, status_msg, calibrator._ear_count)

                # Check for ESC key
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("Calibration cancelled by user.")
                    return None

        finally:
            # Stop background thread
            calibrator.manage_user_check_thread(start=False)

        # Calculate robust average from collected samples
        return calibrator.average_ear()
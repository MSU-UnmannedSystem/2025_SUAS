import cv2
import numpy as np
import math

class VisionPipeline:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print("[VISION] Error: Could not open webcam.")

    def get_target_pixel(self):
        """Reads one frame and returns the (X, Y) pixel of the target, or (None, None) if not found."""
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Your exact color bounds
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        red_mask = mask1 + mask2

        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500: 
                continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * math.pi * (area / (perimeter * perimeter))

            if 0.7 < circularity < 1.2:
                M = cv2.moments(contour)
                if M["m00"] != 0: 
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    return cX, cY  # Return the exact pixel immediately

        return None, None # Return None if no bullseye is in this frame

    def release(self):
        self.cap.release()

import cv2
import numpy as np
import math
import threading

class VisionPipeline:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print("[VISION] Error: Could not open webcam.")
            
        # --- FRAME BUFFER FIX ---
        self.ret = False
        self.frame = None
        self.running = True
        
        # Start background thread to continuously flush the hardware buffer
        self.thread = threading.Thread(target=self._update_frame, daemon=True)
        self.thread.start()
        print("[VISION] Daemon thread started. Buffer flushing active.")

    def _update_frame(self):
        """Runs continuously in the background to grab the freshest frame."""
        while self.running:
            if self.cap.isOpened():
                self.ret, self.frame = self.cap.read()

    def get_target_pixel(self):
        """Reads the freshest frame and isolates Black & White Bullseyes."""
        if not self.ret or self.frame is None:
            return None, None

        # Safely copy the absolute newest frame from the background thread
        frame = self.frame.copy()

        # --- COLOR SPACE FIX: Convert to Grayscale & B&W Binary ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold to isolate stark black and white contrast targets
        _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)
        
        # Morphological cleanup to remove grass/asphalt noise
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500: 
                continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            # Check for a circle (Bullseye)
            circularity = 4 * math.pi * (area / (perimeter * perimeter))

            if 0.7 < circularity < 1.2:
                M = cv2.moments(contour)
                if M["m00"] != 0: 
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    return cX, cY  # Return exact pixel immediately

        return None, None

    def release(self):
        """Safely kill the thread and release the camera hardware."""
        self.running = False
        self.thread.join()
        self.cap.release()
        print("[VISION] Hardware released cleanly.")
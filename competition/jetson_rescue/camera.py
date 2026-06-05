import cv2
import numpy as np
import math
import threading

class VisionPipeline:
    def __init__(self, camera_index=0, target_type='red_bullseye'):
        self.cap = cv2.VideoCapture(camera_index)
        self.target_type = target_type
        
        if not self.cap.isOpened():
            print("[VISION] Error: Could not open webcam.")
            
        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        
        # Start the background daemon thread
        self.thread = threading.Thread(target=self._update_frame, daemon=True)
        self.thread.start()
        print(f"[VISION] Daemon thread active. Mode: {self.target_type.upper()}")

    def _update_frame(self):
        """Continuously flushes the hardware buffer to grab the freshest frame."""
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                with self.lock:
                    self.ret = ret
                    self.frame = frame

    def get_target_pixel(self):
        """Safely pulls the newest frame and routes to the correct math engine."""
        with self.lock:
            if not self.ret or self.frame is None:
                return None, None
            frame = self.frame.copy()

        # The Router
        if self.target_type == 'red_bullseye':
            return self._find_red_bullseye(frame)
        elif self.target_type == 'bw_square':
            return self._find_bw_square(frame)
            
        return None, None

    def _find_red_bullseye(self, frame):
        """Mission: Package Drop & Package Delivery"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red hue wraps around the HSV spectrum
        mask1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
        red_mask = mask1 + mask2

        # Cleanup noise
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500: continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0: continue
            
            circularity = 4 * math.pi * (area / (perimeter * perimeter))
            if 0.7 < circularity < 1.2:
                M = cv2.moments(contour)
                if M["m00"] != 0: 
                    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        return None, None

    def _find_bw_square(self, frame):
        """Mission: Target Localization (GCPs)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000: # Slightly larger area filter to avoid ground noise
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                
                # Check for exactly 4 vertices
                if len(approx) == 4:
                    M = cv2.moments(contour)
                    if M["m00"] != 0: 
                        return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        return None, None

    def release(self):
        """Safely kills the thread and releases the hardware."""
        self.running = False
        self.thread.join()
        self.cap.release()
        print("[VISION] Hardware released cleanly.")
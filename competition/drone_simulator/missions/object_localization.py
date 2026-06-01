from pymavlink import mavutil
import time
import cv2
import numpy as np
from missions.mission_base import MissionBase
from jetson_rescue.geolocation import TargetGeolocator
import csv

class ObjectLocalizationMission(MissionBase):
    def __init__(self, master, config, desk_test=False):
        super().__init__(master, config)
        self.desk_test = desk_test
        self.geolocator = TargetGeolocator(res_w=1920, res_h=1080, hfov_deg=65.0, vfov_deg=39.4)
        self.cap = cv2.VideoCapture(0)
        
        # Setup CSV to hand to the judges
        self.csv_filename = f"target_locations_{int(time.time())}.csv"
        with open(self.csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Target_ID", "Latitude", "Longitude"])

    def arm_and_takeoff(self):
        """Overrides base class to allow for props-off desk testing"""
        if self.desk_test:
            print("[LOCALIZATION] DESK TEST MODE: Skipping physical takeoff wait.")
            self.set_expected_mode("AUTO")
            from vehicle.modes import arm_drone
            arm_drone(self.master)
            time.sleep(2)
        else:
            super().arm_and_takeoff()

    def find_squares(self, frame):
        """Lightweight OpenCV shape detector looking for black/white squares"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Threshold for high contrast black/white targets
        _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000: # Filter out tiny noise
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                
                # If the shape has 4 vertices, it's a rectangle/square
                if len(approx) == 4:
                    # Get center pixel
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        return cx, cy
        return None, None

    def execute(self):
        print("[LOCALIZATION] Entering Passive Monitoring. Drone is flying AUTO path.")
        if not self.desk_test:
            from vehicle.modes import change_mode
            change_mode(self.master, "AUTO")
        self.set_expected_mode("AUTO")

    def monitor(self):
        """
        Overrides the base class monitor. 
        Checks waypoints AND runs vision processing simultaneously.
        """
        # 1. Run the normal base class waypoint check
        super().monitor()

        # 2. Grab a frame and look for a square
        ret, frame = self.cap.read()
        if ret:
            pixel_x, pixel_y = self.find_squares(frame)
            
            if pixel_x is not None:
                # Target Spotted! Grab live telemetry.
                pos_msg = self.master.recv_match(type="GLOBAL_POSITION_INT", blocking=False)
                att_msg = self.master.recv_match(type="ATTITUDE", blocking=False)
                
                if pos_msg and att_msg:
                    current_lat = pos_msg.lat / 1e7
                    current_lon = pos_msg.lon / 1e7
                    alt = pos_msg.relative_alt / 1000.0  
                    
                    # Rulebook enforcement warning (just prints to terminal for pilot)
                    if alt < 7.62:
                        print(f"WARNING: Altitude is {alt:.1f}m! Climb above 25ft to avoid penalty!")

                    # Calculate target GPS
                    t_lat, t_lon, _, _ = self.geolocator.calculate_target_coordinates(
                        pixel_x, pixel_y, current_lat, current_lon, alt, 
                        att_msg.pitch, att_msg.roll, att_msg.yaw
                    )
                    
                    print(f"[LOCALIZATION] Square Found! Target GPS: {t_lat:.6f}, {t_lon:.6f}")
                    
                    # Save to CSV for the judges
                    with open(self.csv_filename, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(["Square_Target", t_lat, t_lon])
                    
                    # Sleep briefly so we don't log the exact same target 30 times a second
                    time.sleep(2.0)
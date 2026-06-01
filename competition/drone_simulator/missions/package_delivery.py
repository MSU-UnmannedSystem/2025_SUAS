from pymavlink import mavutil
import time
import math
from missions.mission_base import MissionBase
from vehicle.modes import change_mode

from jetson_rescue.geolocation import TargetGeolocator
from inference.camera import VisionPipeline
from listener.Logger import TelemetryLogger

class PackageDeliveryMission(MissionBase):
    def __init__(self, master, config, desk_test=False):
        super().__init__(master, config)
        self.desk_test = desk_test
        self.geolocator = TargetGeolocator(res_w=1920, res_h=1080, hfov_deg=65.0, vfov_deg=39.4)
        self.vision = VisionPipeline(camera_index=0)
        self.logger = TelemetryLogger()
        self.drop_altitude = 1.5 # 1.5 meters (5 feet) to avoid the >6ft 0-point penalty

    def arm_and_takeoff(self):
        """Overrides base class to allow for props-off desk testing"""
        if self.desk_test:
            print("[DELIVERY] DESK TEST MODE: Skipping physical takeoff wait.")
            change_mode(self.master, "GUIDED")
            self.set_expected_mode("GUIDED")
            from vehicle.modes import arm_drone
            arm_drone(self.master)
            print("[DELIVERY] Motors armed. Proceeding directly to vision sequence.")
            time.sleep(2)
        else:
            super().arm_and_takeoff()

    def execute(self):
        print("[DELIVERY] Taking control. Switching to GUIDED mode.")
        change_mode(self.master, "GUIDED")
        self.set_expected_mode("GUIDED")
        
        target_locked = False
        target_lat_int = 0
        target_lon_int = 0
        tolerance_meters = 2.0 # Tighter tolerance for the heavy cube

        while self.active:
            pixel_x, pixel_y = self.vision.get_target_pixel()

            if pixel_x is not None and not target_locked:
                print(f"[DELIVERY] Bullseye acquired at pixel ({pixel_x}, {pixel_y})!")
                
                pos_msg = self.master.recv_match(type="GLOBAL_POSITION_INT", blocking=True)
                att_msg = self.master.recv_match(type="ATTITUDE", blocking=True)

                current_lat = pos_msg.lat / 1e7
                current_lon = pos_msg.lon / 1e7
                alt = pos_msg.relative_alt / 1000.0  
                pitch, roll, yaw = att_msg.pitch, att_msg.roll, att_msg.yaw

                t_lat, t_lon, _, _ = self.geolocator.calculate_target_coordinates(
                    pixel_x, pixel_y, current_lat, current_lon, alt, pitch, roll, yaw
                )

                target_lat_int = int(t_lat * 1e7)
                target_lon_int = int(t_lon * 1e7)
                target_locked = True
                
                print(f"[DELIVERY] Math Engine: Flying to Lat {t_lat:.6f}, Lon {t_lon:.6f}")
                self.logger.log_data(current_lat, current_lon, alt, pitch, roll, yaw, t_lat, t_lon)

                # Command drone to fly over target at the original TAKEOFF altitude
                self.master.mav.set_position_target_global_int_send(
                    0, self.master.target_system, self.master.target_component,
                    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                    int(0b110111111000), target_lat_int, target_lon_int, self.takeoff_alt,
                    0, 0, 0, 0, 0, 0, 0, 0)

            if target_locked:
                pos_msg = self.master.recv_match(type="GLOBAL_POSITION_INT", blocking=False)
                if pos_msg:
                    current_alt = pos_msg.relative_alt / 1000.0
                    lat_dif = (abs(target_lat_int - pos_msg.lat) / 1e7) * 111319
                    lon_dif = (abs(target_lon_int - pos_msg.lon) / 1e7) * 111319 * math.cos(math.radians(35))
                    distance = (lat_dif**2 + lon_dif**2)**(0.5)

                    print(f"[DELIVERY] Distance: {distance:.2f}m | Alt: {current_alt:.1f}m")

                    if distance <= tolerance_meters:
                        print("[DELIVERY] Target Reached! Descending to 1.5m (5ft) safe drop height...")
                        
                        # Command descent to 1.5 meters
                        self.master.mav.set_position_target_global_int_send(
                            0, self.master.target_system, self.master.target_component,
                            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                            int(0b110111111000), target_lat_int, target_lon_int, self.drop_altitude,
                            0, 0, 0, 0, 0, 0, 0, 0)

                        # Wait until altitude physically drops below 1.8 meters (6 feet)
                        while current_alt > 1.8:
                            wait_msg = self.master.recv_match(type="GLOBAL_POSITION_INT", blocking=True)
                            current_alt = wait_msg.relative_alt / 1000.0
                            print(f"[DELIVERY] Descending... Current Alt: {current_alt:.2f}m")
                            time.sleep(0.1)

                        print("[DELIVERY] Below 6ft penalty ceiling! Actuating Servo.")
                        
                        # --- SERVO ACTUATION COMMAND ---
                        servo_pin = 6 # <--- Change to Electrical's pin
                        pwm_open = 1900 
                        self.master.mav.command_long_send(
                            self.master.target_system, self.master.target_component,
                            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
                            0, servo_pin, pwm_open, 0, 0, 0, 0, 0
                        )
                        
                        print("[DELIVERY] Package Delivered. Climbing back to safe altitude.")
                        self.master.mav.set_position_target_global_int_send(
                            0, self.master.target_system, self.master.target_component,
                            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                            int(0b110111111000), target_lat_int, target_lon_int, self.takeoff_alt,
                            0, 0, 0, 0, 0, 0, 0, 0)
                        break

            self.update_heartbeat()
            self.check_safety()
            time.sleep(0.2) 

        self.vision.release()
        print("[DELIVERY] Handing back to Base.")
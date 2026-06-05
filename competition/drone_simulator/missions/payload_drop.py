from pymavlink import mavutil
import time
import math
from missions.mission_base import MissionBase
from vehicle.modes import change_mode

# Import your completed engines
from jetson_rescue.geolocation import TargetGeolocator
from inference.camera import VisionPipeline
from listener.Logger import TelemetryLogger

class PayloadDropMission(MissionBase):
    def __init__(self, master, config, desk_test=False):
        super().__init__(master, config)
        self.desk_test = desk_test
        
        # 1. Initialize math engine (Updated for ELP USB12MP02AF camera at 1080p)
        self.geolocator = TargetGeolocator(res_w=1920, res_h=1080, hfov_deg=65.0, vfov_deg=39.4)
        # 2. Initialize the vision pipeline
        self.vision = VisionPipeline(camera_index=0, target_type='red_bullseye')
        # 3. Initialize the logger 
        self.logger = TelemetryLogger()

    def arm_and_takeoff(self):
        """Overrides base class to allow for props-off desk testing"""
        if self.desk_test:
            print("[PAYLOAD DROP] DESK TEST MODE: Skipping physical takeoff wait.")
            change_mode(self.master, "GUIDED")
            self.set_expected_mode("GUIDED")
            
            from vehicle.modes import arm_drone
            arm_drone(self.master)
            
            print("[PAYLOAD DROP] Motors armed. Proceeding directly to vision sequence.")
            time.sleep(2)
        else:
            # If DESK_TEST is False, run the real takeoff sequence from MissionBase
            super().arm_and_takeoff()

    def execute(self):
        print("[PAYLOAD DROP] Taking control. Switching to GUIDED mode.")
        change_mode(self.master, "GUIDED")
        self.set_expected_mode("GUIDED")
        
        target_locked = False
        target_lat_int = 0
        target_lon_int = 0
        tolerance_meters = 5

        # The Main Hunt Loop
        while self.active:
            # 1. Look for the target
            pixel_x, pixel_y = self.vision.get_target_pixel()

            if pixel_x is not None and not target_locked:
                print(f"[PAYLOAD DROP] Bullseye acquired at pixel ({pixel_x}, {pixel_y})!")
                
                # 2. Grab live telemetry from ArduPilot
                pos_msg = self.master.recv_match(type="GLOBAL_POSITION_INT", blocking=True)
                att_msg = self.master.recv_match(type="ATTITUDE", blocking=True)

                current_lat = pos_msg.lat / 1e7
                current_lon = pos_msg.lon / 1e7
                alt = pos_msg.relative_alt / 1000.0  
                pitch, roll, yaw = att_msg.pitch, att_msg.roll, att_msg.yaw

                # 3. Calculate Geographic Coordinates
                t_lat, t_lon, _, _ = self.geolocator.calculate_target_coordinates(
                    pixel_x, pixel_y, current_lat, current_lon, alt, pitch, roll, yaw
                )

                target_lat_int = int(t_lat * 1e7)
                target_lon_int = int(t_lon * 1e7)
                target_locked = True
                
                print(f"[PAYLOAD DROP] Math Engine Output: Flying to Lat {t_lat:.6f}, Lon {t_lon:.6f}")

                # 4. Save the data to memory 
                self.logger.log_data(current_lat, current_lon, alt, pitch, roll, yaw, t_lat, t_lon)

                # 5. Command the drone to fly to the calculated target
                self.master.mav.set_position_target_global_int_send(
                    0, self.master.target_system, self.master.target_component,
                    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                    int(0b110111111000), target_lat_int, target_lon_int, self.takeoff_alt,
                    0, 0, 0, 0, 0, 0, 0, 0)

            # 6. If we have a target, check how close we are
            if target_locked:
                pos_msg = self.master.recv_match(type="GLOBAL_POSITION_INT", blocking=False)
                if pos_msg:
                    latitude_dif_meters = (abs(target_lat_int - pos_msg.lat) / 1e7) * 111319
                    longitude_dif_meters = (abs(target_lon_int - pos_msg.lon) / 1e7) * 111319 * math.cos(math.radians(35))
                    distance = (latitude_dif_meters**2 + longitude_dif_meters**2)**(0.5)

                    print(f"[PAYLOAD DROP] Distance to drop zone: {distance:.2f}m")

                    if distance <= tolerance_meters:
                        print("[PAYLOAD DROP] Target Reached! Initiating Drop Sequence.")
                        
                        # --- SERVO ACTUATION COMMAND ---
                        servo_pin = 6 # <--- IMPORTANT: Change this 9 to whatever pin Electrical uses!
                        pwm_open = 1900 
                        
                        self.master.mav.command_long_send(
                            self.master.target_system,
                            self.master.target_component,
                            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
                            0, servo_pin, pwm_open, 0, 0, 0, 0, 0
                        )
                        break

            self.update_heartbeat()
            self.check_safety()
            time.sleep(0.2) 

        self.vision.release()
        print("[PAYLOAD DROP] Drop complete. Handing back to Base.")
from pymavlink import mavutil
import time
import math
from missions.mission_base import MissionBase
from vehicle.modes import change_mode

# Import your completed engines
from geolocation import TargetGeolocator
from camera import VisionPipeline

class PayloadDropMission(MissionBase):
    def __init__(self, master, config):
        super().__init__(master, config)
        # 1. Initialize your math engine (adjust FOV to your exact webcam specs)
        self.geolocator = TargetGeolocator(res_w=1920, res_h=1080, hfov_deg=78.0, vfov_deg=43.0)
        # 2. Initialize the vision pipeline
        self.vision = VisionPipeline(camera_index=0)

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

                # 4. Command the drone to fly to the calculated target
                self.master.mav.set_position_target_global_int_send(
                    0, self.master.target_system, self.master.target_component,
                    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                    int(0b110111111000), target_lat_int, target_lon_int, self.takeoff_alt,
                    0, 0, 0, 0, 0, 0, 0, 0)

            # 5. If we have a target, check how close we are
            if target_locked:
                pos_msg = self.master.recv_match(type="GLOBAL_POSITION_INT", blocking=False)
                if pos_msg:
                    latitude_dif_meters = (abs(target_lat_int - pos_msg.lat) / 1e7) * 111319
                    longitude_dif_meters = (abs(target_lon_int - pos_msg.lon) / 1e7) * 111319 * math.cos(math.radians(35))
                    distance = (latitude_dif_meters**2 + longitude_dif_meters**2)**(0.5)

                    print(f"[PAYLOAD DROP] Distance to drop zone: {distance:.2f}m")

                    if distance <= tolerance_meters:
                        print("[PAYLOAD DROP] Target Reached! Initiating Drop Sequence.")
                        # TODO: Trigger the physical drop mechanism (e.g., servo PWM command)
                        break

            # Mandatory base class safety checks
            self.update_heartbeat()
            self.check_safety()
            time.sleep(0.2) # Run this loop 5 times a second

        # Cleanup
        self.vision.release()
        print("[PAYLOAD DROP] Drop complete. Handing back to Base.")

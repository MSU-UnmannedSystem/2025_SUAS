from pymavlink import mavutil
import time
from vehicle.modes import change_mode, arm_drone
from vehicle.upload import upload_waypoints


class MissionBase():

    def __init__(self, master, config):
        self.master = master
        self.config = config
        self.takeoff_alt = None
        self.waypoints = None
        self.active = False
        self.start_time = None
        self.time_limit = self.config.get("time_limit", 600)
        self.last_waypoint = None
        self.last_heartbeat = time.time()
        self.expected_mode = "AUTO"
        self.max_alt = self.config.get("max_alt", 120)
        self.min_alt = self.config.get("min_alt", 5)
        self.geofence = self.config.get("geofence", None)
        self.end_mode = self.config.get("end_mode", "RTL") 

    def run(self):
        """Main mission entry point"""
        try:
            #Setup mission
            self.setup()

            #Arm and takeoff
            self.arm_and_takeoff()

            #Execute the mission
            self.execute()

            #Keep track of time
            #Monitor the mission
            self.start_time = time.time()
            self.active = True

            while self.active:
                self.update_heartbeat()
                self.monitor()              # mission-specific
                self.check_time_limit()     # competition rule
                self.check_safety()         # geofence / link loss
                time.sleep(0.2)

            #terminate
            self.end_mission()
        except Exception as e:
            print(f"Mission failed: {e}")
            self.abort()


    def setup(self):
        """Prepare mission (abstract)"""
        if "takeoff_alt" not in self.config:
            raise ValueError("Missing takeoff_alt in mission config")
        
        self.takeoff_alt = self.config["takeoff_alt"]

        if self.takeoff_alt <= 0:
            raise ValueError("takeoff_alt must be > 0")
        
        if "waypoints" not in self.config:
            raise ValueError("Missing way points in mission config")

        self.waypoints = self.config["waypoints"]

        if not isinstance(self.waypoints, list) or len(self.waypoints) == 0:
            raise ValueError("Waypoints must be a non-empty list")
        
        upload_waypoints(self.master, self.waypoints)
        print("Waypoints uploaded")
        print(f"target altitude: {self.takeoff_alt}\nWaypoints: {self.waypoints}")
   
    def execute(self):
        """Start mission (abstract)"""



        print("Starting Mission")
        change_mode(self.master, "AUTO")
        self.set_expected_mode("AUTO")

        start = time.time()
        while True:
            hb = self.master.recv_match(type="HEARTBEAT", blocking=False)
            if hb and hb.custom_mode == self.master.mode_mapping()["AUTO"]:
                break

            if time.time() - start > 5:
                raise RuntimeError("Failed to enter AUTO mode")

    def monitor(self):
        """Mission-specific monitoring (abstract)"""
        
        msg = self.master.recv_match(type="MISSION_CURRENT", blocking=False)

        if not msg:
            return

        curr_waypoint = msg.seq
        final_waypoint = len(self.waypoints) - 1

        if curr_waypoint != self.last_waypoint:
            self.last_waypoint = curr_waypoint
            print(f"Waypoint progress: {curr_waypoint}/{final_waypoint}")

        if curr_waypoint >= final_waypoint:
            print("Mission Complete")
            self.active = False
        



    def arm_and_takeoff(self):
        """Standard arming & takeoff"""

        change_mode(self.master, "GUIDED")
        self.set_expected_mode("GUIDED")
        arm_drone(self.master)
        self.master.mav.command_long_send(self.master.target_system,
                                 self.master.target_component,
                                 mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                                 0,0,0,0,0,0,0, self.takeoff_alt)


        start = time.time()
        timeout = 30    # seconds

        while True:
            msg = self.master.recv_match(type="GLOBAL_POSITION_INT", blocking=False)

            if not msg:
                continue
            
            alt = msg.relative_alt / 1000.0
            print(f"Current Altitude: {round(alt, 2)} m")

            if alt >= self.takeoff_alt * 0.95:  #if current alt greater than takeoff alt with 5% buffer
                print("target altitude reached")
                break

            if time.time() - start >= timeout:
                #replace with self.abort eventually (once implemented)
                print("Takeoff timeout - aborting")
                raise RuntimeError("Takeoff timeout - aborting")
            
            time.sleep(0.1)


    def check_time_limit(self):
        """Enforce 10 minute rule"""
        if self.start_time is None:
            return
        
        if time.time() - self.start_time >= self.time_limit:
            print("Runtime exceeded")
            raise RuntimeError("Runtime exceeded")

    def abort(self):
        """Failsafe behavior (RTL/LAND)"""
        print("Aborting missing - RTL")
        self.end_mission()
    
    def check_safety(self):
        """"Safety check"""

        hb = self.master.recv_match(type="HEARTBEAT", blocking=False)

        if hb:
            self.last_heartbeat = time.time()
            current_mode = mavutil.mode_string_v10(hb)

            if current_mode != self.expected_mode:
                raise RuntimeError(f"Expected {self.expected_mode} recieved {current_mode}")
        
        msg = self.master.recv_match(type="GLOBAL_POSITION_INT", blocking=False)
        if msg:
            alt = msg.relative_alt/1000.0

            if alt < self.min_alt:
                raise RuntimeError("Altitude below minimum")
            elif alt > self.max_alt:
                raise RuntimeError("Altitude above maximum")
            
            if self.geofence:
                lat = msg.lat / 1e7
                lon = msg.lon / 1e7

                lat_min, lat_max, lon_min, lon_max = self.geofence
                if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
                    raise RuntimeError("Geofence violation")
        

    def end_mission(self):
        """Cleanup after mission"""

        print("Ending mission")

        if self.end_mode == "LAND":
            print("Landing")
            change_mode(self.master, self.end_mode)
            self.set_expected_mode(self.end_mode)
        else:
            print("Returning to launch")
            change_mode(self.master, self.end_mode)
            self.set_expected_mode(self.end_mode)
        
        self.active = False
    
    def set_expected_mode(self, mode):
        self.expected_mode = mode

    def update_heartbeat(self):
        hb = self.master.recv_match(type="HEARTBEAT", blocking=False)

        if hb:
            self.last_heartbeat= time.time()
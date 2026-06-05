from pymavlink import mavutil
import time
from missions.mission_base import MissionBase

class CircuitTimeTrialMission(MissionBase):
    def __init__(self, master, config, desk_test=False):
        super().__init__(master, config)
        self.desk_test = desk_test
        self.trial_start_time = None
        self.trial_end_time = None

    def arm_and_takeoff(self):
        if self.desk_test:
            print("[TIME TRIAL] DESK TEST MODE.")
            self.set_expected_mode("AUTO")
            from vehicle.modes import arm_drone
            arm_drone(self.master)
            time.sleep(2)
        else:
            super().arm_and_takeoff()

    def execute(self):
        print("[TIME TRIAL] Switching to AUTO. Pushing for maximum speed.")
        if not self.desk_test:
            from vehicle.modes import change_mode
            change_mode(self.master, "AUTO")
        self.set_expected_mode("AUTO")
        self.trial_start_time = time.time()

    def monitor(self):
        """Monitor MAVLink for waypoint progression."""
        msg = self.master.recv_match(type="MISSION_ITEM_REACHED", blocking=False)

        if not msg:
            return

        reached = msg.seq
        final_waypoint = len(self.waypoints) - 1

        print(f"[TIME TRIAL] Waypoint reached: {reached}/{final_waypoint}")

        if reached >= final_waypoint:
            self.trial_end_time = time.time()
            elapsed = self.trial_end_time - self.trial_start_time
            print(f"========================================")
            print(f"[TIME TRIAL] COMPLETE! Circuit Time: {elapsed:.2f} seconds")
            print(f"========================================")
            self.active = False
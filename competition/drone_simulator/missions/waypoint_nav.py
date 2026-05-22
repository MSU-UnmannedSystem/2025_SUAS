from pymavlink import mavutil
from missions.mission_base import MissionBase

class WaypointNavigation(MissionBase):
    """Navigate through a list of waypoints"""

    def __init__(self, master, config):
        super().__init__(master, config)
    
    def setup(self):
        """setup the waypoint mission"""
        print("waypoint mission setup")
        return super().setup()
    
    def execute(self):
        """Execute the waypoint mission"""
        print("Executing waypoint mission")
        return super().execute()
    
    def monitor(self):
        """Monitors mission progress"""
        return super().monitor()
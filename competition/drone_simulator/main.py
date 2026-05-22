from missions.waypoint_nav import WaypointNavigation
from vehicle.connection import connect_vehicle
from config.mission_params import parameters

master = connect_vehicle("udp:127.0.0.1:14553")

mission = WaypointNavigation(master, parameters())
mission.run()
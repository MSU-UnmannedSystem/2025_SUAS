from pymavlink import mavutil

def upload_waypoints(master, waypoints):
    #clears any existing mission
    master.mav.mission_clear_all_send(master.target_system, master.target_component)

    #Starts a new mission
    ack = master.recv_match(type='MISSION_ACK', blocking=True, timeout=3)

    if not ack:
        raise RuntimeError("Mission rejected")
    #Sends mission count (num of waypoints)
    master.mav.mission_count_send(master.target_system, master.target_component, len(waypoints))

    #Assigns waypoint directions in a sequential order
    for lat, lon, alt in waypoints:
        req = master.recv_match(type=['MISSION_REQUEST', 'MISSION_REQUEST_INT'], blocking=True, timeout=5)

        if not req:
            raise RuntimeError("Mission upload failed: no request")
        
        master.mav.mission_item_int_send(
            master.target_system, 
            master.target_component, 
            req.seq,  # Use the sequence from the request
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, 
            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 
            0,  # current (0 = not current)
            1,  # autocontinue
            0, 0, 0, 0,  # param1-4
            int(lat * 1e7),  # lat
            int(lon * 1e7),   # lon
            alt) # alt
    
    ack = master.recv_match(type="MISSION_ACK", blocking=True, timeout=5)

    if not ack:
        raise RuntimeError("Mission upload failed: no mission ack")
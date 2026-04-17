from pymavlink import mavutil

def upload_waypoints(master, waypoints):

    print(f"[UPLOAD] Clearing existing mission ({len(waypoints)} waypoints to upload)")
 
    # Step 1: Clear existing mission
    master.mav.mission_clear_all_send(
        master.target_system,
        master.target_component
    )
 
    # Drain any ACK from CLEAR_ALL — non-blocking, best-effort.
    # We don't hard-fail here because some FC firmware versions don't ACK this.
    clear_ack = master.recv_match(type="MISSION_ACK", blocking=True, timeout=2)
    if clear_ack:
        print(f"[UPLOAD] Mission cleared (ACK type={clear_ack.type})")
    else:
        print("[UPLOAD] No ACK for MISSION_CLEAR_ALL — continuing (normal on some firmware)")
 
    # Step 2: Send mission count
    master.mav.mission_count_send(
        master.target_system,
        master.target_component,
        len(waypoints)
    )
 
    # Step 3: Upload each waypoint on request
    for i, (lat, lon, alt) in enumerate(waypoints):
        req = master.recv_match(
            type=["MISSION_REQUEST", "MISSION_REQUEST_INT"],
            blocking=True,
            timeout=5
        )
 
        if not req:
            raise RuntimeError(
                f"[UPLOAD] Timeout waiting for MISSION_REQUEST (waypoint {i}). "
                f"Check FC connection and baud rate."
            )
 
        if req.seq != i:
            raise RuntimeError(
                f"[UPLOAD] FC requested waypoint {req.seq}, expected {i}. "
                f"Upload sequence broken."
            )
 
        master.mav.mission_item_int_send(
            master.target_system,
            master.target_component,
            req.seq,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            0,              # current: 0 = not current waypoint
            1,              # autocontinue
            0, 0, 0, 0,     # param1-4 (hold time, acceptance radius, pass-by, yaw)
            int(lat * 1e7),
            int(lon * 1e7),
            alt
        )
 
        print(f"[UPLOAD] Sent waypoint {i+1}/{len(waypoints)}: "
              f"({lat:.5f}, {lon:.5f}, {alt}m)")
 
    # Step 4: Wait for final MISSION_ACK confirming all waypoints accepted
    final_ack = master.recv_match(type="MISSION_ACK", blocking=True, timeout=5)
 
    if not final_ack:
        raise RuntimeError(
            "[UPLOAD] No final MISSION_ACK after uploading all waypoints. "
            "Mission may not be stored on FC."
        )
 
    if final_ack.type != mavutil.mavlink.MAV_MISSION_ACCEPTED:
        raise RuntimeError(
            f"[UPLOAD] Mission rejected by FC (ACK type={final_ack.type}). "
            f"Check waypoint validity."
        )
 
    print(f"[UPLOAD] Mission accepted by FC — {len(waypoints)} waypoints stored")
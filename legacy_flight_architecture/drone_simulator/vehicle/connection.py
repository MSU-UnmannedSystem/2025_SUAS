from pymavlink import mavutil

def connect_vehicle(port, baud_rate=115200):
    master = mavutil.mavlink_connection(port, baud=baud_rate)
    print("Waiting for heartbeat...")
    master.wait_heartbeat()
    print(f"Connected to system {master.target_system}, component {master.target_component}")
    return master
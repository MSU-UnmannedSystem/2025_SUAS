from pymavlink import mavutil


def connect_vehicle(port):
    master = mavutil.mavlink_connection(port)
    print("Waiting for heartbeat...")
    master.wait_heartbeat()
    print(f"Connected to system {master.target_system}, component {master.target_component}")
    return master
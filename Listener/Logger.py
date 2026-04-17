"""
logger.py
---------
Drone telemetry CSV logger for the drop system.
Connects to the Cube Orange via pymavlink and logs:
  - Timestamp
  - Drone GPS (Lat, Lon, Alt in meters)
  - Attitude (Pitch, Roll, Yaw in degrees)
  - Target GPS (Lat, Lon) — placeholder zeros until vision system provides data

Usage:
    python logger.py

    By default connects over USB (adjust PORT and BAUD below for TELEM2 serial wire).
    
    
    edits needed 
    set PORT to whatever COM port the Cube shows up, I am fairly certain that it is COM4 or COM5
    
    
"""

import csv
import math
import os
import time
from datetime import datetime

from pymavlink import mavutil

# CONFIG — change these to match your setup

# USB connection (for bench testing):
PORT = "COM5"        # Windows: "COM4", "COM5" etc. | Linux: "/dev/ttyACM0"
BAUD = 115200        # USB baud rate

# TELEM2 serial wire (swap these in when connecting to Jetson):
# PORT = "/dev/ttyTHS0"   # Jetson serial port
# BAUD = 921600           # Must match SERIAL2_BAUD set in Mission Planner

CSV_FILE = "telemetry_log.csv"

# how often a row written ->  0.1 = 10 rows per second.
LOG_INTERVAL = 0.1



CSV_HEADERS = [
    "timestamp",
    "drone_lat",
    "drone_lon",
    "drone_alt_m",
    "pitch_deg",
    "roll_deg",
    "yaw_deg",
    "target_lat",
    "target_lon",
]


def init_csv(filepath: str) -> None:
    """Create the CSV file and write headers if it doesn't already exist."""
    file_exists = os.path.isfile(filepath)
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if not file_exists:
            writer.writeheader()
            print(f"[logger] Created new log file: {filepath}")
        else:
            print(f"[logger] Appending to existing log file: {filepath}")


def append_row(filepath: str, row: dict) -> None:
    """Append a single data row to the CSV."""
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writerow(row)


# ─────────────────────────────────────────────
# MAVLINK CONNECTION
# ─────────────────────────────────────────────

def connect(port: str, baud: int) -> mavutil.mavfile:
    """Connect to the Cube and wait for a heartbeat."""
    print(f"[logger] Connecting to {port} at {baud} baud...")
    connection = mavutil.mavlink_connection(port, baud=baud)
    print("[logger] Waiting for heartbeat from Cube...")
    connection.wait_heartbeat()
    print(f"[logger] Heartbeat received! System ID: {connection.target_system}")
    return connection


def request_message_stream(connection: mavutil.mavfile) -> None:
    """
    Ask the Cube to stream GLOBAL_POSITION_INT and ATTITUDE at 10Hz.
    This is a backup in case SR2_POSITION wasn't set in Mission Planner.
    """
    connection.mav.request_data_stream_send(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_POSITION,
        10,  # 10 Hz
        1,   # start streaming
    )
    connection.mav.request_data_stream_send(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,
        10,  # 10 Hz — EXTRA1 includes ATTITUDE
        1,
    )
    print("[logger] Requested POSITION and ATTITUDE streams at 10Hz.")



def run():
    # csv
    init_csv(CSV_FILE)

    connection = connect(PORT, BAUD)

    request_message_stream(connection)

    drone_lat    = 0.0
    drone_lon    = 0.0
    drone_alt_m  = 0.0
    pitch_deg    = 0.0
    roll_deg     = 0.0
    yaw_deg      = 0.0
    target_lat   = 0.0   # placeholder — vision system will populate this later
    target_lon   = 0.0   # placeholder

    print(f"[logger] Logging to {CSV_FILE} every {LOG_INTERVAL}s. Press Ctrl+C to stop.\n")

    last_log_time = time.time()

    try:
        while True:
            msg = connection.recv_match(blocking=False)

            if msg is not None:
                msg_type = msg.get_type()

                if msg_type == "GLOBAL_POSITION_INT":
                    # lat/lon arrive as degrees * 1E7 — divide to get real degrees
                    drone_lat   = msg.lat / 1e7
                    drone_lon   = msg.lon / 1e7
                    # relative_alt is altitude above home in millimeters
                    drone_alt_m = msg.relative_alt / 1000.0

                elif msg_type == "ATTITUDE":
                    # pitch/roll/yaw arrive in radians — convert to degrees
                    pitch_deg = math.degrees(msg.pitch)
                    roll_deg  = math.degrees(msg.roll)
                    yaw_deg   = math.degrees(msg.yaw)

            # Write a row at the configured interval
            now = time.time()
            if now - last_log_time >= LOG_INTERVAL:
                row = {
                    "timestamp":    datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "drone_lat":    round(drone_lat, 7),
                    "drone_lon":    round(drone_lon, 7),
                    "drone_alt_m":  round(drone_alt_m, 3),
                    "pitch_deg":    round(pitch_deg, 4),
                    "roll_deg":     round(roll_deg, 4),
                    "yaw_deg":      round(yaw_deg, 4),
                    "target_lat":   target_lat,
                    "target_lon":   target_lon,
                }
                append_row(CSV_FILE, row)

                # prints live status
                print(
                    f"[{row['timestamp']}] "
                    f"Alt: {row['drone_alt_m']}m | "
                    f"Lat: {row['drone_lat']} Lon: {row['drone_lon']} | "
                    f"Pitch: {row['pitch_deg']}° Roll: {row['roll_deg']}° Yaw: {row['yaw_deg']}°"
                )

                last_log_time = now

    except KeyboardInterrupt:
        print(f"\n[logger] Stopped. Data saved to {CSV_FILE}")



if __name__ == "__main__":
    run()

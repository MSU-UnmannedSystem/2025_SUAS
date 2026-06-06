from pymavlink import mavutil
import math
import csv
import os
import sys

def analyze_blackbox(bin_filepath, scoring_waypoints, altitudes_in_feet):
    print(f"=== C-UASC 2026: 3D WAYPOINT LOG ANALYZER ===")
    print(f"Targeting DataFlash Log: {bin_filepath}")
    
    if not os.path.exists(bin_filepath):
        print(f"\n[ERROR] File Not Found: {bin_filepath}")
        print("Ensure you downloaded the .bin file from Mission Planner to your Downloads folder.")
        sys.exit(1)

    try:
        mlog = mavutil.mavlink_connection(bin_filepath)
    except Exception as e:
        print(f"Failed to open log: {e}")
        sys.exit(1)

    # Convert provided altitudes to meters if they were entered in feet
    converted_waypoints = []
    for lat, lon, alt in scoring_waypoints:
        final_alt = alt * 0.3048 if altitudes_in_feet else alt
        converted_waypoints.append((lat, lon, final_alt))

    # Initialize trackers for the tightest 3D approach to each waypoint
    closest_approaches = {i: 9999.0 for i in range(len(converted_waypoints))}
    
    print(f"Scanning GPS & Barometer telemetry against {len(converted_waypoints)} 3D scoring targets...")
    
    gps_count = 0
    while True:
        # We use the 'POS' message because it natively logs RelHomeAlt (Altitude AGL in meters)
        msg = mlog.recv_match(type=['POS'], blocking=False)
        if msg is None:
            break 
        
        lat = msg.Lat
        lon = msg.Lng
        alt_agl = msg.RelHomeAlt 
        gps_count += 1

        for i, (t_lat, t_lon, t_alt_m) in enumerate(converted_waypoints):
            # 1. Calculate Horizontal Offset (Meters)
            lat_dif_m = (abs(t_lat - lat)) * 111319
            lon_dif_m = (abs(t_lon - lon)) * 111319 * math.cos(math.radians(35.05))
            
            # 2. Calculate Vertical Offset (Meters)
            alt_dif_m = abs(t_alt_m - alt_agl)

            # 3. Calculate True 3D Spherical Distance
            distance_3d = (lat_dif_m**2 + lon_dif_m**2 + alt_dif_m**2)**0.5

            # Lock in the tightest margin achieved across the entire flight
            if distance_3d < closest_approaches[i]:
                closest_approaches[i] = distance_3d

    print(f"Parsed {gps_count} positional telemetry points.")

    # --- OUTPUT RESULTS ---
    print("\n--- FINAL 3D SCORING RESULTS ---")
    total_penalty = 0.0
    for i in range(len(converted_waypoints)):
        dist = closest_approaches[i]
        
        # Rulebook: 10 points awarded if within 1m. Deductions for every meter over 1m.
        penalty = max(0, dist - 1.0)
        total_penalty += penalty
        
        print(f"Waypoint {i+1}: Closest 3D Approach = {dist:.2f}m | Penalty = -{penalty:.2f} pts")
    
    print(f"--------------------------------")
    print(f"Estimated Navigation Penalty: -{total_penalty:.2f} Points")
    
    # --- SAVE TO CSV ---
    csv_filename = bin_filepath.replace('.bin', '_scoring_report.csv')
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Waypoint_Index", "Target_Lat", "Target_Lon", "Target_Alt_Meters", "Closest_3D_Approach_M", "Penalty_Points"])
        for i in range(len(converted_waypoints)):
            t_lat, t_lon, t_alt_m = converted_waypoints[i]
            writer.writerow([i+1, t_lat, t_lon, round(t_alt_m, 2), round(closest_approaches[i], 2), round(max(0, closest_approaches[i] - 1.0), 2)])
            
    print(f"\nReport saved successfully: {csv_filename}")


if __name__ == "__main__":
    # ---------------------------------------------------------
    # USER CONFIGURATION BLOCK
    # ---------------------------------------------------------
    
    # 1. Map directly to your Windows Downloads folder
    DOWNLOADS_DIR = r"C:\Users\ishan\Downloads"
    BIN_FILE_NAME = "00000014.bin" # <--- CHANGE THIS to the actual file name from Mission Planner
    
    BIN_FILE_PATH = os.path.join(DOWNLOADS_DIR, BIN_FILE_NAME)
    
    # 2. Toggle this to True if the 50, 100, 150 altitudes below are in FEET
    ALTITUDES_ARE_FEET = False 
    
    # 3. The 7 Scoring Waypoints (Lat, Lon, Alt)
    SCORING_WAYPOINTS = [
        (35.049875, -118.150127, 50),
        (35.050154, -118.149205, 150), 
        (35.052716, -118.150231, 100),
        (35.050101, -118.150687, 50),
        (35.048860, -118.153033, 100),
        (35.047553, -118.151674, 100), # Patched the 35.47 typo here
        (35.048843, -118.151194, 50)
    ]
    
    # Run the math engine
    analyze_blackbox(BIN_FILE_PATH, SCORING_WAYPOINTS, ALTITUDES_ARE_FEET)
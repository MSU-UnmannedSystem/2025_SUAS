from pymavlink import mavutil
import math
import csv
import os
import sys

def analyze_blackbox(bin_filepath, scoring_waypoints, altitudes_in_feet):
    print(f"=== C-UASC 2026: 3D WAYPOINT LOG ANALYZER ===")
    print(f"Targeting DataFlash Log: {bin_filepath}")
    
    # 1. Verify the file exists
    if not os.path.exists(bin_filepath):
        print(f"\n[ERROR] File Not Found: {bin_filepath}")
        print("Ensure you downloaded the .bin file from Mission Planner to your Downloads folder.")
        sys.exit(1)

    # 2. Open the .bin file
    try:
        mlog = mavutil.mavlink_connection(bin_filepath)
    except Exception as e:
        print(f"Failed to open log: {e}")
        sys.exit(1)

    # 3. Convert provided altitudes to meters if they were entered in feet
    converted_waypoints = []
    for lat, lon, alt in scoring_waypoints:
        final_alt = alt * 0.3048 if altitudes_in_feet else alt
        converted_waypoints.append((lat, lon, final_alt))

    # 4. Initialize trackers for the tightest 3D approach to each waypoint
    closest_approaches = {i: 99999999.0 for i in range(len(converted_waypoints))}
    
    print(f"Scanning GPS & Barometer telemetry against {len(converted_waypoints)} 3D scoring targets...")
    
    gps_count = 0
    while True:
        msg = mlog.recv_match(type=['POS'], blocking=False)
        if msg is None:
            break 
        
        lat = msg.Lat
        lon = msg.Lng
        alt_agl = msg.RelHomeAlt 
        gps_count += 1

        for i, (t_lat, t_lon, t_alt_m) in enumerate(converted_waypoints):
            # Horizontal Offset (Meters)
            lat_dif_m = (abs(t_lat - lat)) * 111319
            # Dynamic cosine calculation based on current latitude
            lon_dif_m = (abs(t_lon - lon)) * 111319 * math.cos(math.radians(lat))
            
            # Vertical Offset (Meters)
            alt_dif_m = abs(t_alt_m - alt_agl)

            # True 3D Spherical Distance
            distance_3d = (lat_dif_m**2 + lon_dif_m**2 + alt_dif_m**2)**0.5

            if distance_3d < closest_approaches[i]:
                closest_approaches[i] = distance_3d

    print(f"Parsed {gps_count} positional telemetry points.")

    # --- OUTPUT RESULTS TO TERMINAL (FOR THE TEAM) ---
    print("\n--- FINAL 3D SCORING RESULTS ---")
    max_score_per_wp = 10.0
    total_max_score = len(converted_waypoints) * max_score_per_wp
    total_penalty = 0.0
    
    for i in range(len(converted_waypoints)):
        dist = closest_approaches[i]
        penalty = max(0, dist - 1.0)
        total_penalty += penalty
        
        # Calculate the estimated score for this specific waypoint
        est_wp_score = max(0, max_score_per_wp - penalty)
        
        print(f"Waypoint {i+1}: Closest 3D Approach = {dist:.2f}m | Penalty = -{penalty:.2f} pts | Est. Score = {est_wp_score:.2f}/{max_score_per_wp}")
    
    # Calculate final estimated score (preventing negative total scores)
    estimated_final_score = max(0, total_max_score - total_penalty)
    
    print(f"--------------------------------")
    print(f"Total Penalty Incurred: -{total_penalty:.2f} Points")
    print(f"ESTIMATED TOTAL POINTS: {estimated_final_score:.2f} / {total_max_score:.2f} Points")
    # --- SAVE TO CSV (FOR THE JUDGES) ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Updated to official team naming convention
    csv_filename = os.path.join(output_dir, "MSU_CUASC_WAYPOINT_NAVIGATION.csv")
    
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Penalties commented out for official submission
        writer.writerow(["Waypoint_Index", "Target_Lat", "Target_Lon", "Target_Alt_Meters", "Closest_3D_Approach_M"])
        # writer.writerow(["Waypoint_Index", "Target_Lat", "Target_Lon", "Target_Alt_Meters", "Closest_3D_Approach_M", "Penalty_Points"])
        
        for i in range(len(converted_waypoints)):
            t_lat, t_lon, t_alt_m = converted_waypoints[i]
            
            writer.writerow([i+1, t_lat, t_lon, round(t_alt_m, 2), round(closest_approaches[i], 2)])
            # writer.writerow([i+1, t_lat, t_lon, round(t_alt_m, 2), round(closest_approaches[i], 2), round(max(0, closest_approaches[i] - 1.0), 2)])
            
    print(f"\nReport saved successfully to: {csv_filename}")


if __name__ == "__main__":
    # ---------------------------------------------------------
    # USER CONFIGURATION BLOCK
    # ---------------------------------------------------------
    
    # 1. Map directly to your Windows Downloads folder
    DOWNLOADS_DIR = r"C:\Users\ishan\Downloads"
    BIN_FILE_NAME = "00000030.BIN" 
    
    BIN_FILE_PATH = os.path.join(DOWNLOADS_DIR, BIN_FILE_NAME)
    
    # 2. Toggle this to True if the altitudes below are in FEET
    ALTITUDES_ARE_FEET = True 
    
    # 3. The 7 Scoring Waypoints (Lat, Lon, Alt)
    SCORING_WAYPOINTS = [
        (35.049875, -118.150127, 50),
        (35.050154, -118.149204, 150), 
        (35.052716, -118.150231, 100),
        (35.050101, -118.150687, 50),
        (35.048860, -118.153033, 100),
        (35.047553, -118.151674, 100), 
        (35.048843, -118.151194, 50)
    ]
    
    # Run the math engine
    analyze_blackbox(BIN_FILE_PATH, SCORING_WAYPOINTS, ALTITUDES_ARE_FEET)
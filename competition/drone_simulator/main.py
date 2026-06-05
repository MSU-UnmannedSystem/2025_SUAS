import sys
import os

# PATHING FIX: Forces Python to look in the parent folder, allowing drone_simulator scripts to see jetson_rescue
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vehicle.connection import connect_vehicle
from config.mission_params import parameters

def main():
    print("\n=== MICHIGAN STATE SUAS: BULLSEYE MISSIONS ===")
    print("1. Payload Drop (130g Beanbag - 10ft Test)")
    print("2. Package Delivery (1-2kg Cube - 10ft Test)")
    print("==============================================")
    
    choice = input("Select flight slot mission (1-2): ")
    
    # --- FLIGHT TOGGLE ---
    # True = Props ON. Drone will arm, spool up, and fly to 10 feet. 
    # False = Props OFF. Skips takeoff, allows you to walk the drone by hand.
    LIVE_FLIGHT = True 
    
    if choice == '1':
        from missions.payload_drop import PayloadDropMission
        mission_class = PayloadDropMission
    elif choice == '2':
        from missions.package_delivery import PackageDeliveryMission
        mission_class = PackageDeliveryMission
    else:
        print("Invalid choice. Terminating.")
        sys.exit()

    print("\n[INIT] Connecting to Cube Orange...")
    master = connect_vehicle("/dev/ttyTHS1", baud_rate=921600)

    print(f"\n[INIT] Loading Mission Option {choice}...")
    # desk_test variable is the inverse of LIVE_FLIGHT
    mission = mission_class(master, parameters(), desk_test=(not LIVE_FLIGHT))
    mission.run()

if __name__ == "__main__":
    main()
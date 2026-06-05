import sys
from vehicle.connection import connect_vehicle
from config.mission_params import parameters

def main():
    print("\n=== MICHIGAN STATE SUAS: MISSION CONTROL ===")
    print("1. Package Drop (130g Beanbag - Target > 6ft)")
    print("2. Package Delivery (1-2kg Cube - Target @ 5ft)")
    print("3. Object Localization (Black/White Squares - Target > 25ft)")
    print("4. Circuit Time Trial (Max Speed Waypoints)")
    print("============================================")
    
    choice = input("Select flight slot mission (1-4): ")
    
    # --- TEST TOGGLE ---
    # Set to True for Wednesday's desk test (skips takeoff wait)
    # Set to False for actual field flight
    DESK_TEST = True 
    
    if choice == '1':
        from missions.payload_drop import PayloadDropMission
        mission_class = PayloadDropMission
    elif choice == '2':
        from missions.package_delivery import PackageDeliveryMission
        mission_class = PackageDeliveryMission
    elif choice == '3':
        from missions.object_localization import ObjectLocalizationMission
        mission_class = ObjectLocalizationMission
    elif choice == '4':
        from missions.circuit_trial import CircuitTimeTrialMission
        mission_class = CircuitTimeTrialMission
    else:
        print("Invalid choice. Terminating.")
        sys.exit()

    print("\n[INIT] Connecting to Cube Orange...")
    # /dev/ttyTHS1 is the Jetson's physical serial port
    # 921600 is the baud rate configured in the Cube Orange
    master = connect_vehicle("/dev/ttyTHS1", baud_rate=921600)

    print(f"\n[INIT] Loading Mission Option {choice}...")
    mission = mission_class(master, parameters(), desk_test=DESK_TEST)
    mission.run()

if __name__ == "__main__":
    main()
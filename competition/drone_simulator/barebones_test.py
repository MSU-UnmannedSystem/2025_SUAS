import sys
import os
import time

# Route to sibling folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pymavlink import mavutil
from vehicle.connection import connect_vehicle
from vehicle.modes import change_mode

def run_barebone_test():
    print("\n=== NAKED CUBE I/O TEST ===")
    
    # Connect to the Cube
    master = connect_vehicle("/dev/ttyTHS1", baud_rate=921600)
    print("[SUCCESS] Data link established.")
    
    # --- TEST 1: PIN 6 (CUBE DROP) ---
    print("\n[TEST 1] Firing Pin 6 to 1900 PWM...")
    print(">>> Look at 'ch6out' in Mission Planner! It should jump to 1900.")
    master.mav.command_long_send(master.target_system, master.target_component, 
                                 mavutil.mavlink.MAV_CMD_DO_SET_SERVO, 0, 6, 1900, 0, 0, 0, 0, 0)
    time.sleep(5)

    # --- TEST 2: PIN 7 (BALL DROP) ---
    print("\n[TEST 2] Firing Pin 7 to 1900 PWM...")
    print(">>> Look at 'ch7out' in Mission Planner! It should jump to 1900.")
    master.mav.command_long_send(master.target_system, master.target_component, 
                                 mavutil.mavlink.MAV_CMD_DO_SET_SERVO, 0, 7, 1900, 0, 0, 0, 0, 0)
    time.sleep(5)

    # --- TEST 3: THE MODE HANDOFF ---
    print("\n[TEST 3] Forcing ArduPilot into LOITER mode...")
    change_mode(master, "LOITER")
    print(">>> Look at the Mission Planner HUD! The flight mode should change to LOITER.")
    
    print("\n[SUCCESS] I/O Test Complete. If all 3 tests matched in Mission Planner, your MAVLink bridge is flawless.")

if __name__ == "__main__":
    run_barebone_test()
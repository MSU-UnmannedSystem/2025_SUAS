import sys
import os
import time

# Route to sibling folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pymavlink import mavutil
from vehicle.connection import connect_vehicle
from jetson_rescue.camera import VisionPipeline
from vehicle.modes import change_mode

def run_table_test():
    print("\n=== MANUAL TABLE TEST (HARDWARE-IN-THE-LOOP) ===")
    
    # NOTE: Change to /dev/ttyTHS0 if ttyTHS1 throws a permission/not found error
    master = connect_vehicle("/dev/ttyTHS1", baud_rate=921600)
    vision = VisionPipeline(camera_index=0, target_type='red_bullseye')
    
    cube_dropped = False
    ball_dropped = False
    target_lost_timer = None

    print("\n[READY] Hold the drone over the target. Monitoring telemetry...\n")
    
    try:
        while not (cube_dropped and ball_dropped):
            # 1. Read the Camera
            x, y = vision.get_target_pixel()
            
            # 2. Read the Barometer (Relative Altitude)
            msg = master.recv_match(type="GLOBAL_POSITION_INT", blocking=False)
            alt = msg.relative_alt / 1000.0 if msg else 0.0
            
            # 3. Check Centering (1080p center is ~960, 540. Allow a +/- 150px window)
            centered = False
            if x is not None and y is not None:
                if 810 < x < 1110 and 390 < y < 690:
                    centered = True
                    target_lost_timer = None # Reset the panic timer when centered

            # --- THE ACTIVE AUTO-FAILSAFE ---
            if not centered:
                if target_lost_timer is None:
                    target_lost_timer = time.time()
                elif time.time() - target_lost_timer > 4.0:
                    print("\n\n[CRITICAL ERROR] Vision lock lost for 4 seconds!")
                    print("[AUTO-FAILSAFE] Forcing ArduPilot into LOITER mode...")
                    
                    # Command the Cube to switch modes
                    change_mode(master, "LOITER")
                    
                    print("[SAFEGUARD] Terminating companion computer guidance loop immediately.")
                    print("-> PILOT HAS 100% MANUAL CONTROL.")
                    vision.release()
                    sys.exit(0)

            # 4. Print the Live HUD
            print(f"| ALT: {alt:.2f}m | PIXEL: ({x}, {y}) | CENTERED: {centered} |", end='\r')

            # 5. The Floor Test (Simulating Phase 1 Cube Drop at 0.5m)
            if centered and alt < 0.5 and not cube_dropped:
                print(f"\n\n[TRIGGER] Cube condition met (Alt < 0.5m). Actuating Pin 6 (1900 PWM)...")
                master.mav.command_long_send(master.target_system, master.target_component, 
                                             mavutil.mavlink.MAV_CMD_DO_SET_SERVO, 0, 6, 1900, 0, 0, 0, 0, 0)
                cube_dropped = True
                target_lost_timer = None # Reset to prevent false trips during sleep
                time.sleep(2) 

            # 6. The Ceiling Test (Simulating Phase 2 Ball Drop at 1.5m)
            if centered and alt > 1.5 and not ball_dropped:
                print(f"\n\n[TRIGGER] Ball condition met (Alt > 1.5m). Actuating Pin 7 (1900 PWM)...")
                master.mav.command_long_send(master.target_system, master.target_component, 
                                             mavutil.mavlink.MAV_CMD_DO_SET_SERVO, 0, 7, 1900, 0, 0, 0, 0, 0)
                ball_dropped = True
                target_lost_timer = None
                time.sleep(2)

            time.sleep(0.1)

        print("\n\n[SUCCESS] Both payload drops completed successfully.")
        
    except KeyboardInterrupt:
        vision.release()
        print("\n\n[SHUTDOWN] Test terminated safely by user.")

if __name__ == "__main__":
    run_table_test()

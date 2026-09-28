from pymavlink import mavutil
import time
import random as rand
import math



options = "a: Make drone fly\n" \
"b: Make drone land\n" \
"c: Return to location\n" \
"d: Make drone fly forward\n" \
"t: Traverse payload drop area\n" \
"e: Terminate program\n" \
"f: Print options\n"

def prompt_user(master):
    """Prompts the user for a choice from the options variable"""

    choice = input("What would you like to do: ")

    if choice == 'a':
        make_drone_fly_up(master)
        return 1
    elif choice == 'b':
        land_drone(master)
        return 1
    elif choice == 'c':
        return_to_location(master)
        return 1
    elif choice == 'd':
        make_drone_fly_forward(master)
        return 1
    elif choice == 'e':
        print("\nProgram terminating...")
        return None
    elif choice == 'f':
        print(options)
        return 1
    elif choice == 't':
        traverse_payload(master)
        return 1
    else:
        return None

def arm_drone(master):
    """"Arms the drone when ready to take off"""

    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1, 0, 0, 0, 0, 0, 0) 
    print("arming.....")
    master.motors_armed_wait()
    print("Motors armed")

def change_mode(mode, master):
    mode_id = master.mode_mapping()[mode]
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id
    )

def make_drone_fly_up(master, flight_height=0):
    """"Makes the drone fly (uses arm and changed mode function)"""

    #changed Mode to guided so it can fly upwards and arms drone
    change_mode("GUIDED", master)
    arm_drone(master)
    #sets the baseline to 0 until first itteration
    base_line = None

    #ask for user input for what height the drone should fly to
    if flight_height == 0:
        flight_height = int(input("Target altitude: "))
    
    #starts the take off command
    master.mav.command_long_send(master.target_system,
                                 master.target_component,
                                 mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                                 0,0,0,0,0,0,0, flight_height)

    #starting altitude
    print("Taking off")
    print("Altitude: 0.00m")

    #while loop that prints off the current altitude for the drone
    #itterates unitl drone reaches target height
    while True:
        
        msg = master.recv_match(type="VFR_HUD", blocking=True)

        if not msg:
            continue
        
        if base_line is None:
            base_line = msg.alt
            altitude = msg.alt

        altitude = msg.alt - base_line
        #groundspeed = msg.groundspeed
        if (flight_height - (0.01)) <= altitude:
            print("Altitude Reached!\n")
            break
        elif altitude < 0.01:
            continue
        else:
            print(f"Altitude: {altitude:.2f}m")

def land_drone(master):
    """"Lands the Drone"""
    print("Landing...\n")
    change_mode("LAND", master)
    
    while True:
        msg = master.recv_match(type="GLOBAL_POSITION_INT", blocking = True)
        
        alt = msg.relative_alt/1000.0
        if alt < 0.01:
            print("landed!!!\n")
            break

def make_drone_fly_forward(master):
    """Makes the drone fly forward with proper mission upload protocol"""
    
    print("Starting mission upload...")
    
    # First, clear any existing mission
    print("Clearing existing mission...")
    master.mav.mission_clear_all_send(master.target_system, master.target_component)
    ack = master.recv_match(type='MISSION_ACK', blocking=True, timeout=5)
    if ack:
        print(f"Mission clear result: {ack.type}")
    else:
        print("No clear ACK received, continuing anyway...")
    
    time.sleep(1)
    
    # Step 1: Send mission count
    num_wp = 1
    print(f"Sending mission count: {num_wp}")
    master.mav.mission_count_send(master.target_system, master.target_component, num_wp)
    
    # Step 2: Wait for mission request
    print("Waiting for mission request...")
    try:
        msg = master.recv_match(type=['MISSION_REQUEST', 'MISSION_REQUEST_INT'], blocking=True, timeout=10)
        if not msg:
            print("Timeout waiting for mission request!")
            return False
            
        print(f"Drone requested waypoint: {msg.seq}")
        
        # Step 3: Send the mission item
        flight_height = int(input("Target altitude: "))

        print("Sending mission item...")
        master.mav.mission_item_int_send(
            master.target_system, 
            master.target_component, 
            msg.seq,  # Use the sequence from the request
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, 
            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 
            0,  # current (0 = not current)
            1,  # autocontinue
            0, 0, 0, 0,  # param1-4
            int(42.7782989 * 1e7),  # lat
            int(-84.5959282 * 1e7),   # lon
            flight_height  # alt
        )
        
        # Step 4: Wait for mission ACK
        print("Waiting for mission ACK...")
        ack = master.recv_match(type='MISSION_ACK', blocking=True, timeout=10)
        if ack:
            print(f"Mission upload result: {ack.type}")
            if ack.type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
                print("Mission upload successful!")
                
                # Now set mode and arm
                make_drone_fly_up(master, flight_height)
                change_mode("AUTO", master)

                print("Flying to Waypoint")
                msg = master.recv_match(type="")
            else:
                print(f"Mission upload failed with error: {ack.type}")
        else:
            print("No mission ACK received!")
            
    except Exception as e:
        print(f"Error during mission upload: {e}")

def return_to_location(master):
    change_mode("RTL", master)

def traverse_payload(master):
    random_lattitude = rand.randint(int(35.05935 * 1e7), int(35.06012 * 1e7))
    random_longitude = rand.randint(int(-118.160 * 1e7), int(-118.158 * 1e7))

    flight_height = 30
    tolerance_meters = 5
    change_mode("STABILIZE", master)
    arm_drone(master)
    make_drone_fly_up(master, flight_height)
    change_mode("AUTO", master)

    print(random_lattitude, ", ", random_longitude)
    while True:
        msg = master.recv_match(type="GLOBAL_POSITION_INT", blocking = True)
        
        latitude_dif_meters = (abs(random_lattitude - msg.lat) / 1e7) * 111319
        longitude_dif_meters = (abs(random_longitude - msg.lon) / 1e7) * 111319* math.cos(math.radians(35))

        distance = (latitude_dif_meters**2 + longitude_dif_meters**2)**(1/2)
        print(distance, ", ", msg.lat, ", ", msg.lon)
        if (distance <= tolerance_meters):
            print("Dropping Payload")
            print("Returning to launch")
            return_to_location(master)
            break
    


#Main Function
#Initalizes Mavlink connection to SITL (Software In The Loop)
def main():
    master = mavutil.mavlink_connection("udp:127.0.0.1:14553")

    print("Waiting for heartbeat...")

    master.wait_heartbeat()

    print(f"Connected to system {master.target_system}, component {master.target_component}")

    print(options)
    while True:
        response = prompt_user(master)

        if response:
            continue
        else:
            break
    




if __name__ == "__main__":
    main()
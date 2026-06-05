def parameters():
    return {
        "takeoff_alt": 3.5, # ~11.5 feet. Safe altitude for the 10-foot test.
        "waypoints": [],    # Not needed for autonomous bullseye hunting
        "time_limit": 600,
        "max_alt": 15.0,    # Hard ceiling lowered to prevent flyaways
        "min_alt": 1.0, 
        "end_mode": "RTL",
        "geofence": (35.04709, 35.05322, -118.15354, -118.14854)
    }
def parameters():
    return {
        "takeoff_alt": 10.0, # 10 meters (~33 feet) - Safe altitude for all tasks
        "waypoints": [
            # Will be provided morning of competition. Placeholder for now.
            (35.05000, -118.15000, 10),
            (35.05100, -118.15100, 10)
        ],
        "time_limit": 600, # 10 minutes max flight time per rules
        "max_alt": 120,    # FAA Part 107 ceiling (400 ft)
        "min_alt": 1.0,    # Lowered to 1.0m to allow for Package Delivery descent
        "end_mode": "RTL",
        
        # Mojave Air & Space Port Competition Geofence Bounding Box
        # Derived from Appendix A rulebook coordinates (Min Lat, Max Lat, Min Lon, Max Lon)
        "geofence": (35.04709, 35.05322, -118.15354, -118.14854)
    }
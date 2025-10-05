#!/usr/bin/env python3
"""
This script displays the first SpaceX launch with the following information:
<launch name> (<date>) <rocket name> - <launchpad name> (<launchpad locality>)
"""

import requests


if __name__ == "__main__":
    # Fetch all launches
    launches_url = "https://api.spacexdata.com/v4/launches"
    rockets_url = "https://api.spacexdata.com/v4/rockets"
    launchpads_url = "https://api.spacexdata.com/v4/launchpads"

    launches = requests.get(launches_url).json()
    rockets = requests.get(rockets_url).json()
    launchpads = requests.get(launchpads_url).json()

    # Sort launches by date_unix (ascending)
    launches.sort(key=lambda x: x.get("date_unix", float("inf")))

    # Get the first launch (earliest one)
    first_launch = launches[0]

    # Extract information
    launch_name = first_launch.get("name")
    launch_date = first_launch.get("date_local")
    rocket_id = first_launch.get("rocket")
    launchpad_id = first_launch.get("launchpad")

    # Find rocket name
    rocket_name = next(
        (r.get("name") for r in rockets if r.get("id") == rocket_id), "Unknown"
    )

    # Find launchpad name and locality
    launchpad_data = next(
        (lp for lp in launchpads if lp.get("id") == launchpad_id), {}
    )
    launchpad_name = launchpad_data.get("name", "Unknown")
    launchpad_locality = launchpad_data.get("locality", "Unknown")

    # Print result in required format
    print(f"{launch_name} ({launch_date}) {rocket_name} - "
          f"{launchpad_name} ({launchpad_locality})")

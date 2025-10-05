#!/usr/bin/env python3
"""
Displays the number of SpaceX launches per rocket, sorted by
nnumber of launches descending and alphabetically if tied.
"""

import requests
from collections import Counter


if __name__ == "__main__":
    # Fetch all launches and rockets
    launches = requests.get("https://api.spacexdata.com/v4/launches").json()
    rockets = requests.get("https://api.spacexdata.com/v4/rockets").json()

    # Build a mapping from rocket ID to rocket name
    rocket_names = {r["id"]: r["name"] for r in rockets}

    # Count launches per rocket
    rocket_counter = Counter()
    for launch in launches:
        rocket_id = launch.get("rocket")
        if rocket_id in rocket_names:
            rocket_counter[rocket_names[rocket_id]] += 1

    # Sort first by number of launches descending, then by name ascending
    sorted_rockets = sorted(
        rocket_counter.items(),
        key=lambda x: (-x[1], x[0])
    )

    # Print results
    for name, count in sorted_rockets:
        print(f"{name}: {count}")

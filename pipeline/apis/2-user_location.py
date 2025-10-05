#!/usr/bin/env python3
"""
This script fetches and prints the location of a GitHub user using the GitHub API.
Usage example:
    ./2-user_location.py https://api.github.com/users/holbertonschool
"""

import sys
import requests
from datetime import datetime

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./2-user_location.py <GitHub API user URL>")
        sys.exit(1)

    url = sys.argv[1]
    response = requests.get(url)

    # If the rate limit is exceeded
    if response.status_code == 403:
        reset_time = response.headers.get("X-RateLimit-Reset")
        if reset_time:
            # Convert reset timestamp to minutes from now
            reset_timestamp = int(reset_time)
            now = datetime.now().timestamp()
            minutes_remaining = int((reset_timestamp - now) / 60)
            print(f"Reset in {minutes_remaining} min")
        else:
            print("Reset in unknown time")
        sys.exit(0)

    # If user not found
    if response.status_code == 404:
        print("Not found")
        sys.exit(0)

    # If successful request
    if response.status_code == 200:
        data = response.json()
        print(data.get("location", "Not found"))
    else:
        print("Not found")

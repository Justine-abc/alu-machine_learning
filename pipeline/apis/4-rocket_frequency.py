#!/usr/bin/env python3
"""
Script that displays the number of launches per rocket
using the (unofficial) SpaceX API.
"""
import requests


if __name__ == '__main__':
    # Get all launches
    launches_resp = requests.get("https://api.spacexdata.com/v4/launches")
    if launches_resp.status_code != 200:
        print("Error fetching launches")
        exit(1)

    launches = launches_resp.json()

    # Get all rockets
    rockets_resp = requests.get("https://api.spacexdata.com/v4/rockets")
    if rockets_resp.status_code != 200:
        print("Error fetching rockets")
        exit(1)

    rockets = rockets_resp.json()

    # Build rocket id -> name map
    rocket_map = {}
    for rocket in rockets:
        rocket_map[rocket.get("id")] = rocket.get("name")

    # Count launches per rocket
    launch_counts = {}
    for launch in launches:
        rocket_id = launch.get("rocket")
        rocket_name = rocket_map.get(rocket_id, "Unknown")
        launch_counts[rocket_name] = launch_counts.get(rocket_name, 0) + 1

    # Sort: by count descending, then by name ascending
    sorted_rockets = sorted(
        launch_counts.items(),
        key=lambda x: (-x[1], x[0])
    )

    for rocket_name, count in sorted_rockets:
        print("{}: {}".format(rocket_name, count))

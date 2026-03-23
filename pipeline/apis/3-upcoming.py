#!/usr/bin/env python3
"""
Script that displays the upcoming SpaceX launch info
using the (unofficial) SpaceX API.
"""
import requests


if __name__ == '__main__':
    # Get all upcoming launches
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    response = requests.get(url)

    if response.status_code != 200:
        exit(0)

    launches = response.json()

    if not launches:
        exit(0)

    # Sort by date_unix, pick the soonest
    launches.sort(key=lambda x: x.get("date_unix", float('inf')))
    launch = launches[0]

    # Get launch details
    name = launch.get("name")
    date_local = launch.get("date_local")
    rocket_id = launch.get("rocket")
    launchpad_id = launch.get("launchpad")

    # Get rocket name
    r_url = "https://api.spacexdata.com/v4/rockets/{}"
    rocket_resp = requests.get(r_url.format(rocket_id))
    r_json = rocket_resp.json()
    rocket_name = r_json.get("name") if rocket_resp.status_code == 200 else ""

    # Get launchpad info
    p_url = "https://api.spacexdata.com/v4/launchpads/{}"
    pad_resp = requests.get(p_url.format(launchpad_id))
    if pad_resp.status_code == 200:
        pad_data = pad_resp.json()
        pad_name = pad_data.get("name")
        pad_locality = pad_data.get("locality")
    else:
        pad_name = ""
        pad_locality = ""

    # Print formatted output (multi-line to pass Pycodestyle E501)
    print("{} ({}) {} - {} ({})".format(
        name,
        date_local,
        rocket_name,
        pad_name,
        pad_locality
    ))

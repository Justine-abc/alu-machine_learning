#!/usr/bin/env python3
"""
Module that returns list of ships that can hold
a given number of passengers using the SWAPI API.
"""
import requests


def availableShips(passengerCount):
    """
    Returns a list of ship names that can hold
    at least passengerCount passengers.
    """
    ships = []
    url = "https://swapi.dev/api/starships/"

    while url:
        response = requests.get(url)
        if response.status_code != 200:
            break
        data = response.json()

        for ship in data.get("results", []):
            passengers = ship.get("passengers", "0")
            # Clean the string: remove commas, handle "n/a" and "unknown"
            passengers = passengers.replace(",", "").strip()
            if passengers in ("n/a", "unknown", ""):
                continue
            try:
                if int(passengers) >= passengerCount:
                    ships.append(ship.get("name"))
            except ValueError:
                continue

        url = data.get("next")

    return ships

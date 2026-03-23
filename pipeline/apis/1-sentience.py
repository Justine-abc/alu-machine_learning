#!/usr/bin/env python3
"""
Module that returns the list of home planet names
of all sentient species using the SWAPI API.
"""
import requests


def sentientPlanets():
    """
    Returns a list of names of the home planets
    of all sentient species.
    """
    planets = []
    url = "https://swapi.dev/api/species/"

    while url:
        response = requests.get(url)
        if response.status_code != 200:
            break
        data = response.json()

        for species in data.get("results", []):
            classification = species.get("classification", "").lower()
            designation = species.get("designation", "").lower()

            if "sentient" in classification or "sentient" in designation:
                homeworld = species.get("homeworld")
                if homeworld:
                    hw_response = requests.get(homeworld)
                    if hw_response.status_code == 200:
                        hw_data = hw_response.json()
                        planet_name = hw_data.get("name")
                        if planet_name:
                            planets.append(planet_name)

        url = data.get("next")

    return planets

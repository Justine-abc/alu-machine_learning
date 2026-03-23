#!/usr/bin/env python3
"""
Script that prints the location of a specific GitHub user
using the GitHub API.
"""
import requests
import sys
import time


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: ./2-user_location.py <API URL>")
        sys.exit(1)

    url = sys.argv[1]
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        location = data.get("location")
        if location:
            print(location)
        else:
            print("Not found")
    elif response.status_code == 403:
        reset_time = int(response.headers.get("X-Ratelimit-Reset", 0))
        now = int(time.time())
        minutes = (reset_time - now) // 60
        print("Reset in {} min".format(minutes))
    elif response.status_code == 404:
        print("Not found")
    else:
        print("Not found")

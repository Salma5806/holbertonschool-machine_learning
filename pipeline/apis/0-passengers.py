#!/usr/bin/env python3
""" Get requests"""
import requests


def availableShips(passengerCount):
    """
    Function to get request
    Args:
        passengerCount: number of passangers
    Returns: List of planets
    """
    starships = []
    url = 'https://swapi-api.hbtn.io/api/planets/'
    while url is not None:
        response = requests.get(url,
                                headers={'Accept': 'application/json'},
                                params={"term": 'planets'})
        for ship in response.json()['results']:
            passenger = ship['passengers'].replace(',', '')
            if passenger.isnumeric() and int(passenger) >= passengerCount:
                starships.append(ship['name'])
        url = response.json()['next']
    return starships
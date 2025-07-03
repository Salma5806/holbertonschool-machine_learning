#!/usr/bin/env python3
"""
This script defines a function to load the Frozen Lake environment
from the Gymnasium library. The environment simulates an agent trying
to cross a frozen lake without falling into holes. You can customize
the map, choose whether the ice is slippery, and set the render mode.
"""

import gymnasium as gym

def load_frozen_lake_environment(custom_map=None, map_name=None, slippery_ice=False, render_mode=None):
    """
    Loads the Frozen Lake environment with optional customization.

    Parameters:
    - custom_map (list of lists): a custom layout of the environment (optional)
    - map_name (str): the name of a pre-defined map, like "4x4" or "8x8" (optional)
    - slippery_ice (bool): whether the ice is slippery (True) or not (False)
    - render_mode (str): how the environment is rendered (e.g., "human", "ansi")

    Returns:
    - env: the Frozen Lake environment instance
    """
    env = gym.make(
        'FrozenLake-v1',
        desc=custom_map,
        map_name=map_name,
        is_slippery=slippery_ice,
        render_mode=render_mode or "ansi"
    )
    return env

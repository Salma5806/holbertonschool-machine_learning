#!/usr/bin/env python3
"""
This script defines a function to load the Frozen Lake environment
from the Gymnasium library. The environment simulates an agent trying
to cross a frozen lake without falling into holes. You can customize
the map, choose whether the ice is slippery, and set the render mode.
"""

import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None,
                     is_slippery=False, render_mode=None):
    """Loads the Frozen Lake environment with optional customization"""
    env = gym.make(
        'FrozenLake-v1',
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery,
        render_mode="ansi"
    )
    return env

#!/usr/bin/env python3

"""
A function that initializes the Q-table.
"""

import numpy as np


def q_init(env):
    """
    Initializes the Q-table for the given environment.

    Parameters:
    - env: the FrozenLakeEnv instance

    Returns:
    - A NumPy ndarray filled with zeros, with shape
      (number of states, number of actions)
    """
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    return q_table

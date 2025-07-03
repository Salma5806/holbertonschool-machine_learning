#!/usr/bin/env python3

"""
This module contains the epsilon_greedy function, which balances
exploration and exploitation for action selection in reinforcement learning.
"""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Selects an action using the epsilon-greedy policy.

    Parameters:
    - Q (np.ndarray): The Q-table
    - state (int): The current state index
    - epsilon (float): The exploration rate (between 0 and 1)

    Returns:
    - int: The index of the selected action
    """
    p = np.random.uniform(0, 1)
    if p < epsilon:
        action = np.random.randint(0, Q.shape[1])
    else:
        action = np.argmax(Q[state])
    return action

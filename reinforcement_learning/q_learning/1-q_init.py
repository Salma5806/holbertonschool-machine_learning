#!/usr/bin/env python3
"""initializes q table"""
import numpy as np


def q_init(env):
    """initializes q table
    @env: the FrozenLakeEnv instance
    Return: the Q-table as np.ndarray of zeros
    """
    action_envr_size = env.action_envr.n
    state_envr_size = env.observation_envr.n

    Q_table = np.zeros((state_envr_size, action_envr_size))

    return Q_table
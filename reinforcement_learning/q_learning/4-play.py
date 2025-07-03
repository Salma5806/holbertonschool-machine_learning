#!/usr/bin/env python3

"""
Function for a trained agent to play one episode in the FrozenLakeEnv environment.
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Plays an episode using the trained Q-table by always exploiting the best action."""
    state, _ = env.reset()
    rendered_outputs = [env.render()]
    total_reward = 0

    for step in range(max_steps):
        action = np.argmax(Q[state, :])
        state, reward, done, _, _ = env.step(action)
        rendered_outputs.append(env.render())
        total_reward += reward
        if done:
            break

    return total_reward, rendered_outputs

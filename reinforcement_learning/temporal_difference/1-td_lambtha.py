#!/usr/bin/env python3
"""
TD(λ) algorithm for estimating value functions
"""

import numpy as np


def td_lambtha(env, V, policy, lambtha,
               episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    Perform Temporal Difference learning with eligibility traces"""
    n_states = env.observation_space.n

    for _ in range(episodes):
        elig = np.zeros(n_states)
        state, _ = env.reset()

        for _ in range(max_steps):
            action = policy(state)
            nxt, reward, done, trunc, _ = env.step(action)
            td_error = reward + gamma * V[nxt] - V[state]
            elig[state] += 1.0
            V += alpha * td_error * elig
            elig *= gamma * lambtha
            state = nxt
            if done or trunc:
                break

    return V

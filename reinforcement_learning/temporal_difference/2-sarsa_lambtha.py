#!/usr/bin/env python3
"""
Implementation of SARSA(λ) with eligibility traces
"""
import numpy as np


def epsilon_greedy(Q, state, eps):
    """
    Select an action using epsilon-greedy exploration.
    """
    if np.random.rand() < eps:
        return np.random.randint(Q.shape[1])
    return np.argmax(Q[state])


def sarsa_lambtha(env, Q, lambtha,
                  episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99,
                  epsilon=1.0, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """
    SARSA(λ) algorithm to estimate Q-values with eligibility traces.
    """
    eps0 = epsilon 

    for ep in range(episodes):
        state, _ = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        traces = np.zeros_like(Q)

        for _ in range(max_steps):
            nxt, reward, done, trunc, _ = env.step(action)
            nxt_action = epsilon_greedy(Q, nxt, epsilon)
            td_error = reward + gamma * Q[nxt, nxt_action] - Q[state, action]
            traces[state, action] += 1
            Q += alpha * td_error * traces
            traces *= gamma * lambtha
            state, action = nxt, nxt_action
            if done or trunc:
                break
        epsilon = min_epsilon + (eps0 - min_epsilon) * np.exp(-epsilon_decay * ep)

     return Q

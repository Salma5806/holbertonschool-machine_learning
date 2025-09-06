#!/usr/bin/env python3
"""
Implementation of SARSA(λ) with eligibility traces for Reinforcement Learning
"""
import numpy as np


def choose_action(Q, state, epsilon):
    """Select an action using an epsilon-greedy strategy"""
    if np.random.rand() < epsilon:  # Explore
        return np.random.randint(Q.shape[1])
    else:
        return np.argmax(Q[state])


def sarsa_lambda(
    env,
    Q,
    trace_decay,
    episodes=5000,
    max_steps=100,
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
    min_epsilon=0.1,
    epsilon_decay=0.05
):
    """Train a Q-table using SARSA(λ) algorithm with eligibility traces"""
    eps_init = epsilon

    for ep in range(episodes):
        state, _ = env.reset()
        action = choose_action(Q, state, epsilon)
        traces = np.zeros_like(Q)

        for _ in range(max_steps):
            next_state, reward, done, truncated, _ = env.step(action)
            next_action = choose_action(Q, next_state, epsilon)
            td_error = reward + gamma * Q[next_state, next_action] - Q[state, action]
            traces[state, action] += 1
            Q += alpha * td_error * traces
            traces *= gamma * trace_decay
            state, action = next_state, next_action
            if done or truncated:
                break
        epsilon = min_epsilon + (eps_init - min_epsilon) * np.exp(-epsilon_decay * ep)

    return Q

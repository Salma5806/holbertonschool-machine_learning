#!/usr/bin/env python3
"""
SARSA(λ) algorithm with eligibility traces
"""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Choisir une action avec une stratégie epsilon-greedy"""
    if np.random.rand() < epsilon:
        return np.random.randint(Q.shape[1]) 
    return np.argmax(Q[state]) 


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1,
                  min_epsilon=0.1, epsilon_decay=0.05):
    """
    Implémente l’algorithme SARSA(λ) avec traces d’éligibilité
    """
    n_states, n_actions = Q.shape

    for ep in range(episodes):
        state, _ = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        E = np.zeros((n_states, n_actions)) 

        for _ in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)
            td_error = reward + gamma * Q[next_state, next_action] - Q[state, action]
            E[state, action] += 1
            Q += alpha * td_error * E
            E *= gamma * lambtha
            state, action = next_state, next_action
            if terminated or truncated:
                break
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))
    return Q

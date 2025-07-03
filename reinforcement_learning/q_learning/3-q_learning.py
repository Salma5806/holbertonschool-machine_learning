#!/usr/bin/env python3

"""
Q-learning function to train an agent on the FrozenLakeEnv environment.
"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(
        env,
        Q,
        episodes=5000,
        max_steps=100,
        alpha=0.1,
        gamma=0.99,
        epsilon=1,
        min_epsilon=0.1,
        epsilon_decay=0.05):
    """This function trains an agent using the Q-learning algorithm"""
    total_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        rewards_current_episode = 0

        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, truncated, _ = env.step(action)
            if done and reward == 0:
                reward = -1
            Q[state, action] = Q[state, action] + alpha * \
                (reward + gamma * np.max(Q[new_state] - Q[state, action]))
            state = new_state
            rewards_current_episode += reward
            if done or truncated:
                break
        epsilon = max(min_epsilon, epsilon * np.exp(-epsilon_decay * episode))
        total_rewards.append(rewards_current_episode)
    return Q, total_rewards

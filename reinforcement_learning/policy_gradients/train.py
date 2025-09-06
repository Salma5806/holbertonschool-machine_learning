#!/usr/bin/env python3
"""
Training a policy gradient agent for CartPole-v1
"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    Trains a policy gradient agent.

    env: initial environment
    nb_episodes: number of episodes
    alpha: learning rate
    gamma: discount factor

    Returns: list of scores per episode
    """
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    weight = np.random.rand(n_states, n_actions)

    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        done = False
        episode_rewards = []
        episode_gradients = []
        while not done:
            action, grad = policy_gradient(state, weight)
            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_rewards.append(reward)
            episode_gradients.append(grad)

            state = new_state
        G = 0
        for i in reversed(range(len(episode_rewards))):
            G = gamma * G + episode_rewards[i]
            weight += alpha * G * episode_gradients[i]

        score = sum(episode_rewards)
        scores.append(score)
        print(f"Episode: {episode} Score: {score}")

    return scores

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 02:00:37 2024

@author: TUGRA1
"""

import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Hyperparameters
alpha = 0.5  # learning rate
gamma = 0.9  # discount rate

# Initialize the environment
environment = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")
nb_states = environment.observation_space.n
nb_actions = environment.action_space.n

# Initialize the Q-table
qtable = np.zeros((nb_states, nb_actions))
outcomes = []

# Number of episodes for training
episodes = 1000

# Training loop
for i in tqdm(range(episodes)):
    state, _ = environment.reset()
    done = False  # Indicates if the episode is done
    outcomes.append("Failure")  # Default outcome for each episode

    # While the episode is not finished
    while not done:
        # Choose an action (exploit or explore)
        if np.max(qtable[state]) > 0:
            action = np.argmax(qtable[state])  # Exploit: choose the best action
        else:
            action = environment.action_space.sample()  # Explore: take random action

        # Take the action and get the new state, reward, and other info
        new_state, reward, done, info, = environment.step(action)

        # Update Q-table using the Q-learning formula
        qtable[state, action] = qtable[state, action] + alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])

        # Move to the new state
        state = new_state

        # If the agent receives a reward, mark the episode as successful
        if reward:
            outcomes[-1] = "Success"

# Print the Q-table after training
print("Q-table After Training:")
print(qtable)

# Plot the outcomes of the episodes
plt.bar(range(episodes), outcomes)
plt.xlabel("Episodes")
plt.ylabel("Outcome")
plt.title("Episode Outcomes (Success vs. Failure)")
plt.show()

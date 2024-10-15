# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:04:01 2024

@author: TUGRA1
"""

import gym
import numpy as np
import random
from tqdm import tqdm

env = gym.make("Taxi-v3", render_mode = "ansi")
env.reset()
print(env.render)

action_space = env.action_space.n
state_space = env.observation_space.n
q_table= np.zeros((state_space,action_space))

alpha =0.1 #learning rate
gamma = 0.6 #discount rate
epsilon = 0.1#epsilon


for i in tqdm(range(1,100001)):
    state, _ = env.reset()
    
    done = False
    
    while not done:
        
        if random.uniform(0,1)< epsilon: #explore
            env.action_space.sample()
            
        else: #exploit
            action = np.argmax(q_table[state])
            
        next_state, reward, done, info, _  = env.step(action)
        
        q_table[state,action] = q_table[state,action] + alpha *(reward + gamma*np.max(q_table[next_state])- q_table[state, action])
        
        state = next_state
    

print("Training finished")
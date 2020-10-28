#!/usr/bin/env python

"""
  Author: Adam White, Mohammad M. Ajallooeian
  Purpose: for use of Reinforcement learning course University of Alberta Fall 2017
 
  env transitions *ignore* actions, state transitions, rewards, and terminations are all random
"""

from utils import randn, randInRange, rand_un
import numpy as np

local_observation = None # local_observation: NumPy array
this_reward_observation = (None, None, None) # this_reward_observation: (floating point, NumPy array, Boolean)
nStatesSimpleEnv = 20

def env_init():
    global local_observation, this_reward_observation
    local_observation = np.zeros(1)

    this_reward_observation = (0.0, local_observation, False)


def env_start(): # returns NumPy array
    global local_observation#, this_reward_observation
    local_observation[0] = 0
    return this_reward_observation[1]

def env_step(this_action): # returns (floating point, NumPy array, Boolean), this_action: NumPy array
    global local_observation, this_reward_observation#, nStatesSimpleEnv
    episode_over = False
    the_reward = randn(0.0, 1.0) # rewards drawn from (0, 1) Gaussian

    atp1 = this_action[0] # how to extact action
    stp1 = randInRange(nStatesSimpleEnv) # state transitions are uniform random
    if rand_un() < 0.05:
        episode_over = True # termination is random

    local_observation[0] = stp1
    this_reward_observation = (the_reward, this_reward_observation[1], episode_over)

    return this_reward_observation

def env_cleanup():
    #
    return

def env_message(inMessage): # returns string, inMessage: string
    if inMessage == "what is your name?":
        return "my name is skeleton_environment!"
  
    # else
    return "I don't know how to respond to your message"

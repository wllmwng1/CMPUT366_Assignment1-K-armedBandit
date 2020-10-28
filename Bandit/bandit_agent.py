#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian
  Purpose: for use of Rienforcement learning course University of Alberta Fall 2017

  agent does *no* learning, selects actions randomly from the set of legal actions, ignores observation/state

"""

from utils import rand_in_range
import numpy as np


local_action = None # local_action: NumPy array
this_action = None # this_action: NumPy array
last_observation = None # last_observation: NumPy array

numActions = 10
Qaction = np.zeros(numActions)
for i in range(numActions):
    Qaction[i] = 5
epsilon = 0.00
alpha = 0.1
#numStates = 1


def agent_init():
    global local_action, this_action, last_observation

    local_action = np.zeros(1)
    this_action = local_action
    last_observation = np.zeros(1)

def agent_start(this_observation): # returns NumPy array, this_observation: NumPy array
    global local_action, last_observation, this_action#, numActions

    stp1 = this_observation[0] # how you convert observation to a number, if state is tabular
    atp1 = rand_in_range(numActions)
    local_action[0] = atp1

    last_observation = this_observation # save observation, might be useful on the next step
    this_action = local_action

    return this_action


def agent_step(reward, this_observation): # returns NumPy array, reward: floating point, observation_t: NumPy array
    global local_action, last_observation, this_action, numActions, Naction, Qaction

    stp1 = this_observation[0]
    # might do some learning here
    action = int(this_action[0])
    Qaction[action] = Qaction[action] + alpha*(reward - Qaction[action])

    choice = np.random.random()
    if choice <= epsilon:
        atp1 = rand_in_range(numActions)
    else:
        atp1 = np.argmax(Qaction)
    local_action[0] = atp1
    this_action = local_action
    last_observation = this_observation

    return this_action

def agent_end(reward): # reward: floating point
    # final learning update at end of episode
    return

def agent_cleanup():
    # clean up
    return

def agent_message(inMessage): # returns string, inMessage: string
    # might be useful to get information from the agent

    if inMessage == "what is your name?":
        return "my name is skeleton_agent!"
    # else
    return "I don't know how to respond to your message"

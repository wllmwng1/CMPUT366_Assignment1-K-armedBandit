#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian
  Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
  Last Modified by: Mohammad M. Ajallooeian
  Last Modified on: 12/9/2017
 
  experiment runs 200 episodes, averaging the cummulative reward per episode over 30 independent runs.
  Results are saved to file.
"""

from rl_glue import *  # Required for RL-Glue
RLGlue("simple_env", "simple_agent")

import numpy as np
import sys

def saveResults(data, dataSize, filename): # data: floating point, dataSize: integer, filename: string
    with open(filename, "w") as dataFile:
        for i in range(dataSize):
            dataFile.write("{0}\n".format(data[i]))

if __name__ == "__main__":
    numEpisodes = 200
    maxStepsInEpisode = 100
    numRuns = 30
    result = np.zeros(numEpisodes)

    print "\nPrinting one dot for every run: {0} total Runs to complete".format(numRuns)
    for k in range(numRuns):
        RL_init()

        for i in range(numEpisodes):
            #RL_episode(maxStepsInEpisodes)
            # We would normally use RL_episode to try and run a whole episode,
            # however, for the purpose of clarificationand also because we
            # might want to measure things inside an episode, we will use the
            # expanded code here (which is what is contained within RL_episode.
            
            isTerminal = False

            RL_start()

            while (not isTerminal) and ((maxStepsInEpisode == 0) or (num_steps < maxStepsInEpisode)):
                rl_step_result = RL_step()
                # RL_Step performs a step for the environment and then the
                # agent and returns a const  
                
                # Recieve the reward the agent got, Check if optimal action was
                # taken, ...
                isTerminal = rl_step_result[3]

            result[i] += RL_return()
        RL_cleanup()
        # average things
        print ".",
        sys.stdout.flush()

    print "\nDone"

    # average over runs
    for i in range(numEpisodes):
        result[i] /= numRuns

    # Save data to a file
    saveResults(result, numEpisodes, "RL_EXP_OUT.dat")

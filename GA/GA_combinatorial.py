#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 12:02:54 2024

@author: willy

Genetic Algorithm for combinatorial Quadratic Assignement Problem
"""
import random as rd
import numpy as np
import pandas as pd


###############################################################################
# 
#                   Constances : start
#
###############################################################################
P_C = 1             # Probability of Crossover
P_M = 0.3           # Probability of Mutation
K = 3               # Tournament Selection
N_POPULATION = 100    # Population per generation
M_GENERATION = 30     # Number of Generation
###############################################################################
# 
#                   Constances : ennd
#
###############################################################################




###############################################################################
# 
#                   Variables : start
#
###############################################################################
chromosome = ["D","A","C","B","G","E","F","H"]
Dist = pd.DataFrame([[0,1,2,3,1,2,3,4],
                     [1,0,1,2,2,1,2,3],
                     [2,1,0,1,3,2,1,2],
                     [3,2,1,0,4,3,2,1],
                     [1,2,3,4,0,1,2,3],
                     [2,1,2,3,1,0,1,2],
                     [3,2,1,2,2,1,0,1],
                     [4,3,2,1,3,2,1,0]], 
                    columns=["A","B","C","D","E","F","G","H"], 
                    index=["A","B","C","D","E","F","G","H"])
Flow = pd.DataFrame([[0,5,2,4,1,0,0,6],
                     [5,0,3,0,2,2,2,0],
                     [2,3,0,0,0,0,0,5],
                     [4,0,0,0,5,2,2,10],
                     [1,2,0,5,0,10,0,0],
                     [0,2,0,2,10,0,5,1],
                     [0,2,0,2,0,5,0,10],
                     [6,0,5,10,0,1,10,0]], 
                    columns=["A","B","C","D","E","F","G","H"], 
                    index=["A","B","C","D","E","F","G","H"])
###############################################################################
# 
#                   variables : ennd
#
###############################################################################

###############################################################################
# 
#                   track variables : start
#
###############################################################################
For_plotting_the_best = np.empty((0, len(chromosome)+1))

Final_Best_in_Generation_X = []
Worst_Best_in_Generation_X = []

One_Final_Guy = np.empty((0, len(chromosome)+2))
One_Final_Guy_Final = []

Min_for_all_Generation_for_Mut_1 = np.empty((0, len(chromosome)+1))
Min_for_all_Generation_for_Mut_2 = np.empty((0, len(chromosome)+1))

Min_for_all_Generation_for_Mut_1_1 = np.empty((0, len(chromosome)+2))
Min_for_all_Generation_for_Mut_2_2 = np.empty((0, len(chromosome)+2))

Min_for_all_Generation_for_Mut_1_1_1 = np.empty((0, len(chromosome)+2))
Min_for_all_Generation_for_Mut_2_2_2 = np.empty((0, len(chromosome)+2))
###############################################################################
# 
#                   track variables : ennd
#
###############################################################################


###############################################################################
# 
#                   fitness value : start
#
###############################################################################

def fitness_value(chromosome):
    """
    compute the cost of the initial solution called chromosome

    Parameters
    ----------
    chromosome : list
        DESCRIPTION.
            list of departments
    Returns
    -------
    cost of the chromosome : float.

    """
    re_dist_df = Dist.reindex(columns=chromosome, index=chromosome)
    re_flow_df = Flow.reindex(columns=chromosome, index=chromosome)
    cost_chromo = sum(sum(np.array(re_dist_df) * np.array(re_flow_df)))
    return cost_chromo
    
###############################################################################
# 
#                   fitness value : end
#
###############################################################################

###############################################################################
# 
#             generate RANDOMLY a population of N solutions : start
#
###############################################################################
def generate_population(chromosome):
    """
    shuffle the items in the chromosome vector N_population time and  

    Parameters
    ----------
    chromosome : list
        DESCRIPTION.

    Returns
    -------
    array of N chromosomes (solutions): array.

    """
    shape = (0, len(chromosome))
    n_solutions = np.empty(shape=shape)
    for i in range(int(N_POPULATION)):
        rnd_sol = rd.sample(chromosome, len(chromosome))
        n_solutions = np.vstack(tup=(n_solutions, rnd_sol))
        
    return n_solutions

###############################################################################
# 
#             generate RANDOMLY a population of N solutions : end
#
###############################################################################
if __name__ == "__main__":
    pass

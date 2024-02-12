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

import matplotlib.pyplot as plt


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


###############################################################################
# 
#             solutions for M_GENERATION : start
#
###############################################################################
def find_chromosomes_for_M_generations(chromosome):
    """
    look for solutions=chromosomes after M generations   

    Parameters
    ----------
    chromosome : list
        DESCRIPTION.

    Returns
    -------
    : array.

    """
    n_solutions = generate_population(chromosome)
    
    
    shape = (0, len(chromosome))
    for m_generation in range(M_GENERATION):
        
        New_population = np.empty(shape) # save the new generation after N_population testing
        
        All_in_Generation_X_1 = np.empty((0, len(chromosome)+1))
        All_in_Generation_X_2 = np.empty((0, len(chromosome)+1))
        
        Min_in_Generation_X_1 = []
        Min_in_Generation_X_2 = []
        
        Save_Best_in_Generation_X = np.empty((0, len(chromosome)+1))
        Final_Best_in_Generation_X = []
        Worst_Best_in_Generation_X = []
        
        print(f"---> GENERATION : # {m_generation}")
        
        Family = 1
        
        for n_population in range( int(N_POPULATION/2) ):
            
            print(f"---> Family : # {Family}")
            
            Parent_1, Parent_2, Child_1, Child_2, \
            Mutated_Child_1, Mutated_Child_2, \
            Total_Cost_Mut_1, Total_Cost_Mut_2 \
                = find_chromosomes_for_N_equal_1_population(Family, n_population, 
                                                            chromosome, n_solutions)
            
                
            All_in_Generation_X_1_1_temp = Mutated_Child_1[np.newaxis]
            All_in_Generation_X_1_1 = np.column_stack(tup=(Total_Cost_Mut_1, 
                                                           All_in_Generation_X_1_1_temp ))
            
            All_in_Generation_X_2_1_temp = Mutated_Child_2[np.newaxis]
            All_in_Generation_X_2_1 = np.column_stack(tup=(Total_Cost_Mut_2, 
                                                           All_in_Generation_X_2_1_temp ))
                    
            All_in_Generation_X_1 = np.vstack((All_in_Generation_X_1, All_in_Generation_X_1_1 ))
            All_in_Generation_X_2 = np.vstack((All_in_Generation_X_2, All_in_Generation_X_2_1 ))
            
            
            Save_Best_in_Generation_X = np.vstack((All_in_Generation_X_1, 
                                                   All_in_Generation_X_2))
            
            New_population = np.vstack((New_population, Mutated_Child_1, Mutated_Child_2))
        
        
            t = 0
            R_1 = []
            for i in All_in_Generation_X_1:
                
                if (All_in_Generation_X_1[t,:1]) <= min(All_in_Generation_X_1[:,:1]):
                    R_1 = All_in_Generation_X_1[t,:]
                t = t+1
                
            Min_in_Generation_X_1 = R_1[np.newaxis]
            
            
            t = 0
            R_2 = []
            for i in All_in_Generation_X_2:
                
                if (All_in_Generation_X_2[t,:1]) <= min(All_in_Generation_X_2[:,:1]):
                    R_2 = All_in_Generation_X_2[t,:]
                t = t+1
                
            Min_in_Generation_X_2 = R_2[np.newaxis]
            
            Family = Family+1
            
        t = 0
        R_final = []
        
        for i in Save_Best_in_Generation_X:
            
            if (Save_Best_in_Generation_X[t,:1]) <= min(Save_Best_in_Generation_X[:,:1]):
                R_final = Save_Best_in_Generation_X[t, :]
            t=t+1
            
        Final_Best_in_Generation_X = R_final[np.newaxis]
        
        
        For_Plotting_the_Best = np.vstack((For_Plotting_the_Best, 
                                           Final_Best_in_Generation_X))
        
        t = 0
        R_22_Final = []
        
        for i in Save_Best_in_Generation_X:
            
            if (Save_Best_in_Generation_X[t, :1]) >= max(Save_Best_in_Generation_X[:,:1]):
                R_22_Final = Save_Best_in_Generation_X[t, :]
            t = t+1
            
        Worst_Best_in_Generation_X = R_22_Final[np.newaxis]
        
        
        
        # Elitism, the best in the generation lives
        # Elitism, the best in the generation lives
        # Elitism, the best in the generation lives
        
        Darwin_Guy = Final_Best_in_Generation_X[:]
        Not_So_Darwin_Guy = Worst_Best_in_Generation_X[:]
        
        Darwin_Guy = Darwin_Guy[0:, 1:].tolist()
        Not_So_Darwin_Guy = Not_So_Darwin_Guy[0:,1:].tolist()
        
        
        Best_1 = np.where((New_population == Darwin_Guy).all(axis=1))
        Worst_1 = np.where((New_population == Not_So_Darwin_Guy).all(axis=1))
        
        
        New_population[Worst_1] = Darwin_Guy
        
        
        n_solutions = New_population
        
        
        Min_for_all_Generation_for_Mut_1 = np.vstack((Min_for_all_Generations_for_Mut_1, Min_in_Generation_X_1))
        Min_for_all_Generation_for_Mut_2 = np.vstack((Min_for_all_Generations_for_Mut_2, Min_in_Generation_X_2))
        
        
        Min_for_all_Generation_for_Mut_1_1 = np.insert(Min_in_Generation_X_1, 0, m_generation)
        Min_for_all_Generation_for_Mut_2_2 = np.insert(Min_in_Generation_X_2, 0, m_generation)
        
        
        Min_for_all_Generation_for_Mut_1_1_1 = np.vstack((Min_for_all_Generations_for_Mut_1_1_1, Min_for_all_Generation_for_Mut_1_1))
        Min_for_all_Generation_for_Mut_2_2_2 = np.vstack((Min_for_all_Generations_for_Mut_2_2_2, Min_for_all_Generation_for_Mut_2_2))
        
        
        m_generation = m_generation+1
        
        
    One_Final_Guy = np.vstack((Min_for_all_Generation_for_Mut_1_1_1, 
                               Min_for_all_Generation_for_Mut_2_2_2))
    
    t = 0
    Final_Here = []
    for i in One_Final_Guy:
        if (One_Final_Guy[t,1]) <= min(One_Final_Guy[:,1]):
            Final_2 = []
            Final_2 = [One_Final_Guy[t,1]]
            Final_Here = One_Final_Guy[t, :]
        t = t+1
        
    A_2_Final = min(One_Final_Guy[:,1])
    
    One_Final_Guy_Final = Final_Here[np.newaxis]
    
    print()
    print("Min in all Generation : {One_Final_Guy_Final}")
    print("The Lowest Cost is : {One_Final_Guy_Final[:,1]}")
    
    Look = (One_Final_Guy_Final[:,1]).tolist()
    Look = float(Look[0])
    Look = int(Look)
    
    plt.plot(For_Plotting_the_Best[:,0])
    plt.axline(y=Look, color='r', linestyle="--")
    plt.title("Cost Reached Through Generations", fontsize=20, fontweight="bold")
    plt.xlabel("Generations", fontsize=18, fontweight="bold")
    plt.ylabel("Cost (Flow*Distance)", fontsize=18, fontweight="bold")
    plt.xticks(fontweight="bold")
    plt.yticks(fontweight="bold")
    xyz = (M_GENERATION/2, Look)
    xyzz = (M_GENERATION/1.95, 1)
    
    plt.annotate(f"Minimum Reached  at {Look}", xy=xyz, xytest=xyzz, 
                 arrowprops=dict(facecolor="black", shrink=0.001, width=1, headwidth=5),
                 fontsize=12, fontweight="bold")
    plt.show()
    
    print()
    print(f"Initial Solution: {chromosome}")
    print(f"Final Solution: {One_Final_Guy_Final[:,2:]}")
    print(f"The Lowest cost is : {One_Final_Guy_Final[:,1]}")
    print(f"At Generation {One_Final_Guy_Final[:,0]}")
    print()
    
    print("### METHODS ###")
    print("# Selection Method = Tournament Selection #")
    print("# Crossover = C1 (order) but 2-point selection #")
    print("# Mutate = #1- Inverse #")
    print("# Other = Elitism #")
    print("### METHODS ###")
    
    
    
    
            
    

###############################################################################
# 
#            solutions for M_GENERATION : end
#
###############################################################################

###############################################################################
# 
#             solutions for N_POPULATION : start
#
###############################################################################
def get_Tournament_selection(chromosome, n_solutions):
    """
    get the 3 best solutions into the N SOLUTIONS and infer the 2 parents 

    Returns
    -------
    Parent array : Array, shape=(2, len(chromosome)).

    """
    Parents = np.empty(shape=(0, len(chromosome)))
    
    for i in range(2):
        
        Battle_Troops = []
        
        Warrior_1_index = np.random.randint(0, len(n_solutions))
        Warrior_2_index = np.random.randint(0, len(n_solutions))
        Warrior_3_index = np.random.randint(0, len(n_solutions))
        
        
        # Warrior_1_index != Warrior_2_index != Warrior_3_index
        
        while Warrior_1_index == Warrior_2_index:
            Warrior_1_index = np.random.randint(0, len(n_solutions))
        while Warrior_2_index == Warrior_3_index:
            Warrior_2_index = np.random.randint(0, len(n_solutions))
        while Warrior_3_index == Warrior_1_index:
            Warrior_3_index = np.random.randint(0, len(n_solutions))
    
        
        Warrior_1 = n_solutions[Warrior_1_index, :]
        Warrior_2 = n_solutions[Warrior_2_index, :]
        Warrior_3 = n_solutions[Warrior_3_index, :]
        
        Battle_Troops = [Warrior_1, Warrior_2, Warrior_3]
        
        
        # for Warrior #1
        Prize_warrior_1 = fitness_value(Warrior_1)
        
        # for Warrior #2
        Prize_warrior_2 = fitness_value(Warrior_2)
        
        # for Warrior #3
        Prize_warrior_3 = fitness_value(Warrior_3)
        
        
        Winner = None
        # find the best one between warrior 1, 2, 3
        if Prize_warrior_1 == min(Prize_warrior_1, 
                                  Prize_warrior_2, 
                                  Prize_warrior_3):
            Winner = Prize_warrior_1
        elif Prize_warrior_2 == min(Prize_warrior_1, 
                                  Prize_warrior_2, 
                                  Prize_warrior_3):
            Winner = Prize_warrior_2
        else:
            Winner = Prize_warrior_3
        
        
        
        Parents = np.vstack((Parents, Winner))
      
    return Parents
    
def find_Crossover(Parent_1, Parent_2, chromosome):
    """
    make the crossover between Parent_1, Parent_2 for getting both to children

    Returns
    -------
    Child_1, Child_2: array, array.

    """
    Child_1 = np.empty(shape=(0, len(chromosome)))
    Child_2 = np.empty(shape=(0, len(chromosome)))
    
    Ran_CO_1 = np.random.rand()
    
    if Ran_CO_1 < P_C:
        
        # Choose  two random numbers to crossover with their locations
        Cr_1 = np.random.randint(0, len(chromosome))
        Cr_2 = np.random.randint(0, len(chromosome))
        
        while Cr_1 == Cr_2:
            Cr_2 = np.random.randint(0, len(chromosome))
        
        if Cr_1 < Cr_2:
            Cr_2 = Cr_2 + 1
            
            New_Dep_1 = Parent_1[Cr_1:Cr_2]     # Mid seg. of Parent #1
            New_Dep_2 = Parent_2[Cr_1:Cr_2]     # Mid seg. of Parent #2
            
            First_Seg_1 = Parent_1[:Cr_1]       # first seg. of Parent #1
            First_Seg_2 = Parent_1[:Cr_1]       # first seg. of Parent #2
        
            Temp_First_Seg_1_1 = []             # Temporary for first seg
            Temp_Second_Seg_2_2 = []            # Temporary for second seg
        
            Temp_First_Seg_3_3 = []             # Temporary for first seg
            Temp_Second_Seg_4_4 = []            # Temporary for second seg
            
            
            for i in First_Seg_2:                                           # for i in all the elements of the first segment 
                if i not in New_Dep_1:                                      # if they aren't in seg. parent #1
                    Temp_First_Seg_1_1 = np.append(Temp_First_Seg_1_1, i)   # append them
                    
            Temp_New_Vector_1 = np.concatenate((Temp_First_Seg_1_1, New_Dep_1))
            
            for i in Parent_2:                                              # For Parent #2
                if i not in Temp_New_Vector_1:                              # If not in what is made so far
                    Temp_Second_Seg_2_2 = np.append(Temp_Second_Seg_2_2, i) # Append it
                    
            Child_1 = np.concatenate((Temp_First_Seg_1_1, New_Dep_1, Temp_Second_Seg_2_2))
            
            for i in First_Seg_1:                                           # Do the same for child #2
                if i not in New_Dep_2:
                    Temp_First_Seg_3_3 = np.append(Temp_First_Seg_3_3, i)
                    
            Temp_New_Vector_2 = np.concatenate((Temp_First_Seg_3_3, New_Dep_2))
            
            for i in Parent_1:
                if i not in Temp_New_Vector_2:
                    Temp_Second_Seg_4_4 = np.append(Temp_Second_Seg_4_4, i)
                    
            Child_2 = np.concatenate((Temp_First_Seg_3_3, New_Dep_2, Temp_Second_Seg_4_4))
            
        else:   # the same in reverse of  Cr_1 and Cr_2
            Cr_1 = Cr_1 + 1
            
            New_Dep_1 = Parent_1[Cr_2:Cr_1]
            New_Dep_2 = Parent_2[Cr_2:Cr_1]
            
            First_Seg_1 = Parent_1[:Cr_2]
            First_Seg_2 = Parent_2[:Cr_2]
            
            Temp_First_Seg_1_1 = []
            Temp_Second_Seg_2_2 = []
            
            Temp_First_Seg_3_3 = []
            Temp_Second_Seg_4_4 = []
            
            for i in First_Seg_2:
                if i not in New_Dep_1:
                    Temp_First_Seg_1_1 = np.append(Temp_First_Seg_1_1, i)
                    
            Temp_New_Vector_1 = np.concatenate((Temp_First_Seg_1_1, New_Dep_1))
            
            for i in Parent_2:
                if i not in Temp_New_Vector_1:
                    Temp_Second_Seg_2_2 = np.append(Temp_Second_Seg_2_2, i)
                    
            Child_1 = np.concatenate((Temp_First_Seg_1_1, New_Dep_1, Temp_Second_Seg_2_2))
            
            for i in First_Seg_1:
                if i not in New_Dep_2:
                    Temp_First_Seg_3_3 = np.append(Temp_First_Seg_3_3, i)
                    
            Temp_New_Vector_2 = np.concatenate((Temp_First_Seg_3_3, New_Dep_2))
            
            for i in Parent_1:
                if i not in Temp_New_Vector_2:
                    Temp_Second_Seg_4_4 = np.append(Temp_Second_Seg_4_4, i)
                    
            Child_2 = np.concatenate((Temp_First_Seg_3_3, New_Dep_2, Temp_Second_Seg_4_4)) 
            
    
    else:   # if random number was above P_C
    
        Child_1 = Parent_1
        Child_2 = Parent_2
        
    return Child_1, Child_2

def mutate_child(Child_1, chromosome):
    """
    mutate child

    Returns
    -------
    child: list.

    """
    Mutated_Child_1 = []
    
    
    Ran_Mut_1 = np.random.rand()                                                # probability to mutate
    Ran_Mut_2 = np.random.randint(0, len(chromosome))
    Ran_Mut_3 = np.random.randint(0, len(chromosome))
    
    A1 = Ran_Mut_2
    A2 = Ran_Mut_3
    
    while A1 == A2:
        A2 = np.random.randint(0, len(chromosome))
        
    if Ran_Mut_1 < P_M:                                                         # if probability to mutate is less than P_M, then mutate
        if A1 < A2:
            M_Child_1_Pos_1 = Child_1[A1]                                       # take the index
            M_Child_1_Pos_2 = Child_1[A2]                                       # take the index
            A2 = A2+1
            Rev_1 = Child_1[:]                                                  # Copy the child 1
            Rev_2 = list(reversed(Child_1[A1:A2]))                              # reverse the order
            t = 0
            for i in range(A1, A2):
                Rev_1[i] = Rev_2[t]                                             # The reversed will become instead of 
                t = t+1
                
            Mutated_Child_1 = Rev_1
            
        else:
            M_Child_1_Pos_1 = Child_1[A2]
            M_Child_1_Pos_2 = Child_1[A1]
            A1 = A1+1
            Rev_1 = Child_1[:]
            Rev_2 = list(reversed(Child_1[A2:A1]))
            t = 0
            for i in range(A2, A1):
                Rev_1[i] = Rev_2[t]                                             # The reversed will become instead of 
                t = t+1
                
            Mutated_Child_1 = Rev_1
            
    else:
        Mutated_Child_1 = Child_1
        
    return Mutated_Child_1


def find_chromosomes_for_N_equal_1_population(Family, n_population, 
                                              chromosome, n_solutions):
    """
    look for best solutions=chromosomes for one possible solution   

    Parameters
    ----------
    chromosome : list
        DESCRIPTION.

    : array
        DESCRIPTION:
            

    Returns
    -------
    : array.

    """
    print(f"---> Family : # {Family}, n_population : #{n_population}")
    
    
    # Tournament Selection
    # Tournament Selection
    # Tournament Selection
    
    Parents = np.empty(shape=(0, len(chromosome)))
    
    Parents = get_Tournament_selection(chromosome=chromosome, 
                                       n_solutions=n_solutions)
    
    Parent_1 = Parents[0]
    Parent_2 = Parents[1]
        
    # Crossover
    # Crossover
    # Crossover
    Child_1, Child_2 = None, None
    Child_1, Child_2 = find_Crossover(Parent_1, Parent_2, chromosome)
        
        
    # Mutation Child #1 and #2
    # Mutation Child #1 and #2
    # Mutation child #1 and #2
    
    Mutated_Child_1, Mutated_Child_2 = None, None
    Mutated_Child_1 = mutate_child(Child_1, chromosome)
    Mutated_Child_2 = mutate_child(Child_2, chromosome)
    
    
    
    Total_Cost_Mut_1 = fitness_value(Mutated_Child_1)
    Total_Cost_Mut_2 = fitness_value(Mutated_Child_2)
    
    print()
    print(f"FV at Mutated Child #1 at Gen {n_population} : {Total_Cost_Mut_1}")
    print(f"FV at Mutated Child #2 at Gen {n_population} : {Total_Cost_Mut_2}")
    
    return Parent_1, Parent_2, Child_1, Child_2, \
            Mutated_Child_1, Mutated_Child_2, Total_Cost_Mut_1, Total_Cost_Mut_2
    
    
def find_chromosomes_for_N_population(Family, chromosome, n_solutions):
    """
    look for best solutions=chromosomes for N possible solutions   

    Parameters
    ----------
    chromosome : list
        DESCRIPTION.

    New_population: array
        DESCRIPTION:
            

    Returns
    -------
    : array.

    """
    shape = (0, len(chromosome))
    for n_population in range( int(N_POPULATION) ):
        
        print(f"---> Family : # {Family}")
        
        Parent_1, Parent_2, Child_1, Child_2, \
        Mutated_Child_1, Mutated_Child_2, \
        Total_Cost_Mut_1, Total_Cost_Mut_2 \
            = find_chromosomes_for_N_equal_1_population(Family, n_population, 
                                                        chromosome, n_solutions)
        
            
        All_in_Generation_X_1_1_temp = Mutated_Child_1[np.newaxis]
        All_in_Generation_X_1_1 = np.column_stack(tup=(Total_Cost_Mut_1, 
                                                       All_in_Generation_X_1_1_temp ))
        
        All_in_Generation_X_2_1_temp = Mutated_Child_2[np.newaxis]
        All_in_Generation_X_2_1 = np.column_stack(tup=(Total_Cost_Mut_2, 
                                                       All_in_Generation_X_2_1_temp ))
                
        All_in_Generation_X_1 = np.vstack((All_in_Generation_X_1, All_in_Generation_X_1_1 ))
        All_in_Generation_X_2 = np.vstack((All_in_Generation_X_2, All_in_Generation_X_2_1 ))
        
        
        Save_Best_in_Generation_X = np.vstack((All_in_Generation_X_1, ))
                

###############################################################################
# 
#            solutions for N_POPULATION : end
#
###############################################################################


if __name__ == "__main__":
    pass

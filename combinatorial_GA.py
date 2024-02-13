#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 10:57:26 2023

@author: willy
"""

import numpy as np
import random as rd

###############################################################################
#
#               Objectif function   --->    start
#
###############################################################################

"""
how to decoding a chromosome
decoding = sum_{i=0}^{lx} x_i * 2^{i} * precision + lower_x
with :
- bit == gene = x_i \in {0,1}
- precision = \frac{Upper_x - lower_x}{2^{l} - 1}
- range = [Upper_x, lower_x]
exple of chromosome :
        Y         X
    [1,1,0,1,   0,  1,  1,  0]
                2³, 2², 2¹, 2⁰
    
"""
## testing 
chromosome_test = np.array([1,1,0,1, # y gene = y variable
                            0,1,1,0]) # x gene = x variable

def define_length_gene_x(chromosome_x):
    """
    return the length of gene x

    Parameters
    ----------
    chromosome : np.array
        DESCRIPTION.

    Returns
    -------
    integer.

    """
    return len(chromosome_x) // 2

def compute_sum_gene_x(chromosome_x, length_x, gene_name="gene_x"):
    """
    gene_name = {"gene_x", "gene_y"}
    """
    x_gene_sum = 0
    t = 1
    if gene_name == "gene_x":
        t = 1
        length_xy = define_length_gene_x(chromosome_x)
        for i in range(t, length_xy+1):
            x_gene_sum += chromosome_x[-i] * 2**(i-1)
            # print(f"X i={i}, {chromosome_x[-i] * 2**(i-1)}")
    else:   
        t = 1 + define_length_gene_x(chromosome_x)
        length_xy = len(chromosome_x);
        
        cpt = 0
        for i in range(t, length_xy+1):
            # x_gene_sum += chromosome_x[-i] * 2**(i-1)
            x_gene_sum += chromosome_x[-i] * 2**(cpt)
            cpt += 1
            #  print(f"Y i={i}, {chromosome_x[-i] * 2**(i)}")
        
    return x_gene_sum

def test_compute_sum_gene_x():
    chromosome_x = np.array([0,1,1,0])
    length_x = len(chromosome_x)
    gene_name = "gene_x"
    x_gene_sum = compute_sum_gene_x(chromosome_x, length_x, gene_name)
    
    print(f"x_gene_sum = {x_gene_sum}")
    assert x_gene_sum == 6, "sum of gene different to 6"
    
def precision_gene_x(upper_x, lower_x, length_x):
    """
    precision = \frac{Upper_x - lower_x}{2^{l} - 1}
    """
    return (upper_x - lower_x)/(2**length_x - 1)

def test_precision_gene_x():
    upper_x = 6; lower_x = -6; length_x = 9
    precision = precision_gene_x(upper_x, lower_x, length_x)
    
    print(f"precision = {precision:.5f}")
    
    assert round(precision,5) == 0.02348, "precision different to 0.02348"

def compute_decoding_x(chromosome_x, length_x, upper_x, lower_x, gene_name="gene_x"):
    """
    decoding_x is the fitnss value of X
    return a float value

    length_x : the length of chromosome x
    """
    
    x_gene_sum = compute_sum_gene_x(chromosome_x, length_x, gene_name)
    x_gene_precision = precision_gene_x(upper_x, lower_x, length_x)
     
    x_gene_decoding = x_gene_sum * x_gene_precision + lower_x
    
    print(f'x_gene_sum={x_gene_sum}, x_gene_precision={x_gene_precision}, x_gene_decoding={x_gene_decoding}')
    return x_gene_decoding

def test_compute_decoding_x():
    upper_x = 6; lower_x = -6; length_x = 9;
    chromosome_x = np.array([1,1,0,1,0,
                             0,0,1,0])
    length_x = len(chromosome_x) // 2
    gene_name = "gene_x"
    
    decoding_x = compute_decoding_x(chromosome_x, length_x, upper_x, lower_x, gene_name)
    print(f'decoding_x = {round(decoding_x, 5)}')
    
    assert round(decoding_x, 5) == 3.81605, "decoding_x different to 3.81605" 

def test_compute_decoding_y():
    upper_x = 6; lower_x = -6;
    chromosome_y = np.array([1,1,0,1, # y gene = y variable
                             0,1,1,0])
    length_y = len(chromosome_y) // 2
    gene_name = "gene_y"
    
    decoding_y = compute_decoding_x(chromosome_y, length_y, upper_x, lower_x, gene_name)
    print(f'decoding_y = {round(decoding_y, 5)}')
    
    assert round(decoding_y, 5) == 13.57647, "decoding_x different to 13.57647" 

def test_compute_decoding_xy():
    upper_x = 6; lower_x = -6;
    chromosome_y = np.array([1,1,0,1,1,0,0,1,1,0,0,1, # y gene = y variable
                             0,1,1,0,1,0,1,0,1,1,0,0])
    
    length_y = len(chromosome_y)//2
    gene_name = "gene_y"
    
    decoding_y = compute_decoding_x(chromosome_y, length_y, upper_x, lower_x, 
                                    gene_name=gene_name)
    decoding_x = compute_decoding_x(chromosome_y, length_y, upper_x, lower_x, 
                                    gene_name="gene_x")
    
    print(f"decoding_y = {round(decoding_y, 5)}, decoding_x = {round(decoding_x, 5)},")

def himmelblau_function(decoding_x, decoding_y):
    """
    this function is used for continuous optimization problem
    
    with vector 
    shape: x=(1, N), y=(1, N)
    """
    return pow((decoding_x*decoding_x) + decoding_y - 11, 2) \
            + pow(decoding_x + (decoding_y*decoding_y) - 7, 2)
    
def objective_value(chromosome, upper, lower):
    """
    compute fitness value for the chromosome of 0s and 1s

    Parameters
    ----------
    chromosome : TYPE
        DESCRIPTION.

    Returns
    -------
    a 3-tuple
    float value, flaot value, a float_value.

    """
    length_x = len(chromosome)//2
    length_y = len(chromosome)//2
    upper_x, lower_x = upper, lower
    upper_y, lower_y = upper, lower
    
    decoding_y = compute_decoding_x(chromosome, length_y, 
                                    upper_y, lower_y, 
                                    gene_name="gene_y")
    decoding_x = compute_decoding_x(chromosome, length_x, 
                                    upper_x, lower_x, 
                                    gene_name="gene_x")
    
    # the himmelblau function
    obj_function_value = himmelblau_function(decoding_x, decoding_y)
    
    return decoding_x, decoding_y, obj_function_value
    

def test_objective_value():
    print("######################################################")
    print("#   Testing a objective function #")
    print("######################################################")
    upper, lower = 6, -6
    
    
    print("#   Parent_1 obj value   #")
    chromosome = np.array([1,1,0,1,1,0,0,1,1,0,0,1, # y gene = y variable
                           0,1,1,0,1,0,1,0,1,1,0,0])
    
    decoding_x, decoding_y, obj_func_value \
        = objective_value(chromosome, upper, lower)
    
    assert round(obj_func_value, 4) == 126.8975, \
            "objective value different to 126.8975"
            
    print(f"Parent_1 : obj func val = {obj_func_value}, decoding_x = {decoding_x}, \
              decoding_y = {decoding_y}")
            
    print("#   Parent_2 obj value   #")
    parent_2 = np.array([1,0,1,1,1,1,0,1,1,0,1,1, # y gene = y variable
                         1,1,1,1,0,0,1,1,1,1,0,0])
    
    decoding_x, decoding_y, obj_func_value \
        = objective_value(parent_2, upper, lower)
        
    print(f"Parent_2 : obj func val = {obj_func_value}, decoding_x = {decoding_x}, \
              decoding_y = {decoding_y}")
    
    assert round(obj_func_value, 4) == 502.6585, \
            "objective value different to 502.6585"
    
###############################################################################
#
#               Objectif function   --->    end
#
###############################################################################

###############################################################################
#
#               Selecting Parents   --->    begin
#
###############################################################################
def find_parents_ts(chromosome, population=20, K=3, upper=6, lower=-6):

#def selecting_2_parents(chromosome, population=20, K=3, upper=6, lower=-6):
    """
    find 2 parents from the pool of solutions using the tournament selection method

    Parameters
    ----------
    population : TYPE
        DESCRIPTION.
    chromosome : TYPE
        DESCRIPTION.
    K : integer
        DESCRIPTION.    
        for tournament selection, selecting the K best solutions

    Returns
    -------
    None.

    """
    # create an empty array to place the initial random solutions
    all_solutions = np.empty((0, len(chromosome)))
    
    for i in range(population):
        rd.shuffle(chromosome)
        all_solutions = np.vstack( (all_solutions, chromosome) )
        
    # create an empty array to place the slected parents
    parents = np.empty( (0, np.size(all_solutions, 1) ) )
    
    for i in range(2):
    
        # select K=3 random parents from the pool of solutions you have
        ### get  K=3 random integers (K=3 for tournament selection)
        indices_list = np.random.choice(len(all_solutions), 
                                        3,
                                        replace=False)
        print(f"\n \n round #{i+1} - indices: {indices_list}")
        
        ### get the 3 possible parents for selection from "all solutions"
        possible_parent_1 = all_solutions[indices_list[0]]
        possible_parent_2 = all_solutions[indices_list[1]]
        possible_parent_3 = all_solutions[indices_list[2]]
        
        print()
        print("chromosome #1 is {possible_parent_1}")
        print("chromosome #2 is {possible_parent_2}")
        print("chromosome #3 is {possible_parent_3}")
        
        ### get objective function value (fitness value) for each possible parent
        ### index 2 indicate the 3rd item of returning objective_value function
        obj_func_value_parent1 = objective_value(possible_parent_1, 
                                                 upper, lower)
        obj_func_value_parent2 = objective_value(possible_parent_2, 
                                                 upper, lower)
        obj_func_value_parent3 = objective_value(possible_parent_3, 
                                                 upper, lower)
        
        ### fv is a fitness value == objective function value
        print(f"fv chromosome #1 = {obj_func_value_parent1[2]}")
        print(f"fv chromosome #2 = {obj_func_value_parent2[2]}")
        print(f"fv chromosome #3 = {obj_func_value_parent3[2]}")
        
        ### find which parent is the best
        min_obj_func = min(obj_func_value_parent1[2], 
                           obj_func_value_parent2[2], 
                           obj_func_value_parent3[2])
        
        selected_parent = None
        if min_obj_func == obj_func_value_parent1[2]:
            selected_parent = possible_parent_1
        elif min_obj_func == obj_func_value_parent2[2]:
            selected_parent = possible_parent_2
        else:
            selected_parent = possible_parent_3
            
        print()
        print(f"winner is {selected_parent}")
        print(f"min fv is {min_obj_func}")
        
        ### put the selected parent in the empty array we created above
        parents = np.vstack((parents, selected_parent))
    
    ### after doing the above 2 times, so for i in range(2)
    parent_1 = parents[0,:] # parent_1, first element in the array
    parent_2 = parents[1,:] # parent_2, second element in the array
    
    obj_func_1 = objective_value(parent_1, upper, lower)[2]
    obj_func_2 = objective_value(parent_2, upper, lower)[2]
    
    print("##################################################################")
    print("##################################################################")
    print("##################################################################")

    print()
    print(f"chosen parent #1 : {parent_1}")
    print(f"chosen parent #1 fv: {obj_func_1}")
    
    print(f"chosen parent #2 : {parent_2}")
    print(f"chosen parent #2 fv: {obj_func_2}")

    return parent_1, parent_2    

def test_selecting_2_parents():
    upper, lower = 6, -6
    chromosome = np.array([0,0,1,1,1,0, # y gene = y variable
                           1,1,1,0,1,1]) # x gene
    parent_1, parent_2 = find_parents_ts(chromosome, 
                                            population=20, 
                                            K=3, 
                                            upper=upper, 
                                            lower=lower)

    print(f"chosen parent #1 : {parent_1}")
    print(f"chosen parent #2 : {parent_2}")
    
            
             
###############################################################################
#
#               Selecting Parents   --->    end
#
###############################################################################

###############################################################################
#
#                       Crossover   --->    start
#
###############################################################################
def crossover_operator_2point(parent_1, parent_2, index_1, index_2):
    """
     create children from 2-point-crossover
     operator : 2-point

    Parameters
    ----------
    parent_1 : TYPE
        DESCRIPTION.
    parent_2 : TYPE
        DESCRIPTION.
    index_1 : TYPE
        DESCRIPTION.
    index_2 : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    ### for parent_1 ###
    ### for parent_1 ###
    ### for parent_1 ###
    print()
    print()
    print(f"parent_1: {parent_1}")
    
    # first_seg_parent_1: -->
    # for parent_1: the genes from the beginning of parent_1 to the 
    #       beginning of the middle segment of parent_1
    first_seg_parent_1 = parent_1[:index_1]
    print(f"first_seg_parent_2: {first_seg_parent_1}")
    
    # middle_segment: where the crossover will happen
    # for parent_1: the genes from the index chosen for parent_1 to 
    #       the middle chosen for parent_1
    mid_seg_parent_1 = parent_1[index_1:index_2+1]
    print(f"mid_seg_parent_1: {mid_seg_parent_1}")
    
    # last_seg_parent_1: -->
    # for parent_1: the genes from the end of the middle segment of 
    #       parent_1 to the last gene of parent_1
    last_seg_parent_1 = parent_1[index_2+1:]
    print(f"last_seg_parent_1: {last_seg_parent_1}")
    
    
    ### for parent_2 ###
    ### for parent_2 ###
    ### for parent_2 ###
    print()
    print()
    print(f"parent_2: {parent_2}")
    
    # first_seg_parent_2 --> same as parent_1
    first_seg_parent_2 = parent_2[:index_1]
    print(f"first_seg_parent_2: {first_seg_parent_2}")
    
    # first_seg_parent_2 --> same as parent_1
    mid_seg_parent_2 = parent_2[index_1:index_2+1]
    print(f"mid_seg_parent_2: {mid_seg_parent_2}")
    
    # first_seg_parent_2 --> same as parent_1
    last_seg_parent_2 = parent_2[index_2+1:]
    print(f"last_seg_parent_2: {last_seg_parent_2}")
    
    
    ### creating child_1 ###
    ### creating child_1 ###
    ### creating child_1 ###
    
    # the first segment from parent_1
    # plus the middle segment from parent_2
    # plus the last segment from parent_1
    child_1 = np.concatenate((first_seg_parent_1, 
                              mid_seg_parent_2, 
                              last_seg_parent_1))
    
    print()
    print()
    print(f"child_1: {child_1}")
    
    ### creating child_2 ###
    ### creating child_2 ###
    ### creating child_2 ###
    
    # the first segment from parent_2
    # plus the middle segment from parent_1
    # plus the last segment from parent_2
    child_2 = np.concatenate((first_seg_parent_2, 
                              mid_seg_parent_1, 
                              last_seg_parent_2))
    
    print()
    print()
    print(f"child_2: {child_2}")
    
    return child_1, child_2
    
def crossover(parent_1, parent_2, prob_crossover = 1):
    """
    crossover between the 2 parents to create 2 parents:
        we use 2-point crossover

    Parameters
    ----------
    parent_1 : np.array
        DESCRIPTION.
        chromosome
    parent_2 : np.array
        DESCRIPTION.
        chromosome
    prob_crossover : float, optional
        DESCRIPTION. The default is 1.
         
    Returns
    -------
    None.

    """
    
    child_1 = np.empty((0, len(parent_1)))
    child_2 = np.empty((0, len(parent_2)))
    
    rand_num_2_crossover_or_not = np.random.rand()
    
    print(f"random number is {rand_num_2_crossover_or_not}")
    
    if rand_num_2_crossover_or_not < prob_crossover:
        
        index_1 = np.random.randint(0, len(parent_1))
        index_2 = np.random.randint(0, len(parent_2))
        
        # get different indices
        # to make sure you will crossover at least one gene 
        # because it's the 2-point crossover method
        while index_1 == index_2:
            index_2 = np.random.randint(0, len(parent_1))
            
        print()
        print(f"index_1 = {index_1}, \n index_2 = {index_2}")
        print()
        print(f"parent_1 = {parent_1}, \n parent_2 = {parent_2}")
    
        # if the index from parent_1 comes before parent_2
        """
        e.g. parent_1 = 0,1,>>1<<,1,0,0,1,0  --> index = 2
             parent_2 = 0,0,1,0,0,1,>>1<<,1  --> index = 6
             
        e.g. parent_1 = 0,1,1,1,0,0,>>1<<,0  --> index = 6 xx_wrong_xx
             parent_2 = 0,0,>>1<<,0,0,1,1,1  --> index = 2 xx_wrong_xx
        """
        
        if index_1 < index_2:
            
            child_1, child_2 = \
                crossover_operator_2point(
                   parent_1 = parent_1, 
                   parent_2 = parent_2, 
                   index_1 = index_1, 
                   index_2 = index_2)
              
            pass
        else:
            child_1, child_2 = \
                crossover_operator_2point(
                   parent_1 = parent_1, 
                   parent_2 = parent_2, 
                   index_1 = index_2, 
                   index_2 = index_1)
            
            
        
    # when we will not crossover
    # when rand_num_2_crossover_or_not is not LESS (is greater) than prob_crossover
    # when prob_crossover == 1, then rand_num_2_crossover_or_not will always be less
    #           than prob_crossover
    else:
        child_1 = parent_1
        child_2 = parent_2
        
    return child_1, child_2


def crossover_OLD(parent_1, parent_2, prob_crossover = 1):
    """
    crossover between the 2 parents to create 2 parents:
        we use 2-point crossover

    Parameters
    ----------
    parent_1 : np.array
        DESCRIPTION.
        chromosome
    parent_2 : np.array
        DESCRIPTION.
        chromosome
    prob_crossover : float, optional
        DESCRIPTION. The default is 1.
         
    Returns
    -------
    None.

    """
    
    child_1 = np.empty((0, len(parent_1)))
    child_2 = np.empty((0, len(parent_2)))
    
    rand_num_2_crossover_or_not = np.random.rand()
    
    print(f"random number is {rand_num_2_crossover_or_not}")
    
    if rand_num_2_crossover_or_not < prob_crossover:
        
        index_1 = np.random.randint(0, len(parent_1))
        index_2 = np.random.randint(0, len(parent_2))
        
        # get different indices
        # to make sure you will crossover at least one gene 
        # because it's the 2-point crossover method
        while index_1 == index_2:
            index_2 = np.random.randint(0, len(parent_1))
            
        print()
        print(f"index_1 = {index_1}, \n index_2 = {index_2}")
        print()
        print(f"parent_1 = {parent_1}, \n parent_2 = {parent_2}")
    
        # if the index from parent_1 comes before parent_2
        """
        e.g. parent_1 = 0,1,>>1<<,1,0,0,1,0  --> index = 2
             parent_2 = 0,0,1,0,0,1,>>1<<,1  --> index = 6
             
        e.g. parent_1 = 0,1,1,1,0,0,>>1<<,0  --> index = 6 xx_wrong_xx
             parent_2 = 0,0,>>1<<,0,0,1,1,1  --> index = 2 xx_wrong_xx
        """
        
        if index_1 < index_2:
            
            ### for parent_1 ###
            ### for parent_1 ###
            ### for parent_1 ###
            print()
            print()
            print(f"parent_1: {parent_1}")
            
            # first_seg_parent_1: -->
            # for parent_1: the genes from the beginning of parent_1 to the 
            #       beginning of the middle segment of parent_1
            first_seg_parent_1 = parent_1[:index_1]
            print(f"first_seg_parent_2: {first_seg_parent_1}")
            
            # middle_segment: where the crossover will happen
            # for parent_1: the genes from the index chosen for parent_1 to 
            #       the middle chosen for parent_1
            mid_seg_parent_1 = parent_1[index_1:index_2+1]
            print(f"mid_seg_parent_1: {mid_seg_parent_1}")
            
            # last_seg_parent_1: -->
            # for parent_1: the genes from the end of the middle segment of 
            #       parent_1 to the last gene of parent_1
            last_seg_parent_1 = parent_1[index_2+1:]
            print(f"last_seg_parent_1: {last_seg_parent_1}")
            
            
            ### for parent_2 ###
            ### for parent_2 ###
            ### for parent_2 ###
            print()
            print()
            print(f"parent_2: {parent_2}")
            
            # first_seg_parent_2 --> same as parent_1
            first_seg_parent_2 = parent_2[:index_1]
            print(f"first_seg_parent_2: {first_seg_parent_2}")
            
            # first_seg_parent_2 --> same as parent_1
            mid_seg_parent_2 = parent_2[index_1:index_2+1]
            print(f"mid_seg_parent_2: {mid_seg_parent_2}")
            
            # first_seg_parent_2 --> same as parent_1
            last_seg_parent_2 = parent_2[index_2+1:]
            print(f"last_seg_parent_2: {last_seg_parent_2}")
            
            
            ### creating child_1 ###
            ### creating child_1 ###
            ### creating child_1 ###
            
            # the first segment from parent_1
            # plus the middle segment from parent_2
            # plus the last segment from parent_1
            child_1 = np.concatenate((first_seg_parent_1, 
                                      mid_seg_parent_2, 
                                      last_seg_parent_1))
            
            print()
            print()
            print(f"child_1: {child_1}")
            
            ### creating child_2 ###
            ### creating child_2 ###
            ### creating child_2 ###
            
            # the first segment from parent_2
            # plus the middle segment from parent_1
            # plus the last segment from parent_2
            child_2 = np.concatenate((first_seg_parent_2, 
                                      mid_seg_parent_1, 
                                      last_seg_parent_2))
            
            print()
            print()
            print(f"child_2: {child_2}")
            
            pass
        else:
            
            
            print()
            print()
            print(f"parent_1: {parent_1}")
            
            first_seg_parent_1 = parent_1[:index_2]
            print(f"first_seg_parent_1: {first_seg_parent_1}")
            
            print()
            mid_seg_parent_1 = parent_1[index_2:index_1+1]
            print(f"mid_seg_parent_1: {mid_seg_parent_1}")
            
            last_seg_parent_1 = parent_1[index_1+1:]
            print(f"last_seg_parent_1: {last_seg_parent_1}")
            
            
            print()
            print()
            print(f"parent_2: {parent_2}")
            
            first_seg_parent_2 = parent_2[:index_2]
            print(f"first_seg_parent_2: {first_seg_parent_2}")
            
            # first_seg_parent_2 --> same as parent_1
            mid_seg_parent_2 = parent_2[index_2:index_1+1]
            print(f"mid_seg_parent_2: {mid_seg_parent_2}")
            
            # first_seg_parent_2 --> same as parent_1
            last_seg_parent_2 = parent_2[index_1+1:]
            print(f"last_seg_parent_2: {last_seg_parent_2}")
            
            
            child_1 = np.concatenate((first_seg_parent_1, 
                                      mid_seg_parent_2, 
                                      last_seg_parent_1))
            
            print()
            print()
            print(f"child_1: {child_1}")
            
            ### creating child_2 ###
            ### creating child_2 ###
            ### creating child_2 ###
            
            # the first segment from parent_2
            # plus the middle segment from parent_1
            # plus the last segment from parent_2
            child_2 = np.concatenate((first_seg_parent_2, 
                                      mid_seg_parent_1, 
                                      last_seg_parent_2))
            
            print()
            print()
            print(f"child_2: {child_2}")
            pass
        
    # when we will not crossover
    # when rand_num_2_crossover_or_not is not LESS (is greater) than prob_crossover
    # when prob_crossover == 1, then rand_num_2_crossover_or_not will always be less
    #           than prob_crossover
    else:
        child_1 = parent_1
        child_2 = parent_2

def test_crossover():
    parent_1 = np.array([20, 90, 50, 30, 100,
                         60, 70, 80, 40, 10])
    parent_2 = np.array([50, 60, 40, 10, 90,
                         80, 100, 30, 20, 70])
    
    child_1, child_2 = crossover(parent_1, parent_2, prob_crossover=1)
    
###############################################################################
#
#                       Crossover   --->    end
#
###############################################################################

###############################################################################
#
#                       mutate   --->    begin
#
###############################################################################
def mutate(child_1, prob_mutation = 0.3):
    """
    it mutates the gene of each child from the prob_mutation

    Parameters
    ----------
    child_1 : np.array
        DESCRIPTION.
        
    prob_mutation : float
        DESCRIPTION.

    Returns
    -------
    mutated_child_1 : np.array
        DESCRIPTION.
        chromosome that some genes are mutated.

    """
    
    print()
    print(f"child_1 = {child_1}")
    
    mutated_child_1 = np.empty((0, len(child_1)))
    
    mutated_indices = []
    t = 0 # start at the very first index of child_1
    for i in range(len(child_1)):   # for each gene
        
        print()
        print(f"current index is {t}")
        
        rand_num_2_mutate_or_not = np.random.rand() # do we mutate or no???
        
        # print()
        # print(f"rand_num_2_mutate_or_not = {rand_num_2_mutate_or_not}")
        
        # if the rand_num_2_mutate_or_not is less than the probability of 
        #       then we mutate at that given gene (index we are currently at ) 
        
        if rand_num_2_mutate_or_not < prob_mutation:
            print(f"random number = {round(rand_num_2_mutate_or_not, 5)}")
            print(f"{round(rand_num_2_mutate_or_not,5)} is LESS than {prob_mutation}")
            print(f"SO YES - we will mutate gene t={t}")
            
            
            mutated_indices.append(t)
            
            if child_1[t] == 0: # if we mutate, a 0 become a 1
                child_1[t] = 1
            else:
                child_1[t] = 0
                
            mutated_child_1 = child_1
            
            t = t + 1
            
            print()
            
            print(f"new child_1 = {child_1}")
            
        else:
            
            # print()
            print(f"random number = {round(rand_num_2_mutate_or_not, 5)}")
            print(f"{round(rand_num_2_mutate_or_not,5)} is GREATER than {prob_mutation}")
            print(f"SO NO - we will NOT mutate gene t={t}")
            
            mutated_child_1 = child_1
            
            t = t + 1
            
            
    return mutated_child_1

def mutation(child_1, child_2, prob_mutation = 0.3):
    """
     fonction of mutation for 2 children
     it mutates the gene of each child

    Parameters
    ----------
    child_1 : np.array
        DESCRIPTION.
        chromosome
    child_2 : np.array
        DESCRIPTION.
        chromosome
    prob_mutation : float, optional
        DESCRIPTION. The default is 0.3.
        the probability to mutate a gene

    Returns
    -------
    None.

    """
    mutated_child_1 = mutate(child_1=child_1, 
                             prob_mutation=prob_mutation)
    
    mutated_child_2 = mutate(child_1=child_2, 
                             prob_mutation=prob_mutation)
    
    return mutated_child_1, mutated_child_2

def test_mutate():
    child_1_original = np.array([0,0,1,1,1,0])
    prob_mutation = 0.3
    
    print()
    print(f"child_1_original = {child_1_original}")
    
    mutated_child = mutate(child_1=child_1_original.copy(), 
                           prob_mutation=prob_mutation)
    
    print()
    print("#######################################")
    print(f"child_1_original = {child_1_original}")
    print(f" mutated_child = {mutated_child}")
    
    
def test_mutation():
    child_1 = np.array([0,0,0,0,0,0])
    child_1_original = child_1.copy()
    
    child_2 = np.array([1,1,1,1,1,1])
    child_2_original = child_2.copy()
    
    
    mutated_child_1, mutated_child_2 \
        = mutation(child_1=child_1_original, 
                   child_2=child_2_original, 
                   prob_mutation = 0.3)
        
    print("#######################################")
    print(f"child_1_original = {child_1}")
    print(f" mutated_child_1 = {mutated_child_1}")
    
    print("---------------------------------------")
    print(f"child_2_original = {child_2}")
    print(f" mutated_child_2 = {mutated_child_2}")
    
###############################################################################
#
#                       mutate   --->    end
#
###############################################################################



if __name__ == "__main__":
    # test_compute_decoding_x()
    # test_compute_decoding_y()
    # test_compute_decoding_xy()
    # test_objective_value()
    
    # test_selecting_2_parents()
    # test_crossover()
    # test_mutate()
    test_mutation()
    pass
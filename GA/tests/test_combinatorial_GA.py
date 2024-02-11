#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 13:05:56 2024

@author: willy

test GA combinatorial function
"""
import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) \
                + "/../")


import GA_combinatorial as GA_comb


def test_fitness_value():
    chromosome = ["D","A","C","B","G","E","F","H"]
    assert 272 == GA_comb.fitness_value(chromosome)
    
def test_generate_population():
    chromosome = ["D","A","C","B","G","E","F","H"]
    n_solutions = GA_comb.generate_population(chromosome)
    assert (GA_comb.N_POPULATION,len(chromosome)) == n_solutions.shape 
    

    
    
    


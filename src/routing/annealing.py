#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File: annealing.py
# Created: 22-07-2025
# Author: Lorenzo Calandra Buonaura <lorenzocb01@gmail.com>
# Institution: University of Innsbruck - UIBK
#
# Routines for optimizing wire routing (simulated annealing). 
#


########################
# IMPORT ZONE          #
########################

import random
import numpy as np

from tqdm import tqdm

from src.routing.states import RoutingState


########################
# FUNCTIONS            #
########################

def simulated_annealing(initial_state: RoutingState, initial_temp, cooling_rate, iterations):
    current = initial_state
    best = current.clone()
    best_cost, dc_cost_map = current.compute_cost(return_dc_costs=True)
    temp = initial_temp

    for i in tqdm(range(iterations), desc="Simulated Annealing"):        
        candidate = current.clone()
        candidate.perturb(dc_cost_map=dc_cost_map)
        candidate_cost, dc_cost_map = candidate.compute_cost(return_dc_costs=True)

        delta = candidate_cost - best_cost

        if delta < 0 or random.random() < np.exp(-delta / temp):
            current = candidate

        if candidate_cost < best_cost:
            best = candidate
            best_cost = candidate_cost

        temp *= cooling_rate

        if i % 100 == 0:
            tqdm.write(f"Step {i}: Cost {best_cost:.2f}, Temp {temp:.4f}")
            
    return best

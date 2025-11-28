#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File: costs.py
# Created: 15-04-2025
# Author: Lorenzo Calandra Buonaura <lorenzocb01@gmail.com>
# Institution: University of Innsbruck - UIBK
#
# Cost functions for junction geometry optimization.
#


########################
# IMPORT ZONE          #
########################

import numpy as np

from src.systems.potentials import junction_potential

from electrode import System



def cost_function(x_values, opt_params, h, weights=[1.0, 1.0, 1.0, 1.0], initial_costs=[1.0, 1.0, 1.0, 1.0]):
    # Extract pseudo-potential values
    pseudopot_values = opt_params["pseudopot_values"]
    positions = opt_params["positions"]
    z_values = positions[2]
    omega_sec_values = opt_params["omega_sec_values"]
    
    cost_1 = np.sum(pseudopot_values/1e-3)
    gradient_values = np.gradient(pseudopot_values/1e-3, x_values)
    cost_2 = np.sum(np.abs(gradient_values)**2)
    cost_3 = np.var((omega_sec_values[:, 0] + omega_sec_values[:, 1] + omega_sec_values[:, 2]) / (2*np.pi) / 1e6) 
    cost_4 = np.sum((z_values/h - 1)**2)
    
    cost_1 /= initial_costs[0]
    cost_2 /= initial_costs[1]
    cost_3 /= initial_costs[2]
    cost_4 /= initial_costs[3]
    
    weights = np.array(weights) / np.sum(weights)
    total_cost = weights[0] * cost_1 + weights[1] * cost_2 + weights[2] * cost_3 + weights[3] * cost_4
        
    return total_cost


def initial_cost(s: System, x_values, h, trap_params):        
    # Get pseudopotential values and cost
    opt_params = junction_potential(s, x_values, trap_params)
    
    pseudopot_values = opt_params["pseudopot_values"]
    positions = opt_params["positions"]
    z_values = positions[2]
    omega_sec_values = opt_params["omega_sec_values"]
    
    cost_1 = np.sum(pseudopot_values/1e-3)
    gradient_values = np.gradient(pseudopot_values/1e-3, x_values)
    cost_2 = np.sum(np.abs(gradient_values)**2)
    cost_3 = np.var((omega_sec_values[:, 0] + omega_sec_values[:, 1] + omega_sec_values[:, 2]) / (2*np.pi) / 1e6) 
    cost_4 = np.sum((z_values/h - 1)**2)
       
    costs = [cost_1, cost_2, cost_3, cost_4]
    
    return costs
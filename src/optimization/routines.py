#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File: routines.py
# Created: 16-04-2025
# Author: Lorenzo Calandra Buonaura <lorenzocb01@gmail.com>
# Institution: University of Innsbruck - UIBK
#
# Routines for optimizing junction geometry.
#


########################
# IMPORT ZONE          #
########################

import numpy as np
import warnings
import scipy.constants as ct

from scipy.optimize import minimize
from tqdm import tqdm

from src.systems.junctions import BaseJunction
from src.geometry.modules import ControlPoints
from src.systems.modules import TrapConstants
from src.optimization.costs import initial_cost, cost_function
from src.systems.potentials import junction_potential


########################
# FUNCTIONS            #
########################

def optimize_junction(trap: BaseJunction, x_values, **kwargs):
    # Define required parameters with default values
    default_params = {
        "h": 100,
        "factor": 0.2,
        "U_RF": 200,
        "options": {
            'maxiter': 200,        # Maximum number of iterations
            'disp': True,          # Display progress information
            'xatol': 1e-4,         # Tolerance on the change in the optimization variables
            'fatol': 1e-4,         # Tolerance on the function value change
        },
        "weights": [1.0, 1.0, 1.0, 1.0]
    }
  
    # Check for missing parameters and assign defaults
    for param, default in default_params.items():
      if param not in kwargs or kwargs[param] is None:
        warnings.warn(f"Parameter '{param}' is missing. Using default value: {default}")
        kwargs[param] = default
  
    # Assign parameters
    h = kwargs["h"]
    factor = kwargs["factor"]
    U_RF = kwargs["U_RF"]
    options = kwargs["options"]
    weights = kwargs["weights"]
    
    # Physical constants
    L = 1e-6 # μmlength scale
    M = 40 * ct.atomic_mass # ion mass (calcium)
    Q = 1 * ct.elementary_charge # ion charge (single-ion)

    V_RF = 42.2 # RF peak voltage
    Omega = 2 * np.pi * 20e6 # RF frequency in rad/s
    U_RF = V_RF * np.sqrt(Q / M) / (2 * L * Omega)

    trap_params = TrapConstants(L, M, Q, V_RF, Omega, target_freq=1e6 * 2 * np.pi)
    
    initial_control_points = trap.control_points
    initial_y = np.concatenate([initial_control_points.top, initial_control_points.bottom])
    num_top = len(initial_control_points.top)
      
    # Set bounds if not provided
    if "bounds" not in kwargs or kwargs["bounds"] is None:
        bounds = [(240.5, 242.5)] * num_top + [(40.5, 42.5)] * (len(initial_y) - num_top)
    else:
        bounds = kwargs["bounds"]

    system = trap.build(voltages={"RF": U_RF})
    
    initial_costs = initial_cost(system, x_values, h, trap_params)
    print(f"Initial costs: {initial_costs}")
    
    # Only optimize the y values of the control points
    def optimization_function(control_points_flat):
        top_points = control_points_flat[:num_top]
        bottom_points = control_points_flat[num_top:]
        new_cp = ControlPoints(top=top_points, bottom=bottom_points)
        
        new_system = trap.update_control_points(new_cp)
        
        # Get pseudopotential values and cost
        opt_params = junction_potential(new_system, x_values, trap_params)
        cost = cost_function(x_values, opt_params, h, weights, initial_costs)
        
        return cost

    # Callback function to update progress bar
    def callback(xk):
        pbar.update(1)  # Update progress bar
        pbar.set_postfix({"Current Cost": optimization_function(xk)})  # Show cost in progress bar
    
    # Create a progress bar
    pbar = tqdm(total=options['maxiter'], desc="Optimizing junction", position=0, leave=True)
    
    # Run the optimization -> possible methods are: Nelder-Mead, SLSQP, L-BFGS-B
    result = minimize(optimization_function, initial_y, method='Nelder-Mead', 
                      bounds=bounds, options=options, callback=callback)
    
    # Close progress bar
    pbar.close()
    
    return result
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File: optimizer.py
# Created: 28-05-2025
# Author: Lorenzo Calandra Buonaura <lorenzocb01@gmail.com>
# Institution: University of Innsbruck - UIBK
#
# Main script for optimizing X-junction geometry.
#

########################
# IMPORT ZONE          #
########################

import numpy as np
import scipy.constants as ct

from src.systems.junctions import PiecewiseJunction, SplineJunction
from src.io.loading import load_junction_geometry
from src.io.saving import save_optimization
from src.geometry.geometry_utils import check_all_ccw
from src.systems.potentials import compute_factor
from src.optimization.routines import optimize_junction



########################
# MAIN ZONE            #
########################

# Load geometry
geometry = load_junction_geometry("src/resources/initial_params_optimization.json")
h = 100

# Trench width
trench_width = 5

# Set segmentation flags
flags = {
    "build_RF": True,
    "build_DC": False,
    "build_C": False,
    "segment_DC": False,
    "segment_C": True
}

# Build trap
trap = SplineJunction(geometry, trench_width, flags)
system = trap.build()

check_all_ccw(system)

# Define physical params
L = 1e-6 # μmlength scale
M = 40 * ct.atomic_mass # ion mass (calcium)
Q = 1 * ct.elementary_charge # ion charge (single-ion)

# Voltages: 42.5, 23.5, 10.4
V_RF = 42.2 # RF peak voltage
Omega = 2 * np.pi * 20e6 # RF frequency in rad/s

# RF voltage applied to the electrodes parametrized so that the resulting potential equals the RF pseudo-potential in eV
U_RF = V_RF * np.sqrt(Q / M) / (2 * L * Omega)

# Compute factor for normalization
factor = compute_factor(M, Omega, h, L, Q, V_RF)


# Optimization
x_values = np.linspace(-1500, 0, 500)
#bounds = [(2 * 41.5, 120.75)] * 8 + [(10, 20.75)] * 8

bounds = [(241.45, 241.55)] + [(100, 241.75)] * 7 + [(41.45, 41.55)] + [(10, 55)] * 7
optimization_result = optimize_junction(trap,
                                        x_values,
                                        bounds=bounds,
                                        U_RF=U_RF,
                                        h=h,
                                        factor=factor,
                                        options={'maxiter':500, 'disp': True, 'xatol': 1e-2, 'fatol': 1e-2,},
                                        weights=[1., 1., 1., 1.])

print("Optimization Result:")
print(optimization_result)


# Extract optimization results
top_points = optimization_result.x[:8]
bottom_points = optimization_result.x[8:]
control_points = [top_points, bottom_points]

save_optimization(control_points, "src/resources/initial_params_optimization.json")
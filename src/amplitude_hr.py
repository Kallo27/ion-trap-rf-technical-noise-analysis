#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File: amplitude_hr.py
# Created: 11-07-2025
# Author: Lorenzo Calandra Buonaura <lorenzocb01@gmail.com>
# Institution: University of Innsbruck - UIBK
#
# Main script for simulating heating rates in the 4S trap. 
#



########################
# IMPORT ZONE          #
########################

import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as ct

from src.systems.linear_traps import ThreeRFTrap
from src.io.loading import load_threeRF_geometry
from src.geometry.geometry_utils import check_all_ccw
from src.systems.modules import TrapConstants, SimulationParameters
from src.simulators.trap_simulators import TrapPhysicsSimulator
from src.simulators.heating_rate_simulators import HeatingRateSimulator, rescale_hr_carrier_data
from src.graphics.visualize_potential import plot_minima_analysis, plot_fields, plot_squared_gradients
from src.graphics.visualize_hr import plot_hr_carrier_data, plot_heating_rates, plot_heating_rates_together
from src.io.loading import load_hr_carrier_data


########################
# MAIN                 #
########################

# Load trap geometry

geometry = load_threeRF_geometry("src/resources/ThreeRF_params.json")
trench_width = 5
flags = {
    "build_outer_RF": True,
    "build_central_RF": True,
}

trap = ThreeRFTrap(geometry, trench_width, flags)
system = trap.build()
check_all_ccw(system) # check for consistency


# Simulate pseuopotential and analyze minima

trap_constants = TrapConstants(
    L = 1e-6,
    M = 40 * ct.atomic_mass,
    Q = 1 * ct.elementary_charge,
    V_RF = 100,
    Omega = 2 * np.pi * 19e6
)

simulations_params = SimulationParameters(
    csi_values = np.linspace(-1., 1.5, 1001), 
    x0 = (0., 0., 350.), 
    x1 = (0., 0., 10.)
)

trap_physics_simulator = TrapPhysicsSimulator(
    trap=trap,
    constants=trap_constants,
    parameters=simulations_params
)

trap_physics_simulator.analyze_minima()

# plot_minima_analysis(trap_physics_simulator, mode="both")


# Load experimental data

exp_data = load_hr_carrier_data("src/data/heating rate data.xlsx")
rescaled_exp_data, indices = rescale_hr_carrier_data(exp_data, trap_physics_simulator)
# plot_hr_carrier_data(rescaled_exp_data, trap_physics_simulator)


# Compute gradient components

direction = "Right"
trap_physics_simulator.compute_fields_and_gradients(direction)

# plot_fields(trap_physics_simulator.csi_values, trap_physics_simulator.fields, direction)
# plot_squared_gradients(trap_physics_simulator.csi_values, trap_physics_simulator.squared_gradients, direction)


# Fit heating rates

hr_simulator = HeatingRateSimulator(trap_physics_simulator)
hr_simulator.set_spectral_densities("x", 1e-14, 1e-13, 1e-14)
hr_simulator.set_spectral_densities("y", 1e-14, 1e-13, 1e-14)
hr_simulator.set_spectral_densities("z", 1e-14, 1e-13, 1e-14)

hr_simulator.fit_heating_rates(rescaled_exp_data, "x", indices)
hr_simulator.fit_heating_rates(rescaled_exp_data, "y", indices)
hr_simulator.fit_heating_rates(rescaled_exp_data, "z", indices)

plot_heating_rates(hr_simulator, rescaled_exp_data, axes_to_plot=["x", "y"], log=False)
print(hr_simulator.heating_factor_2)

fig = plot_heating_rates_together(hr_simulator, rescaled_exp_data, axes_to_plot=["x", "y"])

hr_simulator.print_fit_results()


# Close all plots
input("Press Enter to close all plots...")
plt.close('all')
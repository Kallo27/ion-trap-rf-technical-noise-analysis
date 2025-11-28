#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File: junction_hr.py
# Created: 04-11-2025
# Author: Lorenzo Calandra Buonaura <lorenzocb01@gmail.com>
# Institution: University of Innsbruck - UIBK
#
# Main script for simulating heating rates in the X-junction trap. 
#

########################
# IMPORT ZONE          #
########################

import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import gdspy
import scipy.constants as ct
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

from electrode import System, PolygonPixelElectrode

from src.systems.modules import TrapConstants
from src.systems.potentials import junction_potential, fix_secular_frequencies
from src.systems.junctions import build_merged_junction
from src.graphics.visualize_hr import plot_heating_rates_junction, plot_heating_rates_junction_2


########################
# MAIN                 #
########################

# Load the GDS file
lib_sp = gdspy.GdsLibrary(infile='./gds_files/junction/layout_A/spline_junction_A_100um.gds')
lib_pw = gdspy.GdsLibrary(infile='./gds_files/junction/layout_A/piecewise_junction_A_100um.gds')
lib1 = gdspy.GdsLibrary(infile='./gds_files/junction/non_optimized/junction_single_100um.gds')
lib2 = gdspy.GdsLibrary(infile='./gds_files/linear/layout_A/fivewire_A_100um.gds')

polys_sp = []
polys_pw = []
polys1 = []
polys2 = []

# Get all cells
for cell_name, cell in lib_sp.cells.items():
    polys_sp = cell.polygons
    labels_sp = cell.labels
    
# Get all cells
for cell_name, cell in lib_pw.cells.items():
    polys_pw = cell.polygons
    labels_pw = cell.labels

# Get all cells
for cell_name, cell in lib1.cells.items():
    polys1 = cell.polygons
    labels1 = cell.labels
    
# Get all cells
for cell_name, cell in lib2.cells.items():
    polys2 = cell.polygons
    labels2 = cell.labels
                 
pw_polys = [poly.polygons for poly in polys_pw if poly.layers[0] == 0]
sp_polys = [poly.polygons for poly in polys_sp if poly.layers[0] == 0]
nonopt_polys = [poly.polygons for poly in polys1 if poly.layers[0] == 0]
lin_polys = [poly.polygons for poly in polys2 if poly.layers[0] == 0]
sp_polys_rf = sp_polys[-4:]
pw_polys_rf = pw_polys[-4:]
nonopt_polys_rf = nonopt_polys[-4:]
lin_polys_rf = lin_polys[-2:]

merged_polys_sp, merged_paths_sp = build_merged_junction(sp_polys_rf, lin_polys_rf)
merged_polys_pw, merged_paths_pw = build_merged_junction(pw_polys_rf, lin_polys_rf)
merged_polys1, merged_paths1 = build_merged_junction(nonopt_polys_rf, lin_polys_rf)

system_sp = System([PolygonPixelElectrode(name=n.text, paths=map(np.array, p)) for n, p in zip(labels_sp, merged_paths_sp)])
system_pw = System([PolygonPixelElectrode(name=n.text, paths=map(np.array, p)) for n, p in zip(labels_pw, merged_paths_pw)])
nonopt_system = System([PolygonPixelElectrode(name=n.text, paths=map(np.array, p)) for n, p in zip(labels1, merged_paths1)])

# Physical constants
L = 1e-6 # μmlength scale
M = 40 * ct.atomic_mass # ion mass (calcium)
Q = 1 * ct.elementary_charge # ion charge (single-ion)

V_RF = 42.2 # RF peak voltage
Omega = 2 * np.pi * 20e6 # RF frequency in rad/s
U_RF = V_RF * np.sqrt(Q / M) / (2 * L * Omega) # I already put here the constant to go to the pseudopotential in eV!!!

trap_params = TrapConstants(L, M, Q, V_RF, Omega, target_freq=1e6 * 2 * np.pi)
x_values = np.linspace(-1000, 0, 2000)

# Turn on only RF voltages -> we set them equal to the chosen value of energy
for electrode in system_pw:
    electrode.rf = U_RF
    
for electrode in system_sp:
    electrode.rf = U_RF

for electrode in nonopt_system:
    electrode.rf = U_RF

ps_results_sp = junction_potential(system_sp, x_values, trap_params)
ps_results_fixed_sp = fix_secular_frequencies(ps_results_sp)

ps_results_pw = junction_potential(system_pw, x_values, trap_params)
ps_results_fixed_pw = fix_secular_frequencies(ps_results_pw)

nonopt_ps_results = junction_potential(nonopt_system, x_values, trap_params)
nonopt_ps_results_fixed = fix_secular_frequencies(nonopt_ps_results)

nonopt_grad_values_sqr = nonopt_ps_results_fixed["grad_squared_values"] * Q
grad_values_sqr_sp = ps_results_fixed_sp["grad_squared_values"] * Q
grad_values_sqr_pw = ps_results_fixed_pw["grad_squared_values"] * Q

nonopt_hr_values = (1 / (4*M*ct.hbar*1e6)) * nonopt_grad_values_sqr**2 / (V_RF)**2
hr_values_sp = (1 / (4*M*ct.hbar*1e6)) * grad_values_sqr_sp**2 / V_RF**2
hr_values_pw = (1 / (4*M*ct.hbar*1e6)) * grad_values_sqr_pw**2 / V_RF**2

fig1 = plot_heating_rates_junction(x_values, nonopt_hr_values, nonopt_ps_results, indices=slice(1000, None))
fig2 = plot_heating_rates_junction_2(x_values, hr_values_sp, hr_values_pw, nonopt_hr_values, indices=slice(1000, None))

plt.tight_layout()
plt.show(block=False)

# Close all plots
input("Press Enter to close all plots...")
plt.close('all')
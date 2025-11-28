#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File: simulate_ps.py
# Created: 13-10-2025
# Author: Lorenzo Calandra Buonaura <lorenzocb01@gmail.com>
# Institution: University of Innsbruck - UIBK
#
# Main script for simulating pseudopotential analysis.
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
from src.graphics.visualize_trap import plot_trap_nice, plot_rf_nice
from src.graphics.visualize_potential import plot_ps_analysis


########################
# MAIN                 #
########################

# Load the GDS file
lib = gdspy.GdsLibrary(infile='./gds_files/junction/layout_A/piecewise_junction_A_50um.gds')
lib1 = gdspy.GdsLibrary(infile='./gds_files/junction/non_optimized/junction_single_50um.gds')
lib2 = gdspy.GdsLibrary(infile='./gds_files/linear/layout_A/fivewire_A_50um.gds')

polys = []
polys1 = []
polys2 = []

# Get all cells
for cell_name, cell in lib.cells.items():
    polys = cell.polygons
    labels = cell.labels

# Get all cells
for cell_name, cell in lib1.cells.items():
    polys1 = cell.polygons
    labels1 = cell.labels
    
# Get all cells
for cell_name, cell in lib2.cells.items():
    polys2 = cell.polygons
    labels2 = cell.labels
                 
trap_polys = [poly.polygons for poly in polys if poly.layers[0] == 0]
nonopt_polys = [poly.polygons for poly in polys1 if poly.layers[0] == 0]
lin_polys = [poly.polygons for poly in polys2 if poly.layers[0] == 0]
trap_polys_rf = trap_polys[-4:]
nonopt_polys_rf = nonopt_polys[-4:]
lin_polys_rf = lin_polys[-2:]

merged_polys, merged_paths = build_merged_junction(trap_polys_rf, lin_polys_rf, dx=-750)
merged_polys1, merged_paths1 = build_merged_junction(nonopt_polys_rf, lin_polys_rf, dx=-750)

system = System([PolygonPixelElectrode(name=n.text, paths=map(np.array, p)) for n, p in zip(labels, trap_polys)])
system_rf = System([PolygonPixelElectrode(name=n.text, paths=map(np.array, p)) for n, p in zip(labels, trap_polys_rf)])
merged_system = System([PolygonPixelElectrode(name=n.text, paths=map(np.array, p)) for n, p in zip(labels, merged_paths)])
nonopt_system = System([PolygonPixelElectrode(name=n.text, paths=map(np.array, p)) for n, p in zip(labels, merged_paths1)])

fig = plot_trap_nice(system, yellow_indices=[41, 42, 43, 44], blue_indices=[i for i in range(16)])

fig_rf = plot_rf_nice(system_rf) #, y_top, y_bottom) -> to be defined 

# Physical constants
L = 1e-6 # μmlength scale
M = 40 * ct.atomic_mass # ion mass (calcium)
Q = 1 * ct.elementary_charge # ion charge (single-ion)

V_RF = 23.5 # RF peak voltage
Omega = 2 * np.pi * 20e6 # RF frequency in rad/s
U_RF = V_RF * np.sqrt(Q / M) / (2 * L * Omega)

trap_params = TrapConstants(L, M, Q, V_RF, Omega, target_freq=1e6 * 2 * np.pi)
x_values = np.linspace(-500, 1, 500)

# Turn on only RF voltages -> we set them equal to the chosen value of energy
for electrode in merged_system:
    electrode.rf = U_RF

for electrode in nonopt_system:
    electrode.rf = U_RF

ps_results = junction_potential(merged_system, x_values, trap_params)
ps_results_fixed = fix_secular_frequencies(ps_results)

nonopt_ps_results = junction_potential(nonopt_system, x_values, trap_params)
nonopt_ps_results_fixed = fix_secular_frequencies(nonopt_ps_results)

fig1, fig2 = plot_ps_analysis(x_values, nonopt_ps_results_fixed)
fig3, fig4 = plot_ps_analysis(x_values, ps_results_fixed, nonopt_ps_results_fixed)


plt.tight_layout()
plt.show(block=False)

# Close all plots
input("Press Enter to close all plots...")
plt.close('all')
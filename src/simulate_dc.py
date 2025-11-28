#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File: simulate_dc.py
# Created: 29-04-2025
# Author: Lorenzo Calandra Buonaura <lorenzocb01@gmail.com>
# Institution: University of Innsbruck - UIBK
#
# Main script for simulating shims sets analysis. 
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

from electrode import System, PolygonPixelElectrode

from src.systems.modules import TrapConstants
from src.systems.potentials import compute_waveforms
from src.graphics.visualize_potential import plot_waveforms_2x2
from src.graphics.visualize_trap import plot_trap_nice



########################
# MAIN                 #
########################

# Load the GDS file
lib = gdspy.GdsLibrary(infile='./gds_files/junction/layout_A/spline_junction_A_100um.gds')
                       
polys = []

# Get all cells
for cell_name, cell in lib.cells.items():
    polys = cell.polygons
    labels = cell.labels
                 
trap_polys = [poly.polygons for poly in polys if poly.layers[0] == 0]

system = System([PolygonPixelElectrode(name=n.text, paths=map(np.array, p)) for n, p in zip(labels, trap_polys)])

# Physical constants
L = 1e-6 # μmlength scale
M = 40 * ct.atomic_mass # ion mass (calcium)
Q = 1 * ct.elementary_charge # ion charge (single-ion)

V_RF = 42.2 # RF peak voltage
Omega = 2 * np.pi * 20e6 # RF frequency in rad/s
U_RF = V_RF * np.sqrt(Q / M) / (2 * L * Omega)

trap_params = TrapConstants(L, M, Q, V_RF, Omega, target_freq=1e6 * 2 * np.pi)
x_values = np.linspace(-750, 750, 1000)
derivs = "xx x y z"
threshold = 1

waveforms, s_RF, s_DC, k_values, voltages = compute_waveforms(system, x_values, derivs, trap_params)

k_values = np.array(k_values)
mask = (k_values < 1.0)  # keep only “reasonable” range

# Apply mask to all arrays that depend on x
x_values_masked = np.array(x_values)[mask]
voltages_masked = np.array(voltages)[mask]

fig1 = plot_waveforms_2x2(x_values_masked, voltages_masked, 1, s_DC.names, 'tab10')
fig2 = plot_trap_nice(system, [41, 42, 43, 44], list(range(0, 16)))

plt.tight_layout()
plt.show(block=False)

# Close all plots
input("Press Enter to close all plots...")
plt.close('all')
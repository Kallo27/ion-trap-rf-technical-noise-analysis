#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File: potential.py
# Created: 15-04-2025
# Author: Lorenzo Calandra Buonaura <lorenzocb01@gmail.com>
# Institution: University of Innsbruck - UIBK
#
# Functions computing pseudopotential, waveforms, and related quantities for
# different trap geometries. 
#


########################
# IMPORT ZONE          #
########################

import numpy as np
import scipy.constants as ct

from electrode import System

from src.systems.shims import calc_shims
from src.systems.modules import TrapConstants

########################
# FUNCTIONS            #
########################

def compute_factor(M, Omega, h, L, Q, V_RF):
    return (4 * M * Omega**2 * (h*L)**2) / (Q * V_RF**2)


def junction_potential(trap: System, x_values, params: TrapConstants):
    x0 = (x_values[0], 10., 10.)
    positions = []
    mode_directions = []
    omega_sec_values = []
    grad_squared_values = []

    for x in x_values:
        try:
            x0 = trap.minimum(x0=(x, x0[1], x0[2]), axis=(1,2), coord=np.identity(3))
            positions.append(x0)
            curve_z = trap.modes(x0)
            mode_directions.append(np.transpose(curve_z[1]))
            omega_sec_values.append(np.sqrt(params.Q * abs(curve_z[0]) / params.M) / params.L)
            E = -trap.electrical_potential(x=x0, typ="rf", derivative=1)[0]
            grad_E = -trap.electrical_potential(x=x0, typ="rf", derivative=2, expand=True)[0]
            grad_squared_values.append(2 * np.einsum("i,ij->j", E, grad_E))
            
        except Exception as e:
            print(f"Minimum not found at x={x}: {e}")
            positions.append(x0)
            curve_z = trap.modes(x0)
            mode_directions.append(np.transpose(curve_z[1]))
            omega_sec_values.append(np.sqrt(params.Q * abs(curve_z[0]) / params.M) / params.L)
            E = -trap.electrical_potential(x=x0, typ="rf", derivative=1)[0]
            grad_E = -trap.electrical_potential(x=x0, typ="rf", derivative=2, expand=True)[0]
            grad_squared_values.append(2 * np.einsum("i,ij->j", E, grad_E))
            
    pseudopot_values = trap.potential(positions, 0)
    E_values = -trap.potential(positions, 1)[0]
    
    grad_values = np.gradient(pseudopot_values, x_values)

    return {
        "pseudopot_values": np.array(pseudopot_values),
        "E_values": np.array(E_values),
        "grad_values": np.array(grad_values),
        "grad_squared_values": np.array(grad_squared_values),
        "positions": np.array(positions),
        "mode_directions": np.array(mode_directions),
        "omega_sec_values": np.array(omega_sec_values)
    }
    
def fix_secular_frequencies(pot_results, threshold=0.1):
    # Get indices of values near 0
    indices_near_zero = np.where(np.abs(np.array(pot_results["mode_directions"])[:, 2, 1]) > threshold)[0]

    # Swap secular frequencies
    copy_values = pot_results["omega_sec_values"][indices_near_zero, 1]
    pot_results["omega_sec_values"][indices_near_zero, 1] = pot_results["omega_sec_values"][indices_near_zero, 2]
    pot_results["omega_sec_values"][indices_near_zero, 2] = copy_values
    
    return pot_results


def compute_waveforms(system: System, x_values, derivs, params: TrapConstants):
    # Compute effective RF voltage
    U_RF = params.V_RF * np.sqrt(params.Q / params.M) / (2 * params.L * params.Omega)

    # Set RF and DC voltages
    for e in system:
        if "RF" in e.name:
            e.rf = U_RF
        else:
            e.dc = 0

    # Sort electrodes
    s_RF = System([entry for entry in system if "RF" in entry.name])
    s_DC = System([entry for entry in system if "RF" not in entry.name])

    # Initialize arrays
    uxx_vals, ux_vals, uy_vals, uz_vals = [], [], [], []
    x0_positions = []
    k_values = []
    voltages = []

    for x in x_values:
        x0_varied = (x, 10., 10.)
        x0 = system.minimum(x0=x0_varied, axis=(1,2), coord=np.identity(3))
        x0_positions.append(x0)

    prev_k = 1.0
    target_freq=1e6 * 2 * np.pi
    for j, x in enumerate(x_values):
        with system.with_voltages(dcs=0*system.dcs, rfs=system.rfs):
            x0_varied = (x, x0[1], x0[2])
            u_cal = calc_shims(x0_varied, s_DC, derivs)

            uxx_vals.append(np.array(u_cal['xx'] * 1e-4))
            ux_vals.append(np.array(u_cal['x']) * 1e-2)
            uy_vals.append(np.array(u_cal['y']) * 1e-2)
            uz_vals.append(np.array(u_cal['z']) * 1e-2)

            for i, e in enumerate(s_DC):
                e.dc = uxx_vals[-1][i]

            # Compute current axial frequency
            curve_z = system.modes(x0_positions[j])
            omega_z = np.sqrt(params.Q * abs(curve_z[0]) / params.M) / params.L
            k = target_freq / omega_z[0]

            if -50 <= x <= 50:
                if abs(np.log(k / prev_k)) < 0.3:   # allow max 30% change
                    prev_k = prev_k
            else:
                prev_k = k
                
            for e in s_DC:
                e.dc *= prev_k**2

            k_values.append(prev_k)
            voltages.append([e.dc for e in s_DC])

    # Bundle waveforms into a dictionary
    waveforms = {
        "xx": np.array(uxx_vals),
        "x": np.array(ux_vals),
        "y": np.array(uy_vals),
        "z": np.array(uz_vals),
    }

    return waveforms, s_RF, s_DC, k_values, voltages



def turn_potential(trap: System, constants: TrapConstants, path, axis_path):
    positions = []
    omega_sec_values = []
    pseudopot_values = []
    grad_values = []

    Q = constants.Q
    M = constants.M
    L = constants.L

    x0 = path[0]

    for pt, axis in zip(path, axis_path):
        try:
            x0 = trap.minimum(
                x0=(pt[0], pt[1], pt[2]),
                axis=axis,
                coord=np.identity(3),
                method="Powell"
            )
        except Exception as e:
            print(f"Minimum not found at {pt}: {e}")
            # continue with last known x0

        positions.append(x0)

        curve_z = trap.modes(x0)
        omega_sec = np.sqrt(abs(curve_z[0]) / M)
        omega_sec *= constants.pseudopotential_factor * constants.physical_units_factor / L
        omega_sec_values.append(omega_sec)

        pot = trap.potential(x=x0, derivative=0)[0]
        pot *= constants.pseudopotential_factor**2 * constants.physical_units_factor**2
        pseudopot_values.append(pot)
        
        grad_E = -trap.electrical_potential(x=x0, typ="rf", derivative=2, expand=True)[0]
        grad_values.append(grad_E)

    return {
        "pseudopot_values": np.array(pseudopot_values),
        "grad_values": np.array(grad_values),
        "positions": np.array(positions),
        "omega_sec_values": np.array(omega_sec_values),
    }

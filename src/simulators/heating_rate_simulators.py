#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File: heating_rate_simulators.py
# Created: 10-07-2025
# Author: Lorenzo Calandra Buonaura <lorenzocb01@gmail.com>
# Institution: University of Innsbruck - UIBK
#
# Classes defining heating rate simulators.
#


########################
# IMPORT ZONE          #
########################

import numpy as np
import scipy.constants as ct
from scipy.optimize import curve_fit
from abc import ABC, abstractmethod


from src.simulators.trap_simulators import TrapPhysicsSimulator


########################
# CLASSES              #
########################

class BaseHRSimulator(ABC):
    def __init__(self, physics_simulator: TrapPhysicsSimulator, axes):
        self.sim = physics_simulator
        self.axes = axes
        
        self.const = physics_simulator.const
        self.L = physics_simulator.L
        self.M = physics_simulator.M
        self.V0 = physics_simulator.V0
        self.Omega = physics_simulator.Omega
        
        sec_freqs = physics_simulator.get_ordered_secular_frequencies(in_mhz=False)
        self.heating_factor = 1 / (1 * self.M * ct.hbar * (sec_freqs[:, 0]))
        self.heating_factor_2 = 1 + (sec_freqs[:, 0])**2 / (2*self.Omega**2)
        
        self.ps_factor = physics_simulator.ps_factor
        self.pu_factor = physics_simulator.pu_factor
        
        self.params = physics_simulator
        self.csi_values = physics_simulator.csi_values
        
        self.heating_rates_results = {
            axis: {"Outer": [], "Inner": [], "Mixed": []} for axis in self.axes
        }

        self.spectral_densities = {
            axis: {"Outer": None, "Inner": None, "Mixed": None} for axis in self.axes
        }

        self.optimization_results = {axis: {} for axis in self.axes}


    @abstractmethod
    def heating_rates_outer(self, grad_E, S_V_O, axis0, axis1=None):        
        pass

    @abstractmethod
    def heating_rates_inner(self, grad_E, S_V_I, axis0, axis1=None):
        pass

    @abstractmethod
    def heating_rates_mixed(self, grad_E1, grad_E2, S_total, r, axis0, axis1=None):
        pass
    

    def compute_all_heating_rates(self, axis_label):
        # Get axes from axis_map; could be int or tuple
        axes = self.axis_map[axis_label]
        
        # Ensure axes is always a tuple for unpacking
        if isinstance(axes, int):
            axis0, axis1 = axes, None
        else:
            axis0, axis1 = axes

        # (Assuming your gradient fetching code stays the same...)
        if self.sim.squared_gradients["grad_E_O_squared"] is None or len(self.sim.squared_gradients["grad_E_O_squared"]) == 0:
            self.sim.compute_fields_and_gradients()

        grad_outer = self.sim.squared_gradients["grad_E_O_squared"] + self.sim.squared_gradients["grad_E_O_dot_E_I"]
        grad_inner = self.sim.squared_gradients["grad_E_I_squared"] + self.sim.squared_gradients["grad_E_O_dot_E_I"]

        S_V_O = self.spectral_densities[axis_label]["Outer"]
        S_V_I = self.spectral_densities[axis_label]["Inner"]
        S_V_mixed = self.spectral_densities[axis_label]["Mixed"]

        self.heating_rates_results[axis_label]["Outer"] = self.heating_rates_outer(grad_outer, S_V_O, axis0, axis1)
        self.heating_rates_results[axis_label]["Inner"] = self.heating_rates_inner(grad_inner, S_V_I, axis0, axis1)
        self.heating_rates_results[axis_label]["Mixed"] = self.heating_rates_mixed(grad_outer, grad_inner, S_V_mixed, axis0, axis1)

        
    def set_spectral_densities(self, axis_label, S_V_O, S_V_I, S_V_mixed):
        self.spectral_densities[axis_label] = {
            "Outer": S_V_O,
            "Inner": S_V_I,
            "Mixed": S_V_mixed
        }
    
    
    def _model_outer(self, dummy_x, S_V_O, axis):
        grad_outer = self.sim.squared_gradients["grad_E_O_squared"] + self.sim.squared_gradients["grad_E_O_dot_E_I"]
        if isinstance(axis, tuple):
            axis0, axis1 = axis
            hr = self.heating_rates_outer(grad_outer, S_V_O, axis0, axis1)
        else:
            hr = self.heating_rates_outer(grad_outer, S_V_O, axis)
        return hr

    def _model_inner(self, dummy_x, S_V_I, axis):
        grad_inner = self.sim.squared_gradients["grad_E_I_squared"] + self.sim.squared_gradients["grad_E_O_dot_E_I"]
        if isinstance(axis, tuple):
            axis0, axis1 = axis
            hr = self.heating_rates_inner(grad_inner, S_V_I, axis0, axis1)
        else:
            hr = self.heating_rates_inner(grad_inner, S_V_I, axis)
        return hr

    def _model_mixed(self, dummy_x, S_V_mixed, axis):
        grad_outer = self.sim.squared_gradients["grad_E_O_squared"] + self.sim.squared_gradients["grad_E_O_dot_E_I"]
        grad_inner = self.sim.squared_gradients["grad_E_I_squared"] + self.sim.squared_gradients["grad_E_O_dot_E_I"]
        if isinstance(axis, tuple):
            axis0, axis1 = axis
            hr = self.heating_rates_mixed(grad_outer, grad_inner, S_V_mixed, axis0, axis1)
        else:
            hr = self.heating_rates_mixed(grad_outer, grad_inner, S_V_mixed, axis)
        return hr
    

    def fit_heating_rates(self, data_exp, axis_label, indices):
        # Map axis_label(s) to axis indices or tuples
        axis = self.axis_map[axis_label]

        csi_exp, h_exp, error_h_exp, freq_exp = data_exp

        # Prepare dummy x for curve_fit
        x_dummy = np.zeros(len(h_exp))

        # Define wrapped models, passing axis which can be int or tuple
        def model_outer_wrap(dummy_x, S_V_O):
            hr = self._model_outer(dummy_x, S_V_O, axis=axis)
            return hr[indices]

        def model_inner_wrap(dummy_x, S_V_I):
            hr = self._model_inner(dummy_x, S_V_I, axis=axis)
            return hr[indices]

        def model_mixed_wrap(dummy_x, S_V_mixed):
            hr = self._model_mixed(dummy_x, S_V_mixed, axis=axis)
            return hr[indices]

        # Outer only fit
        popt_outer, pcov_outer = curve_fit(
            model_outer_wrap, x_dummy, h_exp,
            p0=self.spectral_densities[axis_label]["Outer"],
            sigma=error_h_exp,
            absolute_sigma=True,
            bounds=(0, np.inf)
        )

        # Inner only fit
        popt_inner, pcov_inner = curve_fit(
            model_inner_wrap, x_dummy, h_exp,
            p0=self.spectral_densities[axis_label]["Inner"],
            sigma=error_h_exp,
            absolute_sigma=True,
            bounds=(0, np.inf)
        )

        # Mixed fit
        popt_mixed, pcov_mixed = curve_fit(
            model_mixed_wrap, x_dummy, h_exp,
            p0=self.spectral_densities[axis_label]["Mixed"],
            sigma=error_h_exp,
            absolute_sigma=True,
            bounds=([0, np.inf])
        )

        # Update densities using the existing method
        self.set_spectral_densities(axis_label, popt_outer[0], popt_inner[0], popt_mixed[0])
        self.compute_all_heating_rates(axis_label)

        # Jacobians for error propagation
        results = {
            "Outer": {"params": popt_outer, "cov": pcov_outer},
            "Inner": {"params": popt_inner, "cov": pcov_inner},
            "Mixed": {"params": popt_mixed, "cov": pcov_mixed}
        }

        self.optimization_results[axis_label] = results

        
    def print_fit_results(self, digits=3):
        if not hasattr(self, "optimization_results") or not self.optimization_results:
            print("No optimization results available.")
            return

        for axis_label, results in self.optimization_results.items():
            print(f"\n📊 Fit results for axis: '{axis_label.upper()}'")
            print("=" * 50)

            for key in ["Outer", "Inner", "Mixed"]:
                section = results.get(key) if results else None

                # Meaningful parameter labels and section headers
                if key == "Outer":
                    param_names = ["S_V_O"]
                    print("\n🔹 Fit with noise only on outer electrodes:")
                elif key == "Inner":
                    param_names = ["S_V_I"]
                    print("\n🔹 Fit with noise only on inner electrode:")
                elif key == "Mixed":
                    param_names = ["S_V_mixed"]
                    print("\n🔹 Fit with noise on both electrodes:")

                if section is None:
                    print("  Fit not done.")
                    continue

                params = section.get("params")
                cov = section.get("cov")

                if params is not None and cov is not None:
                    for name, p, var in zip(param_names, params, np.diag(cov)):
                        std = np.sqrt(var) if var >= 0 else float('nan')
                        print(f"  {name}: {p:.{digits}g} ± {std:.{digits}g}")
                else:
                    print("  Parameters or covariance not available.")
            print("=" * 50)



class HeatingRateSimulator(BaseHRSimulator):
    def __init__(self, physics_simulator):
        super().__init__(physics_simulator, axes=("x", "y", "z"))
        self.axis_map = {
            "x": 0,
            "y": 2,
            "z": 1,
        } 

    def heating_rates_outer(self, grad_E, S_V_O, axis0=0, axis1=None):        
        grad_pot_sqr = np.array(grad_E)[:, axis0]**2
        h_O = self.heating_factor * self.heating_factor_2 * self.ps_factor**4 * grad_pot_sqr * S_V_O / self.V0**2
        return h_O

    def heating_rates_inner(self, grad_E, S_V_I, axis0=0, axis1=None):
        grad_pot_sqr = np.array(grad_E)[:, axis0]**2
        h_I = [self.heating_factor[i] * self.heating_factor_2[i] * self.ps_factor**4 * grad_pot_sqr[i] * S_V_I / (self.V0 * self.csi_values[i])**2 
               for i in range(len(self.csi_values))]
        return np.array(h_I)

    def heating_rates_mixed(self, grad_E1, grad_E2, S_V_mixed, axis0=0, axis1=None):
        grad_pot_sqr1 = np.array(grad_E1)[:, axis0]**2
        grad_pot_sqr2 = np.array(grad_E2)[:, axis0]**2

        h_O = self.heating_factor * self.heating_factor_2 * self.ps_factor**4 * grad_pot_sqr1 * S_V_mixed / self.V0**2
        h_I = self.heating_factor * self.heating_factor_2 * self.ps_factor**4 * grad_pot_sqr2 * S_V_mixed / self.V0**2
        return np.array(h_O + h_I)
    

class HeatingRateSimulatorTilted(BaseHRSimulator):
    def __init__(self, physics_simulator: TrapPhysicsSimulator):
        super().__init__(physics_simulator, axes=("xy", "yz", "xz"))
        self.axis_map = {
            "xy": (0, 2),
            "yz": (2, 1),
            "xz": (0, 1),
        }
        self.angles_rad = physics_simulator.angles_rad
        
    def set_angles(self, new_angles_rad: list[float]):
        if len(new_angles_rad) != 2:
            raise ValueError("new_angles_rad must be a list of two angles.")
        
        self.angles_rad = new_angles_rad
        
    def tilt_gradients(self, gradients, axis0, axis1):
        tilted_gradients = []
        
        mask_vertical = self.csi_values <= 0.76
        mask_horizontal = ~mask_vertical
        
        mode_axis_vertical = get_mode_axis(self.angles_rad[0], axis0, axis1)
        mode_axis_horizontal = get_mode_axis(self.angles_rad[1], axis0, axis1)

        for gradient in gradients:
            projection_vertical = [np.dot(grad, mode_axis_vertical) for grad in gradient[mask_vertical]]            
            projection_horizontal = [np.dot(grad, mode_axis_horizontal) for grad in gradient[mask_horizontal]]
            
            tilted_gradients.append(np.concatenate([projection_vertical, projection_horizontal]))          

        return tilted_gradients
        
    def heating_rates_outer(self, grad_E, S_V_O, axis0, axis1):
        grad_pot_tilted = self.tilt_gradients([grad_E], axis0, axis1)[0]   
        grad_pot_sqr = np.array(grad_pot_tilted)**2
        h_O = self.heating_factor * self.ps_factor**4 * grad_pot_sqr * S_V_O / self.V0**2
        return h_O

    def heating_rates_inner(self, grad_E, S_V_I, axis0, axis1):
        grad_pot_tilted = self.tilt_gradients([grad_E], axis0, axis1)[0]    
        grad_pot_sqr = np.array(grad_pot_tilted)**2
        h_I = [self.heating_factor[i] * self.ps_factor**4 * grad_pot_sqr[i] * S_V_I / (self.V0 * self.csi_values[i])**2 
               for i in range(len(self.csi_values))]
        return np.array(h_I)

    def heating_rates_mixed(self, grad_E1, grad_E2, S_V_mixed, axis0, axis1):
        grad_pot_tilted1, grad_pot_tilted2 = self.tilt_gradients([grad_E1, grad_E2], axis0, axis1)       

        grad_pot_sqr1 = np.array(grad_pot_tilted1)**2
        grad_pot_sqr2 = np.array(grad_pot_tilted2)**2

        h_O = self.heating_factor * self.ps_factor**4 * grad_pot_sqr1 * S_V_mixed / self.V0**2
        h_I = [self.heating_factor[i] * self.ps_factor**4 * grad_pot_sqr2[i] * S_V_mixed / (self.V0 * self.csi_values[i])**2 
               for i in range(len(self.csi_values))]
        return np.array(h_O + h_I)
    
    
########################
# FUNCTIONS            #
########################

def rescale_hr_carrier_data(exp_data, physics_simulator: TrapPhysicsSimulator):
    csi_exp, h_exp, error_h_exp, freq_exp = exp_data
    csi_values = physics_simulator.csi_values
    sec_freqs = physics_simulator.get_ordered_secular_frequencies(in_mhz=True)
    radial_freq = sec_freqs[:, 0]
    
    indices = []
    for csi in csi_exp:
        idx = np.where(np.isclose(csi_values, np.round(csi, 2)))[0]
        if len(idx) == 0:
            raise ValueError(f"csi value {csi} not found in csi_values!")
        indices.append(idx[0])
    indices = np.array(indices)

    for i in range(len(h_exp)):
        scale_factor = freq_exp[i] / radial_freq[indices[i]]
        h_exp[i] *= scale_factor
        error_h_exp[i] *= scale_factor

    return [csi_exp, h_exp, error_h_exp, freq_exp], indices


def build_rotation_matrix(theta_rad, axis0, axis1):
    R = np.eye(3)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    R[axis0, axis0] = c
    R[axis1, axis1] = c
    R[axis0, axis1] = s
    R[axis1, axis0] = -s        # IMPORTANT RULE HERE
    return R

def get_mode_axis(theta_rad, axis0, axis1):
    vec = np.zeros(3)
    vec[axis0] = 1.0  # starting direction

    # Build 3x3 identity rotation matrix
    R = build_rotation_matrix(theta_rad, axis0, axis1)

    return R.T @ vec

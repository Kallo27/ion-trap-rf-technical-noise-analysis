#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File: trap_simulators.py
# Created: 10-07-2025
# Author: Lorenzo Calandra Buonaura <lorenzocb01@gmail.com>
# Institution: University of Innsbruck - UIBK
#
# Classes defining trap simulators.
#


########################
# IMPORT ZONE          #
########################

import numpy as np
from tqdm import tqdm

from src.systems.modules import TrapConstants, SimulationParameters
from src.systems.linear_traps import ThreeRFTrap


########################
# CLASSES              #
########################

class TrapPhysicsSimulator:
    def __init__(self, trap: ThreeRFTrap, constants: TrapConstants, parameters: SimulationParameters):
        self.trap = trap
        
        self.const = constants
        self.L = constants.L
        self.M = constants.M
        self.V0 = constants.V_RF
        self.Omega = constants.Omega

        self.ps_factor = constants.pseudopotential_factor
        self.pu_factor = constants.physical_units_factor
        
        self.params = parameters
        self.csi_values = parameters.csi_values
        self.x0 = parameters.x0
        self.x1 = parameters.x1
        self.angles_rad = parameters.angles_rad

        # Storage for pseudopotential analysis
        self.minimum_positions = {"Up": [], "Down": []}        
        self.psuedopotentials = {"Up": [], "Down": []}
        self.secular_frequencies = []
        
        # Storage for fields and gradients analysis
        self.fields = {"E_O": [], "E_I": []}
        self.gradients = {"grad_E_O": [], "grad_E_I": []}
        self.squared_gradients = {"grad_E_O_squared": [], "grad_E_I_squared": [], "grad_E_O_dot_E_I": []}


    def _reset_minima_analysis(self):
        self.minimum_positions = {"Up": [], "Down": [], "Right": [], "Left": []}
        self.psuedopot_values = {"Up": [], "Down": []}
        self.secular_frequencies = []
        
    def _reset_fields_and_gradients_analysis(self):
        self.fields = {"E_O": [], "E_I": []}
        self.gradients = {"grad_E_O": [], "grad_E_I": []}
        self.squared_gradients = {"grad_E_O_squared": [], "grad_E_I_squared": [], "grad_E_O_dot_E_I": []}
    

    def _finalize_minima_analysis(self):
        """Convert all relevant analysis lists to numpy arrays."""
        for key in self.minimum_positions:
            self.minimum_positions[key] = np.array(self.minimum_positions[key])
            
        for key in self.psuedopotentials:
            self.psuedopotentials[key] = np.array(self.psuedopotentials[key])

        self.secular_frequencies = np.array(self.secular_frequencies)

    def _finalize_fields_and_gradients_analysis(self):
        """Convert all relevant analysis lists to numpy arrays."""
        for key in self.fields:
            self.fields[key] = np.array(self.fields[key])
            
        for key in self.gradients:
            self.gradients[key] = np.array(self.gradients[key])
            
        for key in self.squared_gradients:
            self.squared_gradients[key] = np.array(self.squared_gradients[key])

    
    def analyze_minima(self):
        self._reset_minima_analysis()
        
        for csi in tqdm(self.csi_values, desc="Analyzing minima", unit="csi"):
            voltages = {
                "central_RF": {"attr": "rf", "value": csi},
                "outer_RF": {"attr": "rf", "value": 1},
            }

            system = self.trap.build(voltages)

            try:
                if csi > 0.76:
                    self.x0 = (self.x0[0] + 1, self.x0[1], self.x0[2])
                    self.x1 = (self.x1[0] - 1, self.x1[1], self.x1[2])

                # Right minimum
                self.x0 = system.minimum(x0=self.x0, axis=(0,2), coord=np.identity(3), method="Powell")
                self.minimum_positions["Up"].append(self.x0)
                self.minimum_positions["Right"].append(self.x0)

                pot0 = system.potential(x=self.x0, derivative=0)[0]
                pot0 *= self.ps_factor**2 * self.pu_factor**2
                self.psuedopotentials["Up"].append(pot0)

                # Left minimum (only if csi > 0.16)
                if csi > 0.16:
                    self.x1 = system.minimum(x0=self.x1, axis=(0,2), coord=np.identity(3), method="Powell")
                    self.minimum_positions["Down"].append(self.x1)

                    pot1 = system.potential(x=self.x1, derivative=0)[0]
                    pot1 *= self.ps_factor**2 * self.pu_factor**2
                    self.psuedopotentials["Down"].append(pot1)
                    
                if csi > 0.76:
                    self.minimum_positions["Left"].append(self.x1)
                else:
                    self.minimum_positions["Left"].append(self.x0)

                # secular frequency
                curve_z = system.modes(self.x0)
                omega_sec = np.sqrt(abs(curve_z[0]) / self.M)
                omega_sec *= (self.ps_factor * self.pu_factor) / self.L
                self.secular_frequencies.append(omega_sec)

            except Exception as e:
                print(f"{csi}: minimum not found ({e})")

        self._finalize_minima_analysis()

                
    def get_ordered_secular_frequencies(self, in_mhz: bool = False) -> np.ndarray:
        """
        Return secular frequencies ordered as [radial1, axial, radial2].

        Parameters
        ----------
        in_mhz : bool, optional
            If True, return the frequencies in MHz. Otherwise, in rad/s. By default False.

        Returns
        -------
        np.ndarray
            Ordered secular frequencies.
        """
        if not hasattr(self, 'secular_frequencies') or len(self.secular_frequencies) == 0:
            raise RuntimeError("secular_frequencies must be a 3-element array. Run analyze_minima() first.")

        sorted_freqs = np.sort(self.secular_frequencies, axis=1)
        ordered = np.stack([
            sorted_freqs[:, 1],  # radial1
            sorted_freqs[:, 0],  # axial
            sorted_freqs[:, 2],  # radial2
        ], axis=1)

        if in_mhz:
            ordered = ordered / (2 * np.pi * 1e6)

        return ordered
    
    
    def _compute_gradients(self, E_O, grad_E_O, E_I, grad_E_I):
        """Compute heating rate gradients with and without noise contributions."""
        grad_E_O_squared = 2 * np.einsum("i,ij->j", E_O, grad_E_O)
        grad_E_I_squared = 2 * np.einsum("i,ij->j", E_I, grad_E_I)
        grad_E_O_dot_E_I = np.einsum("i,ij->j", E_O, grad_E_I) + np.einsum("i,ij->j", E_I, grad_E_O)
        
        return grad_E_O_squared, grad_E_I_squared, grad_E_O_dot_E_I
    
    
    def compute_fields_and_gradients(self, direction="Right"):
        self._reset_fields_and_gradients_analysis()
        
        flags_outer = {"build_central_RF": False}
        flags_central = {"build_outer_RF": False}
        
        for i in tqdm(range(len(self.csi_values)), desc="Analyzing gradients"):
            csi = self.csi_values[i]
            voltages_outer = {"outer_RF": {"attr": "rf", "value": 1}}
            voltages_central = {"central_RF": {"attr": "rf", "value": csi}}

            system_outer = self.trap.build(voltages_outer, flags_outer)
            system_central = self.trap.build(voltages_central, flags_central)

            try:
                x = self.minimum_positions[direction][i]
                
                # Compute fields
                E_O = -system_outer.electrical_potential(x=x, typ="rf", derivative=1)[0]
                E_O *= self.pu_factor
                
                E_I = -system_central.electrical_potential(x=x, typ="rf", derivative=1)[0] 
                E_I *= self.pu_factor

                self.fields["E_O"].append(E_O)
                self.fields["E_I"].append(E_I)
                
                # Compute gradients                
                grad_E_O = -system_outer.electrical_potential(x=x, typ="rf", derivative=2, expand=True)[0] 
                grad_E_O *= (self.pu_factor / self.L)
                
                grad_E_I = -system_central.electrical_potential(x=x, typ="rf", derivative=2, expand=True)[0] 
                grad_E_I *= (self.pu_factor / self.L)
                
                self.gradients["grad_E_O"].append(grad_E_O)
                self.gradients["grad_E_I"].append(grad_E_I)

                # Compute gradients of squared and mixed terms
                g_E_O_squared, g_E_I_squared, g_E_O_dot_E_I = self._compute_gradients(E_O, grad_E_O, E_I, grad_E_I)

                self.squared_gradients["grad_E_O_squared"].append(g_E_O_squared)
                self.squared_gradients["grad_E_I_squared"].append(g_E_I_squared)
                self.squared_gradients["grad_E_O_dot_E_I"].append(g_E_O_dot_E_I)

            except Exception as e:
                print(f"{csi}: gradient computation failed ({e})")
                
        self._finalize_fields_and_gradients_analysis()

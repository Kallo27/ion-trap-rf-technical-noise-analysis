#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File: modules.py
# Created: 30-04-2025
# Author: Lorenzo Calandra Buonaura <lorenzocb01@gmail.com>
# Institution: University of Innsbruck - UIBK
#
# Modules defining physical constants and simulation parameters.
#


########################
# IMPORT ZONE          #
########################

import numpy as np
from dataclasses import dataclass


########################
# FUNCTIONS            #
########################

@dataclass
class TrapConstants:
    """
    Container for the physical constants and operating parameters of a surface-electrode ion trap.

    Attributes:
        L (float): Length scale (in meters), typically set by the electrode geometry.
        M (float): Ion mass [kg].
        Q (float): Ion charge [C].
        V_RF (float): Peak RF voltage applied to the trap [V].
        Omega (float): RF drive frequency [rad/s].
        target_freq (float): Target secular frequency [rad/s].
    """
    L: float
    M: float
    Q: float
    V_RF: float
    Omega: float
    target_freq: float | None = None

    @property
    def pseudopotential_factor(self) -> float:
        return self.Q / (2 * self.Omega * np.sqrt(self.M))
    
    @property
    def physical_units_factor(self) -> float:
        return self.V_RF / self.L
    

@dataclass
class SimulationParameters:
    csi_values: np.ndarray
    x0: tuple
    x1: tuple
    angles_deg: tuple[float, float] | None = None

    def __post_init__(self):
        # Replace zeros in csi_values with a small number to avoid division issues
        self.csi_values = np.where(self.csi_values == 0, 1e-12, self.csi_values)
        
        # Convert angles if provided
        if self.angles_deg is not None:
            self.angles_rad = np.deg2rad(self.angles_deg)
        else:
            self.angles_rad = None
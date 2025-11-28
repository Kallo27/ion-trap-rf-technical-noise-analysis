#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File: modules.py
# Created: 10-04-2025
# Author: Lorenzo <lorenzocb01@gmail.com>
# Institution: University of Innsbruck - UIBK
#
# This file contains the definition of the modules containing the geometrical 
# dimensions for the different geometries. Each module is a dataclass.
#


########################
# IMPORT ZONE          #
########################

from dataclasses import dataclass


########################
# CLASSES              #
########################


# Junction

@dataclass
class Junction_Cdimensions:
    """
    Container for the dimensions of the inner DC electrodes.
    """
    width: float
    heights: list[float]
    radius: float | None = None

@dataclass
class Junction_RFdimensions:
    """
    Container for the dimensions of the RF electrodes.
    """
    length: float
    width: float
    x_opt: float | None = None

@dataclass
class Junction_DCdimensions:
    """
    Container for the dimensions of the outer DC electrodes.
    """
    width: float
    height: float
    count: int
    
@dataclass
class ControlPoints:
    """
    Container for the control points of the optimization.
    """
    top: list[float]
    bottom: list[float]

@dataclass
class JunctionGeometry:
    """
    Container for the geometrical parameters of the junction.
    """
    RF: Junction_RFdimensions | None = None
    DC: Junction_DCdimensions | None = None
    C: Junction_Cdimensions | None = None
    points: ControlPoints | None = None


# Linear

@dataclass
class Linear_DCdimensions:
    """
    Container for the dimensions of the outer DC electrodes.
    """
    width: float
    height: float
    count: int
    
@dataclass
class Linear_RFdimensions:
    """
    Container for the dimensions of the RF electrodes.
    """
    length: float
    width: float
    offset: float | None = None
    
@dataclass
class Linear_Cdimensions:
    """
    Container for the dimensions of the inner DC electrodes.
    """
    width: float
    heights: list[float]
    
@dataclass
class FiveWireGeometry:
    RF: Linear_RFdimensions | None = None
    DC: Linear_DCdimensions | None = None
    C: Linear_Cdimensions | None = None

@dataclass
class ThreeRFGeometry:
    OUTER: Linear_RFdimensions
    CENTRAL: Linear_RFdimensions


# Curved

@dataclass
class Turn_DCdimensions:
    """
    Container for the dimensions of the outer DC electrodes.
    """
    height: float
    count: int
    
@dataclass
class Turn_RFdimensions:
    """
    Container for the dimensions of the RF electrodes.
    """
    length: float
    width: float
    
@dataclass
class Turn_Cdimensions:
    """
    Container for the dimensions of the inner DC electrodes.
    """
    width: float
    count: int
    
@dataclass
class TurnGeometry:
    RF: Turn_RFdimensions | None = None
    DC: Turn_DCdimensions | None = None
    C: Turn_Cdimensions | None = None
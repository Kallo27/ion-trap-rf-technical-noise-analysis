#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File: linear_electrodes.py
# Created: 10-04-2025
# Author: Lorenzo Calandra Buonaura <lorenzocb01@gmail.com>
# Institution: University of Innsbruck - UIBK
#
# Functions defining electrode shapes for the linear (fivewire) trap geometry.
#


########################
# IMPORT ZONE          #
########################

import numpy as np

from src.geometry.modules import FiveWireGeometry, ThreeRFGeometry
from src.geometry.geometry_utils import remove_repeated_vertices


########################
# FUNCTIONS            #
########################

def rf_shapes(fivewire_geometry: FiveWireGeometry) -> list[tuple[float, float]]:
    rf = fivewire_geometry.RF
    dc = fivewire_geometry.DC
    c = fivewire_geometry.C
  
    shape = []
  
    # Add fixed endcap segment
    shape.extend([
        (-rf.length/2, c.width/2 + rf.width),
        (-rf.length/2, c.width/2),
        (rf.length/2, c.width/2),
        (rf.length/2, c.width/2 + rf.width),
    ])
  
    # Remove repeated vertices
    new_shape = remove_repeated_vertices(shape, 1e-6)
  
    return new_shape


def central_shape(fivewire_geometry: FiveWireGeometry,
                  trench_width: float) -> list[tuple[float, float]]:
    rf = fivewire_geometry.RF
    dc = fivewire_geometry.DC
    c = fivewire_geometry.C
    
    shape = []
    
    # Add fixed endcap segment
    shape.extend([
        (-rf.length/2 + trench_width/2, c.width/2 - trench_width),
        (-rf.length/2 + trench_width/2, -c.width/2 + trench_width),
        (rf.length/2 - trench_width/2, -c.width/2 + trench_width),
        (rf.length/2 - trench_width/2, c.width/2 - trench_width),
    ])
    
    # Remove repeated vertices
    new_shape = remove_repeated_vertices(shape, tolerance=1e-6)
    
    return new_shape


def dc_shapes(fivewire_geometry: FiveWireGeometry,
              trench_width: float) -> list[list[tuple[float, float]]]:
    rf = fivewire_geometry.RF
    dc = fivewire_geometry.DC
    c = fivewire_geometry.C
    
    shapes = []
    
    # === Build the first DC electrode shape ===    
    shape = []
    shape1 = []
    
    shape.extend([
        (-rf.length/2 + trench_width/2, c.width/2 + rf.width + trench_width + dc.height),
        (-rf.length/2 + trench_width/2, c.width/2 + rf.width + trench_width),
        (-trench_width/2, c.width/2 + rf.width + trench_width),
        (-trench_width/2, c.width/2 + rf.width + trench_width + dc.height),
    ])
    
    shape1.extend([
        (-x, y) for x, y in reversed(shape)
    ])
    
    # Remove duplicate points and add to final shape list
    new_shape = remove_repeated_vertices(shape, tolerance=1e-6)
    new_shape1 = remove_repeated_vertices(shape1, tolerance=1e-6)
    shapes.append(new_shape)
    shapes.append(new_shape1)
    
    
    # === Build the second DC electrode shape (upper wedge) ===
    shape = []
    shape1 = []
    
    shape.extend([
        (-rf.length/2 + trench_width/2, rf.length/2),
        (-rf.length/2 + trench_width/2, c.width/2 + rf.width + 2*trench_width + dc.height),
        (-trench_width/2, c.width/2 + rf.width + 2*trench_width + dc.height),
        (-trench_width/2, rf.length/2),
    ])
    
    shape1.extend([
        (-x, y) for x, y in reversed(shape)
    ])
    
    # Remove duplicate points and add to final shape list
    new_shape = remove_repeated_vertices(shape, tolerance=1e-6)
    new_shape1 = remove_repeated_vertices(shape1, tolerance=1e-6)
    shapes.append(new_shape)
    shapes.append(new_shape1)
    
    return shapes


def segmented_central_shapes(fivewire_geometry: FiveWireGeometry,
                             trench_width: float) -> list[list[tuple[float, float]]]:
    rf = fivewire_geometry.RF
    dc = fivewire_geometry.DC
    c = fivewire_geometry.C
    
    n_C = len(c.heights)
    
    shapes = []

    C_heights_new = []
    
    for h in c.heights:
        C_heights_new.append(trench_width)
        C_heights_new.append(h)
        
    C_heights_new[0] = trench_width/2
    C_heights_new.append(trench_width/2)
    
    points = [(sum(C_heights_new[:j]) - rf.length/2) for j in range(1, len(C_heights_new)+1)]
        
    for i in range(0, 2*n_C, 2):
        shape = []
        shape.extend([
            (points[i], c.width/2 - trench_width),
            (points[i], -c.width/2 + trench_width),
            (points[i+1], -c.width/2 + trench_width),
            (points[i+1], c.width/2 - trench_width)
        ])
        
        # Remove repeated points and append the final shape
        new_shape = remove_repeated_vertices(shape, tolerance=1e-6)
        shapes.append(new_shape)
    
    return shapes


def segmented_dc_shapes(fivewire_geometry: FiveWireGeometry,
                        trench_width: float) -> list[list[tuple[float, float]]]:
    rf = fivewire_geometry.RF
    dc = fivewire_geometry.DC
    c = fivewire_geometry.C
    
    shapes = []
    
    DC_heights_new = []
    
    for _ in range(dc.count):
        DC_heights_new.append(trench_width)
        DC_heights_new.append(dc.width)
    
    DC_heights_new[0] = trench_width/2
    DC_heights_new.append(trench_width/2)
    
    points = [(sum(DC_heights_new[:j]) - rf.length/2) for j in range(1, len(DC_heights_new)+1)]
    
    for i in range(0, 2*dc.count, 2):
        shape = []
        shape.extend([
            (points[i], rf.length/2),
            (points[i], c.width/2 + rf.width + trench_width),
            (points[i+1], c.width/2 + rf.width + trench_width),
            (points[i+1], rf.length/2)
        ])
        
        # Remove repeated points and append the final shape
        new_shape = remove_repeated_vertices(shape, tolerance=1e-6)
        shapes.append(new_shape)
    
    return shapes



def central_RF(threerf_geometry: ThreeRFGeometry):
    central_rf = threerf_geometry.CENTRAL
    outer_rf = threerf_geometry.OUTER
    
    shape = []
    
    # Add fixed endcap segment
    shape.extend([
        (-central_rf.width/2, -central_rf.length/2),
        (central_rf.width/2, -central_rf.length/2),
        (central_rf.width/2, central_rf.length/2),
        (-central_rf.width/2, central_rf.length/2),
    ])
    
    # Remove repeated vertices
    new_shape = remove_repeated_vertices(shape, tolerance=1e-6)
    
    return new_shape


def outer_RF(threerf_geometry: ThreeRFGeometry):
    central_rf = threerf_geometry.CENTRAL
    outer_rf = threerf_geometry.OUTER
    
    shape = []
    
    # Add fixed endcap segment
    shape.extend([
        (-outer_rf.width - outer_rf.offset, -outer_rf.length/2),
        (-outer_rf.offset, -outer_rf.length/2),
        (-outer_rf.offset, outer_rf.length/2),
        (-outer_rf.width - outer_rf.offset, outer_rf.length/2),
    ])
    
    # Remove repeated vertices
    new_shape = remove_repeated_vertices(shape, tolerance=1e-6)
    
    return new_shape
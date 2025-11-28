#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File: curved_shapes.py
# Created: 15-07-2025
# Author: Lorenzo Calandra Buonaura <lorenzocb01@gmail.com>
# Institution: University of Innsbruck - UIBK
#
# Functions defining electrode shapes for the turn trap geometry.
#


########################
# IMPORT ZONE          #
########################

import numpy as np

from src.geometry.modules import TurnGeometry
from src.geometry.geometry_utils import remove_repeated_vertices, arc, splitted_arc, shifted_line_params, single_intersection


########################
# FUNCTIONS            #
########################

def rf_shapes(turn_geometry: TurnGeometry) -> list[tuple[float, float]]:
    rf = turn_geometry.RF
    dc = turn_geometry.DC
    c = turn_geometry.C
  
    shapes = []

    radius = (rf.length - (2*rf.width + c.width))/2
    
    arc1 = arc(radius)
    arc2 = arc(radius + rf.width)
    arc3 = arc(radius + rf.width + c.width)
    arc4 = arc(radius + 2*rf.width + c.width)
    
    
    # === Build the first RF electrode shape (inner) ===    
    shape = []
    
    # Add fixed endcap segment
    shape.extend([(x, y) for x, y in reversed(arc1)])
    shape.extend([(x, y) for x, y in (arc2)])

    # Remove repeated vertices and add to final shapes list
    new_shape = remove_repeated_vertices(shape, tolerance=1e-6)
    shapes.append(new_shape)
    
    
    # === Build the second RF electrode shape (outer) ===    
    shape = []
    
    # Add fixed endcap segment
    shape.extend([(x, y) for x, y in reversed(arc3)])
    shape.extend([(x, y) for x, y in (arc4)])

    # Remove repeated vertices and add to final shapes list
    new_shape = remove_repeated_vertices(shape, tolerance=1e-6)
    shapes.append(new_shape)
  
    return shapes


def central_shape(turn_geometry: TurnGeometry,
                  trench_width: float) -> list[tuple[float, float]]:
    rf = turn_geometry.RF
    dc = turn_geometry.DC
    c = turn_geometry.C
  
    radius = (rf.length - c.width)/2 + trench_width
    
    arc1 = arc(radius, trench_width)
    arc2 = arc(radius + c.width - 2*trench_width, trench_width)
    
    # === Build the central electrode shape  ===    
    shape = []
    
    # Add fixed endcap segment
    shape.extend([(x, y) for x, y in reversed(arc1)])
    shape.extend([(x, y) for x, y in (arc2)])

    # Remove repeated vertices
    new_shape = remove_repeated_vertices(shape, tolerance=1e-6)
  
    return new_shape


def dc_shapes(turn_geometry: TurnGeometry,
              trench_width: float) -> list[list[tuple[float, float]]]:
    rf = turn_geometry.RF
    dc = turn_geometry.DC
    c = turn_geometry.C
    
    shapes = []
    
    radius = (rf.length - (2*rf.width + c.width))/2 - dc.height - trench_width
    
    arc1 = splitted_arc(radius, trench_width, 2)
    arc2 = splitted_arc(radius + dc.height, trench_width, 2)
    arc3 = splitted_arc(radius + 2*rf.width + c.width + 2*trench_width + dc.height, trench_width, 2)
    arc4 = splitted_arc(radius + 2*rf.width + c.width + 2*trench_width + 2*dc.height, trench_width, 2)
    arc5 = splitted_arc(radius - trench_width, trench_width, 2)
    arc6 = splitted_arc(radius + 2*rf.width + c.width + 3*trench_width + 2*dc.height, trench_width, 2)
    
    # === Build the first RF electrode shape (inner) ===
    for arc_inner, arc_outer in zip(arc1, arc2):
        shape = []

        shape.extend([(x, y) for x, y in reversed(arc_inner)])
        shape.extend([(x, y) for x, y in (arc_outer)])

        # Remove repeated vertices and add to final shapes list
        new_shape = remove_repeated_vertices(shape, tolerance=1e-6)
        shapes.append(new_shape)
    
    
    # === Build the second RF electrode shape (outer) ===    
    for arc_inner, arc_outer in zip(arc3, arc4):
        shape = []

        shape.extend([(x, y) for x, y in reversed(arc_inner)])
        shape.extend([(x, y) for x, y in (arc_outer)])

        # Remove repeated vertices and add to final shapes list
        new_shape = remove_repeated_vertices(shape, tolerance=1e-6)
        shapes.append(new_shape)
        
    
    # === Build compensation electrodes (inner) ===
    shape = []
        
    shape.extend([(x, y) for x, y in arc5[0]])
    shape.extend([((1 + np.sqrt(2)) * trench_width/2, trench_width/2)])
    
    new_shape = remove_repeated_vertices(shape, tolerance=1e-6)
    shapes.append(new_shape)
    
    shape = [(y, x) for x, y in reversed(new_shape)]
    new_shape = remove_repeated_vertices(shape, tolerance=1e-6)
    shapes.append(new_shape)
    
    # === Build compensation electrodes (outer) ===
    shape = []
        
    shape.extend([(x, y) for x, y in reversed(arc6[0])])
    shape.extend([
        (rf.length - trench_width/2, trench_width/2),
        (rf.length - trench_width/2, rf.length - (1 + np.sqrt(2)) * trench_width/2)
    ])
    
    new_shape = remove_repeated_vertices(shape, tolerance=1e-6)
    shapes.append(new_shape)
    
    shape = [(y, x) for x, y in reversed(new_shape)]
    new_shape = remove_repeated_vertices(shape, tolerance=1e-6)
    shapes.append(new_shape)
    
    return shapes


def segmented_central_shapes(turn_geometry: TurnGeometry,
                             trench_width: float) -> list[list[tuple[float, float]]]:
    rf = turn_geometry.RF
    dc = turn_geometry.DC
    c = turn_geometry.C
    
    shapes = []
    
    radius = (rf.length - c.width)/2 + trench_width
    
    arc1 = splitted_arc(radius, trench_width, c.count)
    arc2 = splitted_arc(radius + c.width - 2*trench_width, trench_width, c.count)
    
    # === Build the contral electrode shape ===
    for arc_inner, arc_outer in zip(arc1, arc2):
        shape = []

        shape.extend([(x, y) for x, y in reversed(arc_inner)])
        shape.extend([(x, y) for x, y in (arc_outer)])

        # Remove repeated vertices and add to final shapes list
        new_shape = remove_repeated_vertices(shape, tolerance=1e-6)
        shapes.append(new_shape)
  
    return shapes


def segmented_dc_shapes(turn_geometry: TurnGeometry,
                        trench_width: float) -> list[list[tuple[float, float]]]:
    rf = turn_geometry.RF
    dc = turn_geometry.DC
    c = turn_geometry.C
    
    shapes = []
    
    radius = (rf.length - (2*rf.width + c.width))/2 - dc.height - trench_width
    
    arc1 = splitted_arc(radius, trench_width, dc.count)
    arc2 = splitted_arc(radius + dc.height, trench_width, dc.count)
    arc3 = splitted_arc(radius + 2*rf.width + c.width + 2*trench_width + dc.height, trench_width, 2*dc.count)
    arc4 = splitted_arc(radius + 2*rf.width + c.width + 2*trench_width + 2*dc.height, trench_width, 2*dc.count)
    arc5 = splitted_arc(radius - trench_width, trench_width, 2)
    arc6 = splitted_arc(radius + 2*rf.width + c.width + 3*trench_width + 2*dc.height, trench_width, 2)

    # === Build the first RF electrode shape (inner) ===
    for arc_inner, arc_outer in zip(arc1, arc2):
        shape = []

        shape.extend([(x, y) for x, y in reversed(arc_inner)])
        shape.extend([(x, y) for x, y in (arc_outer)])

        # Remove repeated vertices and add to final shapes list
        new_shape = remove_repeated_vertices(shape, tolerance=1e-6)
        shapes.append(new_shape)
    
    
    # === Build the second RF electrode shape (outer) ===    
    for arc_inner, arc_outer in zip(arc3, arc4):
        shape = []

        shape.extend([(x, y) for x, y in reversed(arc_inner)])
        shape.extend([(x, y) for x, y in (arc_outer)])

        # Remove repeated vertices and add to final shapes list
        new_shape = remove_repeated_vertices(shape, tolerance=1e-6)
        shapes.append(new_shape)
        
    # === Build compensation electrodes (inner) ===
    shape = []
        
    shape.extend([(x, y) for x, y in arc5[0]])
    shape.extend([((1 + np.sqrt(2)) * trench_width/2, trench_width/2)])
    
    new_shape = remove_repeated_vertices(shape, tolerance=1e-6)
    shapes.append(new_shape)
    
    shape = [(y, x) for x, y in reversed(new_shape)]
    new_shape = remove_repeated_vertices(shape, tolerance=1e-6)
    shapes.append(new_shape)
    
    # === Build compensation electrodes (outer) ===
    shape = []
        
    shape.extend([(x, y) for x, y in reversed(arc6[0])])
    shape.extend([
        (rf.length - trench_width/2, trench_width/2),
        (rf.length - trench_width/2, rf.length - (1 + np.sqrt(2)) * trench_width/2)
    ])
    
    new_shape = remove_repeated_vertices(shape, tolerance=1e-6)
    shapes.append(new_shape)
    
    shape = [(y, x) for x, y in reversed(new_shape)]
    new_shape = remove_repeated_vertices(shape, tolerance=1e-6)
    shapes.append(new_shape)
    
    return shapes
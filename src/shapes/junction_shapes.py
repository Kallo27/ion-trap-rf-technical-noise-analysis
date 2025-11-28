#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File: junction_shapes.py
# Created: 10-10-2025
# Author: Lorenzo Calandra Buonaura <lorenzocb01@gmail.com>
# Institution: University of Innsbruck - UIBK
#
# Functions defining electrode shapes for the non-optimized X-junction trap geometry.
#


########################
# IMPORT ZONE          #
########################

import numpy as np
from shapely.geometry import Point, Polygon, LinearRing

from src.geometry.modules import JunctionGeometry
from src.geometry.geometry_utils import (remove_repeated_vertices,
                                         lines_intersections,
                                         single_intersection,
                                         shifted_line_params)


########################
# FUNCTIONS            #
########################

def rf_shapes(junction_geometry: JunctionGeometry) -> list[tuple[float, float]]:
    """
    Builds one L shape for the RF of the non-opt junction.

    Parameters
    ----------
    junction_geometry : JunctionGeometry
        Instance of JunctionGeometry that contains the geometrical dimensions of the junction.

    Returns
    -------
    list[tuple[float, float]]
        List of vertices for the RF electrodes of the junction.
    """
    rf = junction_geometry.RF
    dc = junction_geometry.DC
    c = junction_geometry.C

    shape = []
  
    # Add fixed endcap segment
    shape.extend([
        (-(c.width / 2 + rf.width), c.width / 2 + rf.width),
        (-rf.length / 2, c.width / 2 + rf.width),
        (-rf.length / 2, c.width / 2),
        (-c.width / 2, c.width / 2),
    ])

    # Mirror the shape across the y = -x line to complete symmetry
    shape.extend([(-y, -x) for x, y in reversed(shape)])
  
    # Remove repeated vertices
    new_shape = remove_repeated_vertices(shape, 1e-6)
  
    return new_shape


def central_shape(junction_geometry: JunctionGeometry,
                  trench_width: float) -> list[tuple[float, float]]:
    """
    Builds one central X-shape for the central electrode of the junction.
    It accepts bottom control points (lower segment). 

    Parameters
    ----------
    junction_geometry : JunctionGeometry
        Instance of JunctionGeometry that contains the geometrical dimensions of the junction.

    trench_width : float
        Width of the trenches between electrodes.

    Returns
    -------
    list[tuple[float, float]]
        List of vertices for the central electrode of the junction.
    """
    rf = junction_geometry.RF
    dc = junction_geometry.DC
    c = junction_geometry.C
    
    shape = []
    
    # Start with the leftmost fixed shape boundary points
    shape.extend([
        (-rf.length/2 + trench_width/2, -c.width/2 + trench_width),
        (-c.width/2 + trench_width, -c.width/2 + trench_width)
    ])
    
    # Mirror the shape to complete simmetry
    shape.extend([(point[1], point[0]) for point in reversed(shape)])
    shape.extend([(-point[0], point[1]) for point in reversed(shape)])
    shape.extend([(point[0], -point[1]) for point in reversed(shape)])

    shape = remove_repeated_vertices(shape, tolerance=1e-6)

    if c.radius is not None: 
        # Create the circle
        x0, y0 = 0, 0
        circle = Point(x0, y0).buffer(c.radius)
    
        # Compute the union between the shape and the circle
        combined_shape = circle.union(Polygon(shape))
        
        # Extract exterior coordinates
        exterior_coords = list(combined_shape.exterior.coords)

        # Ensure counter-clockwise orientation
        ring = LinearRing(exterior_coords)
        if not ring.is_ccw:
            exterior_coords.reverse()  # Make it CCW

        shape = exterior_coords
    
    # Remove repeated vertices
    new_shape = remove_repeated_vertices(shape, tolerance=1e-6)
    
    return new_shape


def dc_shapes(junction_geometry: JunctionGeometry,
              trench_width: float,
              single: float) -> list[list[tuple[float, float]]]:
    """
    Builds two outer DC control electrodes of the junction (ETH style).
    It accepts top control points (upper segment).

    Parameters
    ----------
    junction_geometry : JunctionGeometry
        Instance of JunctionGeometry that contains the geometrical dimensions of the junction.
    trench_width : float
        Width of the trenches between electrodes.
    single : float
        Flag for single outer DC
    Returns
    -------
    list[list[tuple[float, float]]]
        Shapes of the two DC electrodes for the junction.
    """
    rf = junction_geometry.RF
    dc = junction_geometry.DC
    c = junction_geometry.C
    
    DC_shapes = []
    
    if single:
        DC_shapes.extend([[
            (-(c.width/2 + rf.width + trench_width), c.width/2 + rf.width + trench_width),
            (-(c.width/2 + rf.width + trench_width), rf.length/2 - trench_width/2),
            (-rf.length/2 + trench_width/2, rf.length/2 - trench_width/2),
            (-rf.length/2 + trench_width/2, c.width/2 + rf.width + trench_width),
        ]])
                
        return DC_shapes
    
    # === Build the first DC electrode shape ===    
    shape = []
    
    # Define the wedge corner points
    point1 = (-c.width/2 - rf.width - trench_width, c.width/2 + rf.width + trench_width)
    point2 = (-(c.width/2 + rf.width + (1+1/np.sqrt(2))*trench_width + dc.height), c.width/2 + rf.width + (1+1/np.sqrt(2))*trench_width + dc.height)

    # Get shifted line parameters for the wedge edge
    m, q = shifted_line_params(point1, point2, -trench_width/2)
    
    # Construct the shape using the shifted line
    shape.extend([
        (-rf.length/2 + trench_width/2, c.width/2 + rf.width + trench_width + dc.height),
        (-rf.length/2 + trench_width/2, c.width/2 + rf.width + trench_width),
        ((point1[1] - q) / m, point1[1]),
        (point2[0], m * point2[0] + q)
    ])
        
    # Remove duplicate points and add to final shape list
    new_shape = remove_repeated_vertices(shape, tolerance=1e-6)
    DC_shapes.append(new_shape)
    
    
    # === Build the second DC electrode shape (upper wedge) ===
    shape = [] 

    # Define the wedge corner points
    point1 = (-c.width/2 - rf.width - dc.height - 2*trench_width, c.width/2 + rf.width + dc.height + 2*trench_width)
    point2 = (-rf.length/2 + trench_width/2, rf.length/2 - trench_width/2)

    # Get shifted line parameters for the wedge edge
    m, q = shifted_line_params(point1, point2, -trench_width/2)
    
    # Construct the shape using the shifted line
    shape.extend([
        (-rf.length/2 + trench_width/2, c.width/2 + rf.width + dc.height + 2*trench_width),
        ((point1[1] - q) / m, point1[1]),
        (point2[0], m * point2[0] + q)
    ])
    
    # Remove repeated points and append the final shape
    new_shape = remove_repeated_vertices(shape, tolerance=1e-6)    
    DC_shapes.append(shape)
    
    return DC_shapes


def segmented_central_shapes(junction_geometry: JunctionGeometry,
                             trench_width: float) -> list[list[tuple[float, float]]]:
    """
    Builds the inner segmented DC control electrodes of the junction (ETH style).
    It accepts bottom control points (lower segment).

    Parameters
    ----------
    junction_geometry : JunctionGeometry
        Instance of JunctionGeometry that contains the geometrical dimensions of the junction.
    trench_width : float
        Width of the trenches between electrodes.

    Returns
    -------
    list[list[tuple[float, float]]]
        Shapes of the inner segmented DC electrodes.
    """
    rf = junction_geometry.RF
    dc = junction_geometry.DC
    c = junction_geometry.C
    
    step = 2 if trench_width !=0 else 1
    n_C = len(c.heights)

    shapes = []

    if trench_width != 0:
        # Insert trench widths between each conductor segment
        C_heights_new = []
        for h in c.heights:
            C_heights_new.append(h)
            C_heights_new.append(trench_width)
        c.heights = C_heights_new[:-1]

    bottom_segments = []

    # Build bottom half of segmented shapes with interpolation
    for step_index in range(step * n_C + 1):
        x = -sum(c.heights[:step_index])
        y = c.width / 2 - trench_width
            
        bottom_segments.append((x, y))
        
    # === Build first central shape (largest) ===
    shape = []
    shape.extend([(-c.width / 2 + trench_width, c.width / 2 - trench_width),
                  (-c.heights[0], c.width / 2 - trench_width)])
    shape.extend([(point[0], -point[1]) for point in reversed(shape)])
    shape.extend([(-point[1], point[0]) for point in shape])
    shape.extend([(-point[1], -point[0]) for point in reversed(shape)])
    
    shape = remove_repeated_vertices(shape, tolerance=1e-6)
    
    if c.radius is not None: 
        # Create the circle
        x0, y0 = 0, 0
        circle = Point(x0, y0).buffer(c.radius)
    
        # Compute the union between the shape and the circle
        combined_shape = circle.union(Polygon(shape))
        
        # Extract exterior coordinates
        exterior_coords = list(combined_shape.exterior.coords)

        # Ensure counter-clockwise orientation
        ring = LinearRing(exterior_coords)
        if not ring.is_ccw:
            exterior_coords.reverse()  # Make it CCW

        shape = exterior_coords

    # Remove repeated points and append the final shape
    new_shape = remove_repeated_vertices(shape, tolerance=1e-6)
    shapes.append(new_shape)
    
    # === Build remaining shapes ===
    for j in range(step, step*n_C, step):
        shape = []
        
        # Compute limits
        lower_limit = -(sum(c.heights[:j+1]))
        upper_limit = -(sum(c.heights[:j]))
        
        # Extend with points inside the limits (lower and upper segment)
        shape.extend([point for point in bottom_segments if point[0] >= lower_limit and point[0] <= upper_limit])
        shape.extend([(point[0], -point[1]) for point in reversed(shape)])
    
        # Remove repeated points and append the final shape
        new_shape = remove_repeated_vertices(shape, tolerance=1e-6)
        shapes.append(new_shape)
        
    return shapes


def segmented_dc_shapes(junction_geometry: JunctionGeometry,
                        trench_width: float) -> list[list[tuple[float, float]]]:
    """
    Builds the outer segmented DC control electrodes of the junction (linear trap style).
    It accepts top control points (upper segment).

    Parameters
    ----------
    junction_geometry : JunctionGeometry
        Instance of JunctionGeometry that contains the geometrical dimensions of the junction.
    trench_width : float
        Width of the trenches between electrodes.

    Returns
    -------
    list[list[tuple[float, float]]]
        Shapes of the outer segmented DC electrodes.
    """
    rf = junction_geometry.RF
    dc = junction_geometry.DC
    c = junction_geometry.C
    
    step = 2 if trench_width !=0 else 1
    
    shapes = []
    
    # Build sequence of alternating DC widths and trench gaps
    DC_widths = []
    for i in range(dc.count):
        DC_widths.append(dc.width)
        DC_widths.append(trench_width)
    DC_widths = np.array(DC_widths[:-1])
    DC_widths = DC_widths[DC_widths != 0]
    
    bottom_segments = []
    
    # Build bottom half of segmented shapes with interpolation
    for step_index in range(step * dc.count + 1):
        x = -sum(DC_widths[:step_index])
        y = rf.width + c.width/2 + trench_width
        bottom_segments.append((x, y))

    # Reverse to maintain the correct order
    bottom_segments = bottom_segments[::-1]
    
    # === Build first shape (trapezoidal) ===
    shape = []

    # Define the wedge corner points
    point1 = (-c.width/2 - rf.width - trench_width, c.width/2 + rf.width + trench_width)
    point2 = (-(c.width/2 + rf.width + (1+1/np.sqrt(2))*trench_width + dc.height), c.width/2 + rf.width + (1+1/np.sqrt(2))*trench_width + dc.height)

    # Get shifted line parameters for the wedge edge
    m, q = shifted_line_params(point1, point2, -trench_width/2)
    
    # Construct the shape using the shifted line
    shape.extend([
        (bottom_segments[-2][0] + ((point1[1] - q) / m), m * point2[0] + q),
        (bottom_segments[-2][0] + ((point1[1] - q) / m), point1[1]),
        ((point1[1] - q) / m, point1[1]),
        (point2[0], m * point2[0] + q)
    ])
    
    # Remove repeated points and append the final shape
    new_shape = remove_repeated_vertices(shape, tolerance=1e-6)
    shapes.append(new_shape)
    
    # Shift trench_width left to start next DC shape
    first_point = (shape[0][0] - trench_width, shape[1][1])

    # === Build remaining segmented shapes ===
    for j in range(step, step*dc.count, step):
        shape = [first_point]
        
        # Compute limits
        lower_limit = first_point[0] - dc.width
        
        # Extend with points inside the limits (lower and upper segment)
        shape.extend([(first_point[0], m * point2[0] + q), 
                      (lower_limit, m * point2[0] + q), 
                      (lower_limit, point1[1])])
        
        # Prepare the new first_point for next segment (shift again left)
        first_point = (shape[-1][0] - trench_width, first_point[1])
        
        # Remove repeated points and append the final shape
        new_shape = remove_repeated_vertices(shape, tolerance=1e-3)
        shapes.append(new_shape)
        
    # === Build the upper wedge DC electrode shape ===
    shape = []    

    # Define the wedge corner points
    point1 = (-c.width/2 - rf.width - dc.height - 2*trench_width,
              c.width/2 + rf.width + dc.height + 2*trench_width)
    point2 = (-rf.length/2 + trench_width/2, rf.length/2 - trench_width/2)

    # Get shifted line parameters for the wedge edge
    m, q = shifted_line_params(point1, point2, -trench_width/2)
    
    # Construct the shape using the shifted line
    shape.extend([
        (-rf.length/2 + trench_width/2, c.width/2 + rf.width + dc.height + 2*trench_width),
        ((point1[1] - q) / m, point1[1]),
        (point2[0], m * point2[0] + q)
    ])
    
    # Remove repeated points and append the final shape
    new_shape = remove_repeated_vertices(shape, tolerance=1e-6)    
    shapes.append(shape)
        
    return shapes
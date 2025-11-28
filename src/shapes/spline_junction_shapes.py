#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File: spline_junction_electrodes.py
# Created: 10-04-2025
# Author: Lorenzo Calandra Buonaura <lorenzocb01@gmail.com>
# Institution: University of Innsbruck - UIBK
#
# Functions defining electrode shapes for the spline-optimized X-junction trap geometry.
#


########################
# IMPORT ZONE          #
########################

import numpy as np
from scipy.interpolate import make_interp_spline
from shapely.geometry import Point, Polygon, LinearRing

from src.geometry.modules import JunctionGeometry
from src.geometry.geometry_utils import (remove_repeated_vertices,
                                         shifted_line_params,
                                         shift_spline)

########################
# FUNCTIONS            #
########################

def spline_rf_shapes(junction_geometry: JunctionGeometry,
                     bottom_points: list[np.ndarray],
                     top_points: list[np.ndarray]) -> list[tuple[float, float]]:
    """
    Builds one L shape for the RF of the junction (spline). It accepts control points (both top and bottom).

    Parameters
    ----------
    junction_geometry : JunctionGeometry
        Instance of JunctionGeometry that contains the geometrical dimensions of the junction.
    bottom_points : list[np.ndarray]
        List of bottom control points (x, y).
    top_points : list[np.ndarray]
        List of top control points (x, y).

    Returns
    -------
    list[tuple[float, float]]
        List of vertices for the RF electrodes of the junction (spline).
    """
    rf = junction_geometry.RF
    dc = junction_geometry.DC
    c = junction_geometry.C

    X_bottom, Y_bottom = bottom_points
    X_top, Y_top = top_points

    shape = []
    
    Y_top[0] = rf.width + c.width/2
    Y_bottom[0] = c.width/2
    
    # Cubic spline interpolation with boundary conditions
    bc_type = ([(1, 0.0)], [(1, 0.0)])
    spline_top = make_interp_spline(X_top, Y_top, bc_type=bc_type, k=3)
    spline_bottom = make_interp_spline(X_bottom, Y_bottom, bc_type=bc_type, k=3)

    # Sample interpolated points and evaluate spline
    num_samples = 50
    X_top = np.linspace(min(X_top), max(X_top), num_samples)
    Y_top = spline_top(X_top)
    X_bottom = np.linspace(min(X_bottom), max(X_bottom), num_samples)
    Y_bottom = spline_bottom(X_bottom)

    # Start with the topmost point on the y = -x line (symmetric anchor)
    shape.append((-Y_top[-1], Y_top[-1]))

    # Add top control points in counter-clockwise order
    for i in range(1, num_samples+1):
        shape.append((X_top[-i], Y_top[-i]))
    
    # Add fixed endcap segment
    shape.extend([
        (-rf.x_opt, c.width/2 + rf.width),
        (-rf.length/2, c.width/2 + rf.width), 
        (-rf.length/2, c.width/2), 
        (-rf.x_opt, c.width/2)
    ])

    # Add bottom control points in counter-clockwise order
    for i in range(1, num_samples):
        shape.append((X_bottom[i], Y_bottom[i]))

    # Add bottom-most point on the y = -x line (symmetric anchor)
    shape.append((-Y_bottom[-1], Y_bottom[-1]))

    # Mirror the shape across the y = -x line to complete symmetry
    shape.extend([(-y, -x) for x, y in reversed(shape)])
  
    # Remove repeated vertices
    new_shape = remove_repeated_vertices(shape, 1e-6)
  
    return new_shape


def spline_central_shape(junction_geometry: JunctionGeometry,
                         control_points: list[np.ndarray],
                         trench_width: float) -> list[list[tuple[float, float]]]:
    """
    Builds one central X-shape for the central electrode of the junction (spline).
    It accepts bottom control points (lower segment). 

    Parameters
    ----------
    junction_geometry : JunctionGeometry
        Instance of JunctionGeometry that contains the geometrical dimensions of the junction.
    control_points : list[np.ndarray]
        List of bottom control points (x, y).
    trench_width : float
        Width of the trenches between electrodes.

    Returns
    -------
    list[tuple[float, float]]
        List of vertices for the central electrode of the junction (spline).
    """
    rf = junction_geometry.RF
    dc = junction_geometry.DC
    c = junction_geometry.C
    
    X_bottom, Y_bottom = control_points
    Y_bottom = list(-np.array(Y_bottom))
    
    shape = []
    
    # Cubic spline interpolation with boundary conditions
    bc_type = ([(1, 0.0)], [(1, 0.0)])
    spline_bottom = make_interp_spline(X_bottom, Y_bottom, bc_type=bc_type, k=3)

    # Sample interpolated points and evaluate spline
    num_samples = 50
    X_bottom = np.linspace(min(X_bottom), max(X_bottom), num_samples)
    Y_bottom = spline_bottom(X_bottom)
    
    # Compute shifted spline
    X_bottom, Y_bottom = shift_spline([X_bottom, Y_bottom], trench_width)
    X_bottom = np.concatenate((X_bottom[1:], [Y_bottom[-1]]))
    Y_bottom = np.concatenate((Y_bottom[1:], [Y_bottom[-1]]))
    
    # Start with the leftmost fixed shape boundary points
    shape.extend([
        (-rf.length/2 + trench_width/2, -c.width/2 + trench_width),
        (-rf.x_opt, -c.width/2 + trench_width)
    ])

    # Extend shape with new spline points
    for i in range(num_samples):
        shape.append((X_bottom[i], Y_bottom[i]))

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



def spline_dc_shapes(junction_geometry: JunctionGeometry,
                     control_points: list[np.ndarray],
                     trench_width: float) -> list[list[tuple[float, float]]]:
    """
    Builds two outer DC control electrodes of the junction (spline, ETH style).
    It accepts top control points (upper segment).

    Parameters
    ----------
    junction_geometry : JunctionGeometry
        Instance of JunctionGeometry that contains the geometrical dimensions of the junction.
    control_points : list[np.ndarray]
        List of top control points (x, y).
    trench_width : float
        Width of the trenches between electrodes.

    Returns
    -------
    list[list[tuple[float, float]]]
        Shapes of the two DC electrodes for the junction (spline).
    """
    rf = junction_geometry.RF
    dc = junction_geometry.DC
    c = junction_geometry.C
    
    X_top, Y_top = control_points
    DC_shapes = []
    
    # Cubic spline interpolation with boundary conditions
    bc_type = ([(1, 0.0)], [(1, 0.0)])
    spline_bottom = make_interp_spline(X_top, Y_top, bc_type=bc_type, k=3)

    # Sample interpolated points and evaluate spline
    num_samples = 50
    X_top = np.linspace(min(X_top), max(X_top), num_samples)
    Y_top = spline_bottom(X_top)

    # Compute shifted spline   
    X_top, Y_top = shift_spline([X_top, Y_top], trench_width)
    X_top = np.concatenate((X_top[1:], [-Y_top[-1]]))
    Y_top = np.concatenate((Y_top[1:], [Y_top[-1]]))

    # === Build the first DC electrode shape ===
    shape = []
    
    # Start the shape with leftmost endcap boundary 
    shape.extend([
        (-rf.length/2 + trench_width/2, c.width/2 + rf.width + trench_width + dc.height),
        (-rf.length/2 + trench_width/2, c.width/2 + rf.width + trench_width),
        (-rf.x_opt, c.width/2 + rf.width + trench_width)
    ])

    for i in range(num_samples-1):
        shape.append((X_top[i], Y_top[i]))
    
    # Define shifted points on the y=-x axis and compute lines params
    point1 = (X_top[-1], Y_top[-1])
    point2 = (-c.width/2 - rf.width - trench_width - dc.height, c.width/2 + rf.width + trench_width + dc.height)
    m, q = shifted_line_params(point1, point2, -trench_width/2)
    
    # Extend shape with computed points
    shape.extend([
        ((point1[1] - q) / m, point1[1]),
        ((point2[1] - q) / m, point2[1]),
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


def spline_segmented_central_shapes(junction_geometry: JunctionGeometry,
                                    control_points: list[np.ndarray],
                                    trench_width: float) -> list[list[tuple[float, float]]]:
    """
    Builds the inner segmented DC control electrodes of the junction (spline, ETH style).
    It accepts bottom control points (lower segment).

    Parameters
    ----------
    junction_geometry : JunctionGeometry
        Instance of JunctionGeometry that contains the geometrical dimensions of the junction.
    control_points : list[np.ndarray]
        List of bottom control points (x, y).
    trench_width : float
        Width of the trenches between electrodes.

    Returns
    -------
    list[list[tuple[float, float]]]
        Shapes of the inner segmented DC electrodes for the junction (spline).
    """
    rf = junction_geometry.RF
    dc = junction_geometry.DC
    c = junction_geometry.C
    
    X_bottom, Y_bottom = control_points
    step = 2 if trench_width !=0 else 1
    n_C = len(c.heights)
    
    shapes = []
    
    # Cubic spline interpolation with boundary conditions
    bc_type = ([(1, 0.0)], [(1, 0.0)])
    spline_bottom = make_interp_spline(X_bottom, Y_bottom, bc_type=bc_type, k=3)

    # Sample interpolated points and evaluate spline
    num_samples = 50
    X_bottom = np.linspace(min(X_bottom), max(X_bottom), num_samples)
    Y_bottom = spline_bottom(X_bottom)

    # Compute shifted spline
    X_bottom, Y_bottom = shift_spline([X_bottom, Y_bottom], -trench_width)
    X_bottom = np.concatenate((X_bottom[1:], [-Y_bottom[-1]]))
    Y_bottom = np.concatenate((Y_bottom[1:], [Y_bottom[-1]]))
    spline_bottom = make_interp_spline(X_bottom, Y_bottom, bc_type=bc_type, k=3)

    # Insert trench widths between each conductor segment
    if trench_width != 0:
        C_heights_new = []
        for h in c.heights:
            C_heights_new.append(h)
            C_heights_new.append(trench_width)
        
        c.heights = C_heights_new[:-1]
        
    i = -1
    bottom_segments = []

    # Build bottom half of segmented shapes with interpolation
    for step_index in range(step * n_C + 1):
        target_x = -sum(c.heights[:step_index])

        # Add all points from the control list that are to the right of target_x
        while -i <= len(X_bottom) and X_bottom[i] > target_x:
            bottom_segments.append((X_bottom[i], Y_bottom[i]))
            i -= 1

        # At each step (excluding the first), interpolate or assign a fallback point
        if step_index != 0:
            if -i <= len(X_bottom):
                y_interp = spline_bottom(target_x).item()
            else:
                y_interp = c.width / 2 - trench_width
            
            bottom_segments.append((target_x, y_interp))

    # Reverse to maintain the correct order    
    bottom_segments = bottom_segments[::-1]

    # === Build first central shape (largest) ===
    shape = []
    shape.extend([point for point in bottom_segments[::-1] if point[0] >= -c.heights[0]])
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
        shape.extend([point for point in bottom_segments[::-1] if point[0] >= lower_limit and point[0] <= upper_limit])
        shape.extend([(point[0], -point[1]) for point in reversed(shape)])
    
        # Remove repeated points and append the final shape
        new_shape = remove_repeated_vertices(shape, tolerance=1e-6)
        shapes.append(new_shape)
        
    return shapes


def spline_segmented_dc_shapes(junction_geometry: JunctionGeometry,
                               control_points: list[np.ndarray],
                               trench_width: float) -> list[list[tuple[float, float]]]:
    """
    Builds the outer segmented DC control electrodes of the junction (spline, linear trap style).
    It accepts top control points (upper segment).

    Parameters
    ----------
    junction_geometry : JunctionGeometry
        Instance of JunctionGeometry that contains the geometrical dimensions of the junction.
    control_points : list[np.ndarray]
        List of bottom control points (x, y).
    trench_width : float
        Width of the trenches between electrodes.

    Returns
    -------
    list[list[tuple[float, float]]]
        Shapes of the outer segmented DC electrodes for the junction (spline).
    """
    rf = junction_geometry.RF
    dc = junction_geometry.DC
    c = junction_geometry.C
    
    X_top, Y_top = control_points
    step = 2 if trench_width !=0 else 1
    
    shapes = []
    
    # Cubic spline interpolation with boundary conditions
    bc_type = ([(1, 0.0)], [(1, 0.0)])
    spline_bottom = make_interp_spline(X_top, Y_top, bc_type=bc_type, k=3)

    # Sample interpolated points and evaluate spline
    num_samples = 50
    X_top = np.linspace(min(X_top), max(X_top), num_samples)
    Y_top = spline_bottom(X_top)

    # Compute shifted spline
    X_top, Y_top = shift_spline([X_top, Y_top], trench_width)
    X_top = np.concatenate(([X_top[0]], X_top[1:], [-Y_top[-1]]))
    Y_top = np.concatenate(([c.width/2 + rf.width + trench_width], Y_top[1:], [Y_top[-1]]))
    spline_bottom = make_interp_spline(X_top, Y_top, bc_type=bc_type, k=3)

    # Build sequence of alternating DC widths and trench gaps
    DC_widths = []
    for i in range(dc.count):
        DC_widths.append(dc.width)
        DC_widths.append(trench_width)
    DC_widths = np.array(DC_widths[:-1])
    DC_widths = DC_widths[DC_widths != 0]
        
    i = -1
    bottom_segments = []
    
    # Build bottom half of segmented shapes with interpolation
    for step_index in range(step * dc.count + 1):
        target_x = -sum(DC_widths[:step_index]) + X_top[-1]
        
        # Add all points from the control list that are to the right of target_x
        while -i <= len(X_top) and X_top[i] > target_x:
            bottom_segments.append((X_top[i], Y_top[i]))
            i -= 1
        
        # At each step (excluding the first), interpolate or assign a fallback point
        if step_index != 0:
            if -i <= len(X_top):
                y_interp = spline_bottom(target_x).item()
            else:
                y_interp = rf.width + c.width/2 + trench_width

            bottom_segments.append((target_x, y_interp))

    # Reverse to maintain the correct order
    bottom_segments = bottom_segments[::-1]


    # === Build first shape (trapezoidal) ===
    shape = []
    
    # Define shifted points on the y=-x axis and compute shifted line params
    point1 = (X_top[-1], Y_top[-1])
    point2 = (-c.width/2 - rf.width - trench_width - dc.height, c.width/2 + rf.width + trench_width + dc.height)
    m, q = shifted_line_params(point1, point2, -trench_width/2)
    
    # Extend shape with computed points
    shape.extend([
        ((point1[1] - q) / m, point1[1]),
        ((point2[1] - q) / m, point2[1])
    ])

    shape.extend([(point[0], point2[1]) for point in bottom_segments[::-1] if point[0] >= -DC_widths[0] + bottom_segments[-1][0] and point[0] <= shape[-1][0]])

    shape.extend([point for point in bottom_segments if point[0] >= -DC_widths[0] + bottom_segments[-1][0] and point[0] < bottom_segments[-1][0]])
    
    # Remove repeated points and append the final shape
    new_shape = remove_repeated_vertices(shape, tolerance=1e-6)    
    shapes.append(new_shape)
    
    # === Initialize first_point for next segments ===     
    for i in range(1, len(shape)):
        if shape[i][1] != shape[1][1]:
            break
        
    # Shift trench_width left to start next DC shape
    first_point = (shape[i][0] - trench_width, shape[1][1])

    # === Build remaining shapes ===
    for j in range(step, step*dc.count, step):
        shape = [first_point]
        
        # Compute limits
        lower_limit = bottom_segments[-1][0] - sum(DC_widths[:j+1])
        upper_limit = bottom_segments[-1][0] - sum(DC_widths[:j])
        
        # Extend with points inside the limits (lower and upper segment)
        shape.extend([(point[0], first_point[1]) for point in bottom_segments[::-1] if point[0] >= lower_limit and point[0] <= upper_limit])
        shape.extend([point for point in bottom_segments if point[0] >= lower_limit and point[0] <= upper_limit])
        
        # Prepare the new first_point for next segment (shift again left)
        first_point = (bottom_segments[-1][0] - sum(DC_widths[:j+step]), first_point[1])
        
        # Remove repeated points and append the final shape
        new_shape = remove_repeated_vertices(shape, tolerance=1e-3)
        shapes.append(new_shape)
        
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
    shapes.append(shape)
        
    return shapes
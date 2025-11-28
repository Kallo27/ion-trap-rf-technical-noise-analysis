#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File: piecewise_junction_electrodes.py
# Created: 10-04-2025
# Author: Lorenzo Calandra Buonaura <lorenzocb01@gmail.com>
# Institution: University of Innsbruck - UIBK
#
# Functions defining electrode shapes for the piecewise-optimized X-junction trap geometry.
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

def pw_rf_shapes(junction_geometry: JunctionGeometry,
                 bottom_points: list[np.ndarray],
                 top_points: list[np.ndarray]) -> list[tuple[float, float]]:
    """
    Builds one L shape for the RF of the junction. It accepts control points (both top and bottom).

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
        List of vertices for the RF electrodes of the junction.
    """
    rf = junction_geometry.RF
    dc = junction_geometry.DC
    c = junction_geometry.C
  
    X_bottom, Y_bottom = bottom_points
    X_top, Y_top = top_points
  
    shape = []
  
    # Start with the topmost point on the y = -x line (symmetric anchor)
    shape.append((-Y_top[-1], Y_top[-1]))
  
    # Add top control points in counter-clockwise order
    for i in range(len(X_top)):
        shape.append((X_top[-i - 1], Y_top[-i - 2]))
  
    # Add fixed endcap segment
    shape.extend([
        (-rf.x_opt, c.width / 2 + rf.width),
        (-rf.length / 2, c.width / 2 + rf.width),
        (-rf.length / 2, c.width / 2),
        (-rf.x_opt, c.width / 2),
    ])
  
    # Add bottom control points in counter-clockwise order
    for i in range(len(X_bottom)):
        shape.append((X_bottom[i], Y_bottom[i]))
  
    # Add bottom-most point on the y = -x line (symmetric anchor)
    shape.append((-Y_bottom[-1], Y_bottom[-1]))
  
    # Mirror the shape across the y = -x line to complete symmetry
    shape.extend([(-y, -x) for x, y in reversed(shape)])
  
    # Remove repeated vertices
    new_shape = remove_repeated_vertices(shape, 1e-6)
  
    return new_shape


def pw_central_shape(junction_geometry: JunctionGeometry,
                     control_points: list[np.ndarray],
                     trench_width: float) -> list[tuple[float, float]]:
    """
    Builds one central X-shape for the central electrode of the junction.
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
        List of vertices for the central electrode of the junction.
    """
    rf = junction_geometry.RF
    dc = junction_geometry.DC
    c = junction_geometry.C
    
    X_bottom, Y_bottom = control_points
    shape = []
    
    if trench_width != 0:
        # Extend X and Y to define the trench geometry including flipped segments
        X_bottom = np.concatenate(([-rf.x_opt], X_bottom, [-Y_bottom[-1]], [-Y_bottom[-2]]))
        Y_bottom = np.concatenate(([-c.width/2 + trench_width], -np.array(Y_bottom), [X_bottom[-3]]))
        
        # Compute intersection points with angle correction
        int_points = lines_intersections([X_bottom, Y_bottom], trench_width, angle=True)
    else:
        # Build mirrored control points without trench
        int_points = [(X_bottom[i], -Y_bottom[i]) for i in range(len(X_bottom+1))]
        int_points.append((-Y_bottom[-1], -Y_bottom[-1]))
    
    # Start with the leftmost fixed shape boundary points
    shape.extend([
        (-rf.length/2 + trench_width/2, -c.width/2 + trench_width),
        (-rf.x_opt, -c.width/2 + trench_width)
    ])
    
    # Extend shape with intersection/control points
    shape.extend(int_points)
    
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


def pw_dc_shapes(junction_geometry: JunctionGeometry,
                 control_points: list[np.ndarray],
                 trench_width: float) -> list[list[tuple[float, float]]]:
    """
    Builds two outer DC control electrodes of the junction (ETH style).
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
        Shapes of the two DC electrodes for the junction.
    """
    rf = junction_geometry.RF
    dc = junction_geometry.DC
    c = junction_geometry.C
    
    X_top, Y_top = control_points
    DC_shapes = []
    
    # === Build the first DC electrode shape ===    
    shape = []
    
    if trench_width != 0:
        # Extend X and Y to define the trench geometry including flipped segments
        X_top = np.concatenate(([-rf.length, -rf.x_opt], X_top, [-Y_top[-1]], [-Y_top[-2]]))
        Y_top = np.concatenate(([c.width/2 + rf.width, c.width/2 + rf.width], np.array(Y_top), [-X_top[-3]]))
        
        # Compute intersection points
        int_points = lines_intersections([X_top, Y_top], trench_width, angle=False)
        
        # Define shifted points on the y=-x axis
        point1 = (-Y_top[-1], Y_top[-1])
        point2 = (-c.width/2 - rf.width - dc.height - trench_width, c.width/2 + rf.width + dc.height + trench_width)
        
        # Compute lines params (shifted and un-shifted)
        m2, q2 = shifted_line_params(point1, point2, -trench_width/2)
        m1, q1 = shifted_line_params(int_points[-2], int_points[-1], 0)
        
        # Compute intersection for smooth electrode termination
        intersection = single_intersection([m1, q1], [m2, q2])
        
        # Replace last point with new intersections
        int_points = int_points[:-1]
        int_points.extend([
            intersection, 
            ((point2[1] - q2) / m2, point2[1])
        ])
                        
    else:
        # No trench: directly use and extend control points
        int_points = [(X_top[i], Y_top[i]) for i in range(len(X_top))]
        int_points.extend([
            (-Y_top[-1], Y_top[-1]),
            (-c.width/2 - rf.width - dc.height, c.width/2 + rf.width + dc.height)
        ])

    # Start the shape at the leftmost boundary
    shape.extend([
        (-rf.length/2 + trench_width/2, c.width/2 + rf.width + trench_width),
        (-rf.x_opt, c.width/2 + rf.width + trench_width)
    ])
    
    # Add the intersection/extension points
    shape.extend(int_points)

    # Close the shape with the topmost corner
    shape.append((-rf.length/2 + trench_width/2, c.width/2 + rf.width + dc.height + trench_width))
        
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


def pw_segmented_central_shapes(junction_geometry: JunctionGeometry,
                                control_points: list[np.ndarray],
                                trench_width: float) -> list[list[tuple[float, float]]]:
    """
    Builds the inner segmented DC control electrodes of the junction (ETH style).
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
        Shapes of the inner segmented DC electrodes.
    """
    rf = junction_geometry.RF
    dc = junction_geometry.DC
    c = junction_geometry.C
    
    X_bottom, Y_bottom = control_points
    step = 2 if trench_width !=0 else 1
    n_C = len(c.heights)

    shapes = []
        
    # Linear interpolation between two consecutive control points
    def interpolate_line(x, i):
        x1, y1 = X_bottom[i], Y_bottom[i]
        x2, y2 = X_bottom[i-1], Y_bottom[i-1]

        if x1 == x2:
            return y1

        m = (y2 - y1) / (x2 - x1)
        return m * (x - x1) + y1

    # Extend control points for symmetry and trench processing
    X_bottom = np.concatenate((X_bottom, [-Y_bottom[-1]]))
    Y_bottom = np.concatenate(([c.width/2 + trench_width], Y_bottom))

    if trench_width != 0:
        # Add mirrored control points for trench geometry
        X_bottom = np.concatenate((X_bottom, -np.array([Y_bottom[-2]])))
        Y_bottom = np.concatenate((Y_bottom, -np.array([X_bottom[-3]])))
        
        # Compute intersection points for trench offset
        int_points = lines_intersections([X_bottom, Y_bottom], -trench_width, angle=True)
        X_bottom = np.array(int_points)[:, 0]
        Y_bottom = np.array(int_points)[:, 1]

        # Insert trench widths between each conductor segment
        C_heights_new = []
        for h in c.heights:
            C_heights_new.append(h)
            C_heights_new.append(trench_width)
        c.heights = C_heights_new[:-1]

    i = - 1
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
                y_interp = interpolate_line(target_x, i + 1)
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


def pw_segmented_dc_shapes(junction_geometry: JunctionGeometry,
                           control_points: list[np.ndarray],
                           trench_width: float) -> list[list[tuple[float, float]]]:
    """
    Builds the outer segmented DC control electrodes of the junction (linear trap style).
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
        Shapes of the outer segmented DC electrodes.
    """
    rf = junction_geometry.RF
    dc = junction_geometry.DC
    c = junction_geometry.C
    
    X_top, Y_top = control_points
    step = 2 if trench_width !=0 else 1
    
    shapes = []
    
    # Linear interpolation between two consecutive control points
    def interpolate_line(x, i):
        x1, y1 = X_top[i], Y_top[i]
        x2, y2 = X_top[i-1], Y_top[i-1]

        if x1 == x2:
            return y1

        m = (y2 - y1) / (x2 - x1)
        return m * (x - x1) + y1
    
    # Extend control points for symmetry and trench processing
    X_top = np.concatenate((X_top, [-Y_top[-1]]))
    Y_top = np.concatenate(([c.width/2 + rf.width + trench_width], Y_top))

    # Start from the last point, to go backward and segment the bottom part    
    if trench_width != 0:
        # Add mirrored control points for trench geometry
        X_top = np.concatenate((X_top, -np.array([Y_top[-2]])))
        Y_top = np.concatenate((Y_top, -np.array([X_top[-2]])))
        
        # Compute intersection points for trench offset
        int_points = lines_intersections([X_top, Y_top], trench_width, angle=False)
        
        # Define shifted points on the y=-x axis
        point1 = (-Y_top[-1], Y_top[-1])
        point2 = (-c.width/2 - rf.width - dc.height - trench_width, c.width/2 + rf.width + dc.height + trench_width)
        
        # Compute lines params (shifted and un-shifted)
        m2, q2 = shifted_line_params(point1, point2, -trench_width/2)
        m1, q1 = shifted_line_params(int_points[-2], int_points[-1], 0)
        
        # Compute intersection for smooth electrode termination
        intersection = single_intersection([m1, q1], [m2, q2])
        
        # Replace last point with new intersections
        int_points = int_points[:-1]
        int_points.extend([
            intersection, 
            ((point2[1] - q2) / m2, point2[1])
        ])
                
        # Compute offset entry point for trench
        m, q = shifted_line_params((X_top[0], Y_top[0]), (X_top[1], Y_top[1]), trench_width)

        # Truncate X_top/Y_top to truncated trench segment before full trench arc
        X_top = np.concatenate(([(Y_top[0] + trench_width)/m - q/m], np.array(int_points)[:-1, 0]))
        Y_top = np.concatenate(([Y_top[0]], np.array(int_points)[:-1, 1]))
    else:
        # Basic fallback point if no trench is used
        int_points = [(-c.width/2 - rf.width - dc.height, c.width/2 + rf.width + dc.height)]
    
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
                y_interp = interpolate_line(target_x, i + 1)
            else:
                y_interp = rf.width + c.width/2 + trench_width

            bottom_segments.append((target_x, y_interp))

    # Reverse to maintain the correct order
    bottom_segments = bottom_segments[::-1]
    
    # === Build first shape (trapezoidal) ===
    shape = [int_points[-1]]
    shape.extend([(point[0], int_points[-1][1]) for point in bottom_segments[::-1] if point[0] >= -DC_widths[0] + bottom_segments[-1][0] and point[0] <= int_points[-1][0]])
    shape.extend([point for point in bottom_segments if point[0] >= -DC_widths[0] + bottom_segments[-1][0]])
    
    # Remove repeated points and append the final shape
    new_shape = remove_repeated_vertices(shape, tolerance=1e-6)
    shapes.append(new_shape)

    # === Initialize first_point for next segments === 
    for i in range(len(shape)):
        if shape[i][1] != shape[0][1]:
            break
    
    # Shift trench_width left to start next DC shape
    first_point = (shape[i][0] - trench_width, shape[0][1])

    # === Build remaining segmented shapes ===
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
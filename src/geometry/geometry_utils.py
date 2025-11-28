#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File: geometry_utils.py
# Created: 12-04-2025
# Author: Lorenzo Calandra Buonaura <lorenzocb01@gmail.com>
# Institution: University of Innsbruck - UIBK
#
# This file contains some geometrical functions used throughout the code: remove repeated
# vertices, find the intersections when shifting the control points, CCW checks, ...
#


########################
# IMPORT ZONE          #
########################

import numpy as np

from electrode import System

########################
# FUNCTIONS            #
########################

def remove_repeated_vertices(shape: list[tuple[float, float]], tolerance: float) -> list[tuple[float, float]]:
    """
    Remove repeated vertices in a shape.

    Parameters
    ----------
    shape : list[tuple[float, float]]
        Electrode shape.
    tolerance : float
        Tolerance threshold for considering two points equal.

    Returns
    -------
    list[tuple[float, float]]
        Shape without repeated vertices.
    """
    new_shape = []
    shape = [item for item in shape if item is not None]

    for item in shape:
        x, y = item
        is_duplicate = any(np.isclose(x, x2, atol=tolerance) and np.isclose(y, y2, atol=tolerance) for x2, y2 in new_shape)

        if not is_duplicate:
            new_shape.append(item)
            
    return new_shape


def shifted_lines_for_consecutive_points(coordinates: list[np.ndarray], distance: float, angle: bool) -> list[tuple[float, float]]:
    """
    Finds shifted line parameters for a set of consecutive points.

    Parameters
    ----------
    coordinates : list[np.ndarray]
        Control points coordinates.
    distance : float
        Distance between the original lines and the shifted ones.
    angle : bool
        If true, it considers the last point as 'back-propagating' and changes
        the sign of the distance.

    Returns
    -------
    list[tuple[float, float]]
        Shifted line parameters (m, q).
    """
    X, Y = coordinates
    
    # Ensure the lengths of X and Y coordinates match
    if len(X) != len(Y):
        raise ValueError("The X and Y coordinate lists must have the same length.")
    
    shifted_line_params_list = []
    length = len(X) - 1 if not angle else len(X) - 2
    
    for i in range(length):
        point1 = (X[i], Y[i])
        point2 = (X[i + 1], Y[i + 1])
        
        # Calculate the shifted line parameters (m, q)
        m_shifted, q_shifted = shifted_line_params(point1, point2, distance)
        shifted_line_params_list.append((m_shifted, q_shifted))
        
    if angle:
        point1 = (X[length], Y[length])
        point2 = (X[length+1], Y[length+1])
        
        # Calculate the shifted line parameters (m, q)
        m_shifted, q_shifted = shifted_line_params(point1, point2, -distance)
        shifted_line_params_list.append((m_shifted, q_shifted))
    
    return shifted_line_params_list


def lines_intersections(coordinates: list[np.ndarray], distance: float, angle: bool) -> tuple[list[float], list[float]]:
    """
    Finds the intersection between the shifted lines.

    Parameters
    ----------
    coordinates : list[np.ndarray]
        Control points coordinates.
    distance : float
        Distance between the original lines and the shifted ones.
    angle : bool
        If true, it considers the last point as 'back-propagating' and changes
        the sign of the distance.

    Returns
    -------
    tuple[list[float], list[float]]
        Coordinates of the new intersection points (X_val, Y_val).
    """
    X, Y = coordinates

    # Ensure the lengths of X and Y coordinates match
    if len(X) != len(Y):
        raise ValueError("The X and Y coordinate lists must have the same length.")
    
    # Compute shifted line parameters for consecutive points
    shifted_line_params_list = shifted_lines_for_consecutive_points([X, Y], distance, angle)
    
    intersection_points = []
    for i in range(len(shifted_line_params_list) - 1):
        m1, q1 = shifted_line_params_list[i]
        m2, q2 = shifted_line_params_list[i + 1]
        
        try:
            # Find intersection between shifted lines
            x_intersection, y_intersection = single_intersection([m1, q1], [m2, q2])
            intersection_points.append((x_intersection, y_intersection))
            
        except ValueError:
            intersection_points.append(None)
    
    return intersection_points


def single_intersection(params1: list[float], params2: list[float]) -> tuple[float, float]:
    """
    Finds the intersection between two lines.

    Parameters
    ----------
    params1 : list[float]
        (m, q) of the first line.
    params2 : list[float]
        (m, q) of the second line.

    Returns
    -------
    tuple[float, float]
        Coordinates of the intersection point.
    """
    m1, q1 = params1
    m2, q2 = params2
    
    if m1 == m2:
        raise ValueError("The lines are parallel and do not intersect.")
    
    x_intersection = (q2 - q1) / (m1 - m2)
    y_intersection = m1 * x_intersection + q1
    
    return x_intersection, y_intersection


def line_params(point1: tuple[float, float], point2: tuple[float, float]) -> tuple[float, float]:
    """
    Finds the parameters (m, q) of a line given two points.

    Parameters
    ----------
    point1 : tuple[float, float]
        Coordinates of the first point.
    point2 : tuple[float, float]
        Coordinates of the second point.

    Returns
    -------
    tuple[float, float]
        Parameters of the line (m, q).
    """
    x1, y1 = point1
    x2, y2 = point2
    
    if y1 == y2:
        return 0, y1
    
    m = (y2 - y1) / (x2 - x1)
    q = y1 - m * x1
    
    return m, q


def shifted_line_params(point1: tuple[float, float], point2: tuple[float, float], distance: float) -> tuple[float, float]:
    """
    Finds the parameters (m, q) of a line parallel to the one passing between
    two points given a specific distance.

    Parameters
    ----------
    point1 : tuple[float, float]
        Coordinates of the first point.
    point2 : tuple[float, float]
        Coordinates of the second point.
    distance : float
        Distance between the lines.

    Returns
    -------
    tuple[float, float]
        Parameters of the shifted line (m, q).
    """
    x1, y1 = point1
    x2, y2 = point2
    
    m, q = line_params(point1, point2)
    
    # Calculate the perpendicular distance and shift the points along it
    normal_vector = np.array([-m, 1]) / np.sqrt(1 + m**2)
    shift_vector = normal_vector * distance
    
    # New points after shifting
    new_point1 = (x1 + shift_vector[0], y1 + shift_vector[1])
    new_point2 = (x2 + shift_vector[0], y2 + shift_vector[1])
    
    m_new, q_new = line_params(new_point1, new_point2)
    
    return m_new, q_new


def is_ccw(path: list[np.ndarray]) -> bool:
    """
    Check if a path is counterclockwise using the shoelace theorem.
    
    (https://www.101computing.net/the-shoelace-algorithm)

    Parameters
    ----------
    path : list[np.ndarray]
        Electrode path.

    Returns
    -------
    bool
        Returns True if the vertices of the electrode are CCW, False otherwise.
    """
    area = 0
    for i in range(len(path)):
        x1, y1 = path[i]
        x2, y2 = path[(i + 1) % len(path)]
        area += (x1 * y2 - x2 * y1)
        
    return area > 0


def check_all_ccw(s: System):
    """
    Check if all the electrodes in the system are CCW, otherwise prints the names
    of the electrode that are CW (and that are considered to have a "negative 
    contribution" by the 'electrode' package).

    Parameters
    ----------
    s : System
        Electrode system
    """
    all_ccw = True
    for electrode in s:
        for path in electrode.paths:
            ccw = is_ccw(path)

            if not ccw:
                print(f"Electrode {electrode.name} is CW.")
                all_ccw = False

    if all_ccw:
        print("All electrodes are CCW.")
    else:
        print("Fix the CW electrodes.")


def shift_spline(spline: tuple[list[float], list[float]], distance: float) -> tuple[list[float], list[float]]:
    """
    Shifts a spline of a specific distance.

    Parameters
    ----------
    spline : tuple[list[float], list[float]]
        Coordinates of the spline.
    distance : float
        Distance between the original spline and the shifted one.

    Returns
    -------
    tuple[list[float], list[float]]
        Coordinates of the shifted spline
    """
    X_spline, Y_spline = spline

    # Compute derivatives (tangent vectors)
    dx = np.gradient(X_spline)
    dy = np.gradient(Y_spline)

    # Normalize the tangent vectors
    lengths = np.sqrt(dx**2 + dy**2)
    tangent_x = dx / lengths
    tangent_y = dy / lengths

    # Compute normals (perpendicular to tangent)
    normal_x = -tangent_y
    normal_y = tangent_x

    # Offset by gap along normal
    X_offset = X_spline + distance * normal_x
    Y_offset = Y_spline + distance * normal_y
    
    return X_offset, Y_offset


def arc(radius, trench_width=0):
    # Compute angular cutoff based on arc length = radius * angle
    delta_theta = (trench_width / 2) / radius if trench_width > 0 else 0.0

    theta_min = delta_theta
    theta_max = np.pi / 2 - delta_theta
    theta = np.linspace(theta_min, theta_max, 201)

    return [(radius * np.cos(t), radius * np.sin(t)) for t in theta]



def splitted_arc(radius, trench_width=0, n=2):
    total_angle = np.pi / 2  # 90 degrees
    angle_per_slice = total_angle / n

    # Convert trench width to angular margin
    delta_theta = (trench_width / 2) / radius if trench_width > 0 else 0.0
    arcs = []

    for i in range(n):
        theta_min = i * angle_per_slice + delta_theta
        theta_max = (i + 1) * angle_per_slice - delta_theta
        if theta_max <= theta_min:
            arcs.append([])  # Avoid invalid segments
            continue
        theta = np.linspace(theta_min, theta_max, 201//n)
        arcs.append([(radius * np.cos(t), radius * np.sin(t)) for t in theta])

    return arcs
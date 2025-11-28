#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File: states.py
# Created: 22-07-2025
# Author: Lorenzo Calandra Buonaura <lorenzocb01@gmail.com>
# Institution: University of Innsbruck - UIBK
#
# Classes for designing routing paths and managing states for TS-VIAs.
#


########################
# IMPORT ZONE          #
########################

import gdspy
import random
import numpy as np

from shapely import unary_union
from shapely.geometry import Point, Polygon, LineString


########################
# CLASSES              #
########################

class RoutingPath:
    def __init__(self, dc, connection_point, layer, slide=0.5):
        self.dc = dc
        self.connection_point = connection_point
        self.layer = layer
        self.path = self.generate_path(slide=slide)

    def generate_path(self, wire_width=10, bend_radius=6, slide=0.5):
        path = gdspy.FlexPath([(self.dc.x, self.dc.y)], wire_width, corners="circular bend", bend_radius=bend_radius, layer=self.layer)

        # Calculate sliding middle points
        mid_x = self.dc.x + slide * (self.connection_point.x - self.dc.x)
        mid_y = self.dc.y + slide * (self.connection_point.y - self.dc.y)

        # First segment: DC → (mid_x, dc.y)
        path.segment((mid_x, self.dc.y))

        # Second segment: (mid_x, dc.y) → (mid_x, mid_y)
        path.segment((mid_x, mid_y))

        # Third segment: (mid_x, mid_y) → connection_point
        path.segment((self.connection_point.x, self.connection_point.y))

        return path

    def get_buffer_zone(self, buffer_radius=7):
        polys = self.path.get_polygons()
        return unary_union([Polygon(p).buffer(buffer_radius) for p in polys])

class RoutingState:
    def __init__(self, pin_zone_placer, electrodes, layer_options, margin=25, initialize=True):
        self.pin_zone_placer = pin_zone_placer
        self.all_dc_points = self.pin_zone_placer.get_pin_centroids()
        self.pin_map = self.pin_zone_placer.get_pin_buffer_map()
        self.all_electrodes = electrodes
        self.layer_options = layer_options
        self.valid_points = {}

        self.assignments = {}          # dc -> (electrode, connection_point, layer)
        self.paths = {}                # dc -> RoutingPath
        self.fixed_assignments = {}    # pins that should not be perturbed

        # Keep track of which DCs/electrodes are still available
        self.remaining_dc_points = list(self.all_dc_points)
        self.remaining_electrodes = list(self.all_electrodes)

        self._assign_fixed_connections(margin)
        
        if initialize:
            self._initialize_greedy()

    def _assign_fixed_connections(self, margin):
        for dc in self.all_dc_points:
            dc_point = Point(dc)
            for el in self.remaining_electrodes:
                expanded_poly = el.polygon.buffer(-margin)
                if expanded_poly.contains(dc_point):
                    layer = self.layer_options[0]  # fixed layer (can change if needed)
                    self.assignments[dc] = (el, dc, layer)
                    self.fixed_assignments[dc] = (el, dc, layer)

                    self.remaining_dc_points.remove(dc)
                    self.remaining_electrodes.remove(el)
                    break  # move to next DC
    
    def get_dc_from_electrode(self, target_electrode):
        for dc, (el, _, _) in self.assignments.items():
            if el == target_electrode:
                return dc

    def _distance(self, a, b):
        """Compute Euclidean distance between two points a and b."""
        return ((a.x - b.x)**2 + (a.y - b.y)**2) ** 0.5

    def _initialize_greedy(self):
        """Greedily assign each DC to the closest electrode (minimizing path length)."""

        num_connections = min(len(self.remaining_dc_points), len(self.remaining_electrodes))
        dcs = self.remaining_dc_points[:]
        els = self.remaining_electrodes[:]

        # Precompute valid connection points for each electrode
        for el in els:
            self.valid_points[el] = el.valid_connection_points()

        # Create all candidate (dc, el, point, layer, cost) tuples
        candidates = []
        for dc in dcs:
            for el in els:
                for point in self.valid_points[el]:
                    layer = random.choice(self.layer_options)
                    path = RoutingPath(dc, point, layer)
                    coords = np.array(path.path.points)
                    line = LineString(coords)
                    dist = line.length
                    candidates.append((dist, dc, el, point, layer))

        # Sort all candidates by distance (greedy selection)
        candidates.sort()

        assigned_dcs = set()
        assigned_els = set()

        for dist, dc, el, point, layer in candidates:
            if dc in assigned_dcs or el in assigned_els:
                continue
            self.assignments[dc] = (el, point, layer)
            self.paths[dc] = RoutingPath(dc, point, layer)
            assigned_dcs.add(dc)
            assigned_els.add(el)
            if len(assigned_dcs) >= num_connections:
                break
            
        print(f"Finished greedy initialization, current cost:{self.compute_cost()}")


    def clone(self):
        new = RoutingState(self.pin_zone_placer, self.all_electrodes[:], self.layer_options[:], initialize=False)
        new.assignments = self.assignments.copy()
        new.paths = self.paths.copy()
        new.fixed_assignments = self.fixed_assignments.copy()
        new.remaining_dc_points = self.remaining_dc_points[:]
        new.remaining_electrodes = self.remaining_electrodes[:]
        new.valid_points = {el: points[:] for el, points in self.valid_points.items()}
        return new

    def perturb(self, dc_cost_map=None):
        if not self.remaining_dc_points or not self.remaining_electrodes:
            return

        used_electrodes = [v[0] for k, v in self.assignments.items() if k not in self.fixed_assignments]
        if not used_electrodes:
            return

        current_dcs = [dc for dc in self.assignments if dc not in self.fixed_assignments]
        available_dcs = list(set(self.remaining_dc_points) - set(current_dcs))
        if not available_dcs:
            return

        # Decide: 80% swap two worsts, 20% do random reassignment
        if random.random() < 0.8:
            # Get top 2 worst DCs
            sorted_dcs = sorted(
                [(dc, cost) for dc, cost in dc_cost_map.items() if dc in current_dcs],
                key=lambda x: x[1],
                reverse=True
                )

            dc1, dc2 = sorted_dcs[0][0], sorted_dcs[1][0]
            el1, point1, layer1 = self.assignments[dc1]
            el2, point2, layer2 = self.assignments[dc2]

            # Swap electrodes
            self.assignments[dc1] = (el2, point2, layer2)
            self.paths[dc1] = RoutingPath(dc1, self.assignments[dc1][1], self.assignments[dc1][2])

            self.assignments[dc2] = (el1, point1, layer1)
            self.paths[dc2] = RoutingPath(dc2, self.assignments[dc2][1], self.assignments[dc2][2])

        else:
            # Random reassignment
            dc_to_replace = random.choice(current_dcs)
            el = self.assignments[dc_to_replace][0]
            new_dc = random.choice(available_dcs)
            point = random.choice(self.valid_points[el])
            layer = random.choice(self.layer_options)
            new_path = RoutingPath(new_dc, point, layer)

            del self.assignments[dc_to_replace]
            del self.paths[dc_to_replace]
            self.assignments[new_dc] = (el, point, layer)
            self.paths[new_dc] = new_path

            
    def compute_cost(self, collision_penalty=1e5, 
                    wire_buffer_radius=7,
                    layer_weights={'3': 0, '4': 1000},
                    return_dc_costs=False):

        total_length = 0
        wire_buffers = []
        dc_cost_map = {}

        # Build wire buffer zones and accumulate wire lengths
        for dc, path in self.paths.items():
            buf = path.get_buffer_zone(buffer_radius=wire_buffer_radius)
            wire_buffers.append(buf)

            raw_points = path.path.points
            coords = np.array(raw_points) if not isinstance(raw_points, np.ndarray) else raw_points

            if coords.ndim == 2 and coords.shape[0] >= 2:
                line = LineString(coords)
                length = line.length
                total_length += length
                dc_cost_map[dc] = length  # start with length

        # --- Collision checks ---
        wire_dc_pairs = list(self.paths.items())
        num_paths = len(wire_dc_pairs)
        penalty = 0

        # 1. Wire-Wire intersections with same layer only
        for i in range(num_paths):
            dc_i, path_i = wire_dc_pairs[i]
            layer_i = path_i.layer  # Assuming 'path' object has .layer attribute

            for j in range(i + 1, num_paths):
                dc_j, path_j = wire_dc_pairs[j]
                layer_j = path_j.layer

                if layer_i != layer_j:
                    # Different layers — no collision penalty
                    continue

                if wire_buffers[i].intersects(wire_buffers[j]):
                    penalty += collision_penalty
                    dc_cost_map[dc_i] = dc_cost_map.get(dc_i, 0) + collision_penalty
                    dc_cost_map[dc_j] = dc_cost_map.get(dc_j, 0) + collision_penalty

        # 2. Wire-Pin intersections (exclude own pin)
        for dc, wire in zip(list(self.assignments.keys())[-len(self.remaining_electrodes):], wire_buffers):
            key = Point(round(dc.x, 6), round(dc.y, 6))
            for other_key, other_buffer in self.pin_map.items():
                if other_key == key:
                    continue
                if not wire.intersection(other_buffer).is_empty:
                    penalty += 100*collision_penalty
                    dc_cost_map[dc] = dc_cost_map.get(dc, 0) + 100*collision_penalty
                    
        # 3. Layer cost penalty
        for dc, path in self.paths.items():
            layer_penalty = layer_weights.get(str(path.layer), 0)
            penalty += layer_penalty
            dc_cost_map[dc] = dc_cost_map.get(dc, 0) + layer_penalty

        total_cost = total_length + penalty

        if return_dc_costs:
            return total_cost, dc_cost_map
        else:
            return total_cost
        


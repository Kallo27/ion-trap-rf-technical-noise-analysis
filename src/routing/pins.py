#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File: pins.py
# Created: 22-07-2025
# Author: Lorenzo Calandra Buonaura <lorenzocb01@gmail.com>
# Institution: University of Innsbruck - UIBK
#
# Classes for designing pin zones for TS-VIAs.
#


########################
# IMPORT ZONE          #
########################

import gdspy
import numpy as np

from shapely.geometry import Point, Polygon


########################
# CLASSES              #
########################

class Pin:
    def __init__(self, center=(0, 0), radius=15, layer=10, num_points=64):
        self.center = center
        self.radius = radius
        self.layer = layer
        self.num_points = num_points
        self.cell_name = f"CIRCULAR_PIN_{id(self)}"
        self._cell = self._create_cell()

    def _create_cell(self):
        if self.cell_name in gdspy.current_library.cells:
            del gdspy.current_library.cells[self.cell_name]
        cell = gdspy.Cell(self.cell_name)
        circle = gdspy.Round(self.center, self.radius, number_of_points=self.num_points, layer=self.layer)
        cell.add(circle)
        return cell

    def get_cell(self):
        return self._cell

class PinZone:
    def __init__(self, name="PIN_ZONE", pin=None, rows=2, cols=2, spacing=30, staggered=True):
        self.name = name
        self.pin = pin or Pin()
        self.rows = rows
        self.cols = cols
        self.spacing = spacing
        self.staggered = staggered
        self._cell = self._create_zone()

    def _create_zone(self):
        if self.name in gdspy.current_library.cells:
            del gdspy.current_library.cells[self.name]
        cell = gdspy.Cell(self.name)

        for i in range(self.rows):
            for j in range(self.cols):
                offset = (self.spacing / 2) if self.staggered and i % 2 == 1 else 0
                x = i * self.spacing
                y = j * self.spacing + offset
                ref = gdspy.CellReference(self.pin.get_cell(), (x, y))
                cell.add(ref)
        return cell

    def get_cell(self):
        return self._cell

class CompositePinZone:
    def __init__(self, name="COMPOSITE_PIN_ZONE", sub_rows=2, sub_cols=2,
                 spacing=100, radius=15, layer=10):
        self.name = name
        self.sub_rows = sub_rows
        self.sub_cols = sub_cols
        self.spacing = spacing
        self.radius = radius
        self.layer = layer
        self.subzone_spacing = (np.sqrt(2) + 1) * 100
        self.pin = Pin(radius=self.radius, layer=self.layer)
        self._cell = self._build_composite_zone()

    def _build_subzone(self):
        name = "ROTATED_SUBZONE"
        if name in gdspy.current_library.cells:
            del gdspy.current_library.cells[name]
        cell = gdspy.Cell(name)
        dx = (self.sub_rows - 1) * self.spacing / 2
        dy = (self.sub_cols - 1) * self.spacing / 2
        for i in range(self.sub_rows):
            for j in range(self.sub_cols):
                x = i * self.spacing - dx
                y = j * self.spacing - dy
                ref = gdspy.CellReference(self.pin.get_cell(), (x, y))
                cell.add(ref)
        return cell

    def _build_composite_zone(self):
        if self.name in gdspy.current_library.cells:
            del gdspy.current_library.cells[self.name]

        composite = gdspy.Cell(self.name)
        subzone = self._build_subzone()

        offsets = [
            (-self.subzone_spacing / 2, -self.subzone_spacing / 2),
            ( self.subzone_spacing / 2, -self.subzone_spacing / 2),
            (-self.subzone_spacing / 2,  self.subzone_spacing / 2),
            ( self.subzone_spacing / 2,  self.subzone_spacing / 2),
        ]
        for ox, oy in offsets:
            composite.add(gdspy.CellReference(subzone, (ox, oy), rotation=45))

        # Add central pin
        composite.add(gdspy.CellReference(self.pin.get_cell(), (0, 0)))

        return composite

    def get_cell(self):
        return self._cell


class PinZonePlacer:
    def __init__(self, square_size=1500, pin_spacing=100, pin_radius=15, offset=35):
        self.square_size = square_size
        self.pin_spacing = pin_spacing
        self.pin_radius = pin_radius
        self.offset = offset
        self.cell_name = "PIN_ZONES_IN_CORNERS"

        if self.cell_name in gdspy.current_library.cells:
            del gdspy.current_library.cells[self.cell_name]

        self.cell = gdspy.Cell(self.cell_name)
        self._place_zones()

    def _place_zones(self):
        pin_zone = CompositePinZone(spacing=self.pin_spacing, radius=self.pin_radius).get_cell()
        zone_size = (
            (1 + 2 * np.sqrt(2)) * self.pin_spacing + 2 * self.pin_radius,
            (1 + 2 * np.sqrt(2)) * self.pin_spacing + 2 * self.pin_radius
        )
        shift = self.square_size / 2

        corners = [
            ((-shift + self.offset + zone_size[0] / 2, -shift + self.offset + zone_size[1] / 2), 0),
            (( shift - self.offset - zone_size[0] / 2, -shift + self.offset + zone_size[1] / 2), 90),
            ((-shift + self.offset + zone_size[0] / 2,  shift - self.offset - zone_size[1] / 2), -90),
            (( shift - self.offset - zone_size[0] / 2,  shift - self.offset - zone_size[1] / 2), 180)
        ]

        for origin, angle in corners:
            self.cell.add(gdspy.CellReference(pin_zone, origin, rotation=angle))

        rect = gdspy.Rectangle((-shift, -shift), (shift, shift), layer=2)
        self.cell.add(rect)

    def get_cell(self):
        return self.cell

    def get_pin_centroids(self):
        centroids = []
        for poly in self.get_pin_polygons():
            centroids.append(poly.centroid)
        return centroids
    
    def get_pin_polygons(self):
        polygons = []
        # By default, get all polygons on all layers
        # If you want to filter by layer, use get_polygons(by_spec=True)
        for poly in self.cell.get_polygons():
            if isinstance(poly, (list, np.ndarray)) and len(poly) > 2:
                polygons.append(Polygon(poly))
        return polygons[1:]
    
    def get_pin_buffer_map(self, buffer_radius=7):
        """
        Returns a dict mapping (rounded_x, rounded_y) -> buffered Polygon
        """
        buffer_map = {}
        for poly in self.get_pin_polygons():
            centroid = poly.centroid
            key = Point(round(centroid.x, 6), round(centroid.y, 6))
            buffer_map[key] = Polygon(poly).buffer(buffer_radius)
        return buffer_map
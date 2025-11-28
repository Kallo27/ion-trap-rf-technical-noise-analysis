#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File: electrodes.py
# Created: 22-07-2025
# Author: Lorenzo Calandra Buonaura <lorenzocb01@gmail.com>
# Institution: University of Innsbruck - UIBK
#
# Class for DC electrodes for wire routing.
#


########################
# IMPORT ZONE          #
########################

import random

from shapely.geometry import Point


########################
# CLASSES              #
########################

class DCElectrode:
    def __init__(self, polygon, id=None):
        self.polygon = polygon
        self.id = id
    
    def valid_connection_points(self, margin=5.0, num_points=100):
        interior = self.polygon.buffer(-margin)
        if interior.is_empty:
            return []
        # Sample points along the interior boundary or inside
        minx, miny, maxx, maxy = interior.bounds
        points = []
        while len(points) < num_points:
            x = random.uniform(minx, maxx)
            y = random.uniform(miny, maxy)
            p = Point(x, y)
            if interior.contains(p):
                points.append(p)
        return points
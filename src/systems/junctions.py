#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File: junctions.py
# Created: 10-04-2025
# Author: Lorenzo Calandra Buonaura <lorenzocb01@gmail.com>
# Institution: University of Innsbruck - UIBK
#
# This file defines the classes representing an X-junction ion trap geometry.
# Two implementations are provided:
# - A piecewise junction, constructed from straight segments with fixed transitions
# - A spline-based junction, using smooth curves for continuous electrode paths
#
# Both classes inherit from a common abstract base class that defines the interface
# and shared functionality for junction construction and manipulation.
#
# These classes are used to build and analyze junction geometries in surface-electrode
# ion trap simulations.
#


########################
# IMPORT ZONE          #
########################

import warnings
import numpy as np
from abc import ABC, abstractmethod
from shapely.geometry import Polygon

from electrode import (System, PolygonPixelElectrode)

from src.shapes.junction_shapes import (rf_shapes,
                                        central_shape,
                                        dc_shapes,
                                        segmented_central_shapes,
                                        segmented_dc_shapes)

from src.shapes.piecewise_junction_shapes import (pw_rf_shapes,
                                                  pw_central_shape,
                                                  pw_dc_shapes,
                                                  pw_segmented_central_shapes,
                                                  pw_segmented_dc_shapes)

from src.shapes.spline_junction_shapes import (spline_rf_shapes,
                                               spline_central_shape,
                                               spline_dc_shapes,
                                               spline_segmented_central_shapes,
                                               spline_segmented_dc_shapes)

from src.geometry.modules import JunctionGeometry, ControlPoints


########################
# CLASSES              #
########################

class BaseJunction(ABC):
    def __init__(self, junction_geometry, trench_width=0, flags=None):
        self.jm = junction_geometry
        self.rf = junction_geometry.RF
        self.dc = junction_geometry.DC
        self.c = junction_geometry.C
        self.control_points = junction_geometry.points
        self.trench_width = trench_width

        default_flags = {
            "build_RF": True,
            "build_DC": True,
            "build_C": True,
            "segment_DC": False,
            "segment_C": False
        }

        if flags is None:
            flags = default_flags
        else:
            default_flags.update(flags)
            flags = default_flags

        self.flags = flags
        self._validate_flags()

        self.default_voltages = {
            "RF": {"attr": "rf", "value": 2},
            "C":  {"attr": "dc", "value": -2.0},
            "DC": {"attr": "dc", "value": -1}
        }

        self.last_voltages = None


    def _validate_flags(self) -> None:
        if self.flags["build_RF"] and self.rf is None:
            raise ValueError("Flag 'build_RF' is True, but RF geometry is missing.")
        if not self.flags["build_RF"] and self.rf is not None:
            warnings.warn("Flag 'build_RF' is False, but RF geometry is provided.")
            
        if self.flags["build_DC"] and self.dc is None:
            raise ValueError("Flag 'build_DC' is True, but DC geometry is missing.")
        if not self.flags["build_DC"] and self.dc is not None:
            warnings.warn("Flag 'build_DC' is False, but DC geometry is provided.")

        if self.flags["build_C"] and self.c is None:
            raise ValueError("Flag 'build_C' is True, but C geometry is missing.")
        if not self.flags["build_C"] and self.c is not None:
            warnings.warn("Flag 'build_C' is False, but C geometry is provided.")


    @abstractmethod
    def _generate_grids(self):
        pass
    
    @abstractmethod
    def _build_rf_electrodes(self):
        pass
    
    @abstractmethod
    def _build_central_electrodes(self):
        pass
    
    @abstractmethod
    def _build_dc_electrodes(self):
        pass
    

    def assign_voltages(self, system, electrodes_dict, voltages=None):
        pot = self.default_voltages.copy()
        if voltages:
            for key, val in voltages.items():
                if isinstance(val, dict):
                    pot[key] = val
                else:
                    pot[key] = {"attr": self.default_voltages[key]["attr"], "value": val}
  
        for key, electrodes in electrodes_dict.items():
            if key not in pot:
                continue
            attr = pot[key]["attr"]
            value = pot[key]["value"]
            for name, _ in electrodes:
                setattr(system[name], attr, value)
  
    def build(self, voltages=None):
        self._generate_grids()
        RF_electrodes = self._build_rf_electrodes()
        C_electrodes = self._build_central_electrodes()
        DC_electrodes = self._build_dc_electrodes()
    
        all_electrodes = RF_electrodes + C_electrodes + DC_electrodes
        system = System([PolygonPixelElectrode(name=name, paths=map(np.array, paths)) for name, paths in all_electrodes])
  
        electrodes_by_type = {
            "RF": RF_electrodes,
            "C": C_electrodes,
            "DC": DC_electrodes
        }
  
        volt = self.default_voltages.copy()
        if voltages:
            volt.update({k: {"attr": self.default_voltages[k]["attr"], "value": v} for k, v in voltages.items()})
        self.last_voltages = volt
        self.assign_voltages(system, electrodes_by_type, voltages)
  
        return system
  
    def update_control_points(self, new_control_points):
        if not isinstance(new_control_points, ControlPoints):
            raise ValueError("The provided control points must be an instance of ControlPoints.")
        self.jm.points = new_control_points
        self.control_points = new_control_points
  
        updated_system = self.build(voltages={k: v["value"] for k, v in self.last_voltages.items()}
                                    if self.last_voltages else None)
        return updated_system



class NonOptimizedJunction(BaseJunction):
    def _generate_grids(self) -> None:
        pass

    def _build_central_electrodes(self) -> list[tuple[str, list[tuple[float, float]]]]:
        if not self.flags["build_C"]:
            return []

        electrodes = []
        c_index = 1

        if not self.flags["segment_C"]:
            shape = central_shape(junction_geometry=self.jm, 
                                  trench_width=self.trench_width)
            
            electrodes.append(("C", [shape]))
            
        else:
            shapes = segmented_central_shapes(junction_geometry=self.jm,
                                              trench_width=self.trench_width)
            
            electrodes.append(("C", [shapes[0]]))
        
            for shape in shapes[1:]:
                electrodes.extend([
                    (f"L{c_index}", [shape]),
                    (f"D{c_index}", [[(-y, x) for x, y in shape]]),
                    (f"R{c_index}", [[(-x, y) for x, y in reversed(shape)]]),
                    (f"U{c_index}", [[(-y, -x) for x, y in reversed(shape)]]),
                ])
                
                c_index += 1
        
        return electrodes


    def _build_dc_electrodes(self) -> list[tuple[str, list[tuple[float, float]]]]:
        if not self.flags["build_DC"]:
            return []

        electrodes = []
        dc_index = 1

        if not self.flags["segment_DC"]:
            shapes = dc_shapes(junction_geometry=self.jm,
                               trench_width=self.trench_width,
                               single=True)
            
        else:
            shapes = segmented_dc_shapes(junction_geometry=self.jm,
                                         trench_width=self.trench_width)

        for shape in shapes:
            electrodes.extend([
                (f"DC{dc_index}", [shape]),
                # (f"DC{dc_index+1}", [[(-y, -x) for x, y in reversed(shape)]]),
                (f"DC{dc_index+2}", [[(y, -x) for x, y in shape]]),
                # (f"DC{dc_index+3}", [[(-x, y) for x, y in reversed(shape)]]),
                (f"DC{dc_index+4}", [[(-x, -y) for x, y in shape]]),
                # (f"DC{dc_index+5}", [[(y, x) for x, y in reversed(shape)]]),
                (f"DC{dc_index+6}", [[(-y, x) for x, y in shape]]),
                # (f"DC{dc_index+7}", [[(x, -y) for x, y in reversed(shape)]]),
            ])
            
            # dc_index += 8
            dc_index += 4

        return electrodes
 
    
    def _build_rf_electrodes(self) -> list[tuple[str, list[tuple[float, float]]]]:
        if not self.flags["build_RF"]:
            return []

        shape = rf_shapes(junction_geometry=self.jm)

        electrodes = [
            ("RF_1", [shape]),
            ("RF_2", [[(x, -y) for x, y in reversed(shape)]]),
            ("RF_3", [[(-x, y) for x, y in reversed(shape)]]),
            ("RF_4", [[(-x, -y) for x, y in shape]])
        ]
        
        return electrodes
    
    

class PiecewiseJunction(BaseJunction):
    def _generate_grids(self) -> None:
        self.X_top = np.linspace(
            start=-self.rf.x_opt,
            stop=-self.c.width/2 - self.rf.width,
            num=len(self.control_points.top) + 1
        )
        
        self.X_bottom = np.linspace(
            start=-self.rf.x_opt,
            stop=-self.c.width/2,
            num=len(self.control_points.bottom) + 1
        )
        
        self.Y_top = self.control_points.top
        self.Y_bottom = self.control_points.bottom
    

    def _build_central_electrodes(self) -> list[tuple[str, list[tuple[float, float]]]]:
        if not self.flags["build_C"]:
            return []

        electrodes = []
        c_index = 1

        if not self.flags["segment_C"]:
            shape = pw_central_shape(junction_geometry=self.jm, 
                                     control_points=[self.X_bottom[1:-1], self.Y_bottom],
                                     trench_width=self.trench_width)
            
            electrodes.append(("C", [shape]))
            
        else:
            shapes = pw_segmented_central_shapes(junction_geometry=self.jm,
                                                 control_points=[self.X_bottom[:-1], self.Y_bottom],
                                                 trench_width=self.trench_width)
            
            electrodes.append(("C", [shapes[0]]))
        
            for shape in shapes[1:]:
                electrodes.extend([
                    (f"L{c_index}", [shape]),
                    (f"D{c_index}", [[(-y, x) for x, y in shape]]),
                    (f"R{c_index}", [[(-x, y) for x, y in reversed(shape)]]),
                    (f"U{c_index}", [[(-y, -x) for x, y in reversed(shape)]]),
                ])
                
                c_index += 1
        
        return electrodes


    def _build_dc_electrodes(self) -> list[tuple[str, list[tuple[float, float]]]]:
        if not self.flags["build_DC"]:
            return []

        electrodes = []
        dc_index = 1

        if not self.flags["segment_DC"]:
            shapes = pw_dc_shapes(junction_geometry=self.jm,
                                  control_points=[self.X_top[1:-1], self.Y_top],
                                  trench_width=self.trench_width)
            
        else:
            shapes = pw_segmented_dc_shapes(junction_geometry=self.jm,
                                            control_points=[self.X_top[:-1], self.Y_top],
                                            trench_width=self.trench_width)

        for shape in shapes:
            electrodes.extend([
                (f"DC{dc_index}", [shape]),
                (f"DC{dc_index+1}", [[(-y, -x) for x, y in reversed(shape)]]),
                (f"DC{dc_index+2}", [[(y, -x) for x, y in shape]]),
                (f"DC{dc_index+3}", [[(-x, y) for x, y in reversed(shape)]]),
                (f"DC{dc_index+4}", [[(-x, -y) for x, y in shape]]),
                (f"DC{dc_index+5}", [[(y, x) for x, y in reversed(shape)]]),
                (f"DC{dc_index+6}", [[(-y, x) for x, y in shape]]),
                (f"DC{dc_index+7}", [[(x, -y) for x, y in reversed(shape)]]),
            ])
            
            dc_index += 8

        return electrodes
 
    
    def _build_rf_electrodes(self) -> list[tuple[str, list[tuple[float, float]]]]:
        if not self.flags["build_RF"]:
            return []

        shape = pw_rf_shapes(junction_geometry=self.jm,
                             bottom_points=[self.X_bottom[1:-1], self.Y_bottom],
                             top_points=[self.X_top[1:-1], self.Y_top])

        electrodes = [
            ("RF_1", [shape]),
            ("RF_2", [[(x, -y) for x, y in reversed(shape)]]),
            ("RF_3", [[(-x, y) for x, y in reversed(shape)]]),
            ("RF_4", [[(-x, -y) for x, y in shape]])
        ]
        
        return electrodes




class SplineJunction(BaseJunction):
    def _generate_grids(self) -> None:
        self.X_top = np.linspace(
            start=-self.rf.x_opt,
            stop=-self.c.width/2 - self.rf.width,
            num=len(self.control_points.top)
        )
        
        self.X_bottom = np.linspace(
            start=-self.rf.x_opt,
            stop=-self.c.width/2,
            num=len(self.control_points.bottom)
        )
        
        self.Y_top = self.control_points.top
        self.Y_bottom = self.control_points.bottom
    

    def _build_central_electrodes(self) -> list[tuple[str, list[tuple[float, float]]]]:
        if not self.flags["build_C"]:
            return []

        electrodes = []
        c_index = 1

        if not self.flags["segment_C"]:
            shape = spline_central_shape(junction_geometry=self.jm, 
                                         control_points=[self.X_bottom, self.Y_bottom],
                                         trench_width=self.trench_width)
            
            electrodes.append(("C", [shape]))
            
        else:
            shapes = spline_segmented_central_shapes(junction_geometry=self.jm,
                                                     control_points=[self.X_bottom, self.Y_bottom],
                                                     trench_width=self.trench_width)
            
            electrodes.append(("C", [shapes[0]]))
        
            for shape in shapes[1:]:
                electrodes.extend([
                    (f"L{c_index}", [shape]),
                    (f"D{c_index}", [[(-y, x) for x, y in shape]]),
                    (f"R{c_index}", [[(-x, y) for x, y in reversed(shape)]]),
                    (f"U{c_index}", [[(-y, -x) for x, y in reversed(shape)]]),
                ])
                c_index += 1
        
        return electrodes


    def _build_dc_electrodes(self) -> list[tuple[str, list[tuple[float, float]]]]:
        if not self.flags["build_DC"]:
            return []

        electrodes = []
        dc_index = 1

        if not self.flags["segment_DC"]:
            shapes = spline_dc_shapes(junction_geometry=self.jm,
                                     control_points=[self.X_top, self.Y_top],
                                     trench_width=self.trench_width)
            
        else:
            shapes = spline_segmented_dc_shapes(junction_geometry=self.jm,
                                               control_points=[self.X_top, self.Y_top],
                                               trench_width=self.trench_width)

        for shape in shapes:
            electrodes.extend([
                (f"DC{dc_index}", [shape]),
                (f"DC{dc_index+1}", [[(-y, -x) for x, y in reversed(shape)]]),
                (f"DC{dc_index+2}", [[(y, -x) for x, y in shape]]),
                (f"DC{dc_index+3}", [[(-x, y) for x, y in reversed(shape)]]),
                (f"DC{dc_index+4}", [[(-x, -y) for x, y in shape]]),
                (f"DC{dc_index+5}", [[(y, x) for x, y in reversed(shape)]]),
                (f"DC{dc_index+6}", [[(-y, x) for x, y in shape]]),
                (f"DC{dc_index+7}", [[(x, -y) for x, y in reversed(shape)]]),
            ])
            
            dc_index += 8

        return electrodes
 
    
    def _build_rf_electrodes(self) -> list[tuple[str, list[tuple[float, float]]]]:
        if not self.flags["build_RF"]:
            return []

        shape = spline_rf_shapes(junction_geometry=self.jm,
                                 bottom_points=[self.X_bottom, self.Y_bottom],
                                 top_points=[self.X_top, self.Y_top])

        electrodes = [
            ("RF_1", [shape]),
            ("RF_2", [[(x, -y) for x, y in reversed(shape)]]),
            ("RF_3", [[(-x, y) for x, y in reversed(shape)]]),
            ("RF_4", [[(-x, -y) for x, y in shape]])
        ]
        
        return electrodes
    
    
    
def shift_polys(polys, dx=0.0, dy=0.0):
    shifted_polys = []
    for poly in polys:
        shifted_poly = [arr + np.array([dx, dy]) for arr in poly]
        shifted_polys.append(shifted_poly)
    return shifted_polys

def rotate_polys(polys, angle_deg=0.0):
    angle_rad = np.deg2rad(angle_deg)
    rot_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                            [np.sin(angle_rad),  np.cos(angle_rad)]])
    rotated_polys = []
    for poly in polys:
        rotated_poly = [np.round(arr @ rot_matrix.T, 1) for arr in poly]
        rotated_polys.append(rotated_poly)
    return rotated_polys

def build_merged_junction(trap_polys_rf, lin_polys_rf, dx=-1500):
    # Generate rotated and shifted arms
    lin_polys_rf_sx = shift_polys(lin_polys_rf, dx=dx)
    lin_polys_rf_up = rotate_polys(lin_polys_rf_sx, -90)
    lin_polys_rf_dx = rotate_polys(lin_polys_rf_sx, -180)
    lin_polys_rf_down = rotate_polys(lin_polys_rf_sx, -270)

    # Convert trap polygons
    trap_polys = [Polygon(p[0]) for p in trap_polys_rf]

    # Convert all linear arm polygons (keeping exact order)
    lin_sets = [lin_polys_rf_sx, lin_polys_rf_up, lin_polys_rf_dx, lin_polys_rf_down]
    lin_polys = [Polygon(arm[i][0]) for arm in lin_sets for i in range(2)]

    # Unpack for readability (same order as your original code)
    poly1, poly2, poly3, poly4 = trap_polys
    lin1, lin2, lin3, lin4, lin5, lin6, lin7, lin8 = lin_polys


    # Merge in fixed order
    merged_poly1 = poly1.union(lin6).union(lin7)
    merged_poly2 = poly2.union(lin4).union(lin5)
    merged_poly3 = poly3.union(lin8).union(lin1)
    merged_poly4 = poly4.union(lin2).union(lin3)

    merged_polys = [merged_poly1, merged_poly2, merged_poly3, merged_poly4]
    merged_paths = [[list(poly.exterior.coords)] for poly in merged_polys]

    return merged_polys, merged_paths
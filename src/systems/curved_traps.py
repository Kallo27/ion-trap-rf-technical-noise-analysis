#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File: curved_traps.py
# Created: 15-07-2025
# Author: Lorenzo Calandra Buonaura <lorenzocb01@gmail.com>
# Institution: University of Innsbruck - UIBK
#
# Classes defining turn trap geometry.
#


########################
# IMPORT ZONE          #
########################

import warnings
import numpy as np

from electrode import (System, PolygonPixelElectrode)

from src.shapes.curved_shapes import (rf_shapes,
                                      central_shape,
                                      dc_shapes,
                                      segmented_central_shapes,
                                      segmented_dc_shapes)

from src.geometry.modules import TurnGeometry

########################
# CLASSES              #
########################

class TurnTrap:
    def __init__(self, turn_geometry: TurnGeometry, trench_width=0, flags=None):
        self.tm = turn_geometry
        self.rf = turn_geometry.RF
        self.dc = turn_geometry.DC
        self.c = turn_geometry.C
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


    def _build_central_electrodes(self) -> list[tuple[str, list[tuple[float, float]]]]:
        if not self.flags["build_C"]:
            return []

        electrodes = []
        c_index = 1

        if not self.flags["segment_C"]:
            shape = central_shape(turn_geometry=self.tm, 
                                  trench_width=self.trench_width)
            
            electrodes.append(("C", [shape]))
            
        else:
            shapes = segmented_central_shapes(turn_geometry=self.tm,
                                              trench_width=self.trench_width)
                    
            for shape in shapes:
                electrodes.extend([
                    (f"C{c_index}", [shape])
                ])
                
                c_index += 1
        
        return electrodes


    def _build_dc_electrodes(self) -> list[tuple[str, list[tuple[float, float]]]]:
        if not self.flags["build_DC"]:
            return []

        electrodes = []
        dc_index = 1

        if not self.flags["segment_DC"]:
            shapes = dc_shapes(turn_geometry=self.tm,
                               trench_width=self.trench_width)
            
        else:
            shapes = segmented_dc_shapes(turn_geometry=self.tm,
                                         trench_width=self.trench_width)

        for shape in shapes:
            electrodes.extend([
                (f"DC_{dc_index}", [shape]),
            ])

            dc_index += 1

        return electrodes
 
    
    def _build_rf_electrodes(self) -> list[tuple[str, list[tuple[float, float]]]]:
        if not self.flags["build_RF"]:
            return []

        shapes = rf_shapes(turn_geometry=self.tm)
        
        electrodes = [
            ("RF_1", [shapes[0]]),
            ("RF_2", [shapes[1]]),
        ]
        
        return electrodes
    

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

    def shift_electrodes(self, electrodes, center):
        shifted = []
        for name, paths in electrodes:
            shifted_paths = [np.array(path) - np.array(center) for path in paths]
            shifted.append((name, shifted_paths))
        return shifted

    def build(self, voltages=None):
        RF_electrodes = self._build_rf_electrodes()
        C_electrodes = self._build_central_electrodes()
        DC_electrodes = self._build_dc_electrodes()
        
        # Get center and shift all electrodes
        center = np.array([self.rf.length / 2, self.rf.length / 2])
        RF_electrodes = self.shift_electrodes(RF_electrodes, center)
        C_electrodes = self.shift_electrodes(C_electrodes, center)
        DC_electrodes = self.shift_electrodes(DC_electrodes, center)

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
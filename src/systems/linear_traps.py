#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File: linear_traps.py
# Created: 12-04-2025
# Author: Lorenzo Calandra Buonaura <lorenzocb01@gmail.com>
# Institution: University of Innsbruck - UIBK
#
# This file defines the classes representing a surface electrode linear ion trap.
# Two configurations are supported:
# - A 5-wire symmetric configuration (standard geometry)
# - A 3-RF configuration (with three radio-frequency electrodes for additional control)
#
# The classes provides methods to construct, modify, and analyze the geometry and
# potentials of the traps based on user-defined electrode layouts and voltage settings.
#


########################
# IMPORT ZONE          #
########################

import warnings
import numpy as np

from electrode import (System, PolygonPixelElectrode)

from src.shapes.linear_shapes import (rf_shapes,
                                      central_shape,
                                      dc_shapes,
                                      segmented_central_shapes,
                                      segmented_dc_shapes)

from src.shapes.linear_shapes import (outer_RF, central_RF)


from src.geometry.modules import FiveWireGeometry, ThreeRFGeometry


########################
# CLASSES              #
########################

class FiveWireTrap:
    def __init__(self, fivewire_geometry: FiveWireGeometry, trench_width=0, flags=None):
        self.fwm = fivewire_geometry
        self.rf = fivewire_geometry.RF
        self.dc = fivewire_geometry.DC
        self.c = fivewire_geometry.C
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
            shape = central_shape(fivewire_geometry=self.fwm, 
                                  trench_width=self.trench_width)
            
            electrodes.append(("C", [shape]))
            
        else:
            shapes = segmented_central_shapes(fivewire_geometry=self.fwm,
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
            shapes = dc_shapes(fivewire_geometry=self.fwm,
                               trench_width=self.trench_width)
            
        else:
            shapes = segmented_dc_shapes(fivewire_geometry=self.fwm,
                                         trench_width=self.trench_width)

        for shape in shapes:
            electrodes.extend([
                (f"DC_{dc_index}", [shape]),
                (f"DC_{dc_index+1}", [[(x, -y) for x, y in reversed(shape)]]),
            ])
            
            dc_index += 2

        return electrodes
 
    
    def _build_rf_electrodes(self) -> list[tuple[str, list[tuple[float, float]]]]:
        if not self.flags["build_RF"]:
            return []

        shape = rf_shapes(fivewire_geometry=self.fwm)
        
        electrodes = [
            ("RF_1", [shape]),
            ("RF_2", [[(x, -y) for x, y in reversed(shape)]]),
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


    def build(self, voltages=None):
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
    
    
class ThreeRFTrap:
    def __init__(self, threerf_geometry: ThreeRFGeometry, trench_width=0, flags=None):
        self.tm = threerf_geometry
        self.outer_rf = threerf_geometry.OUTER
        self.central_rf = threerf_geometry.CENTRAL
        self.trench_width = trench_width

        default_flags = {
            "build_central_RF": True,
            "build_outer_RF": True,
        }

        if flags is None:
            flags = default_flags
        else:
            default_flags.update(flags)
            flags = default_flags

        self.flags = flags
        self._validate_flags()

        self.default_voltages = {
            "central_RF": {"attr": "rf", "value": -100},
            "outer_RF": {"attr": "rf", "value": 100},
        }

        self.last_voltages = None


    def _validate_flags(self) -> None:
        if self.flags["build_central_RF"] and self.central_rf is None:
            raise ValueError("Flag 'build_central_RF' is True, but RF geometry is missing.")
        if not self.flags["build_central_RF"] and self.central_rf is not None:
            warnings.warn("Flag 'build_central_RF' is False, but RF geometry is provided.")
        if self.flags["build_outer_RF"] and self.outer_rf is None:
            raise ValueError("Flag 'build_outer_RF' is True, but RF geometry is missing.")
        if not self.flags["build_outer_RF"] and self.outer_rf is not None:
            warnings.warn("Flag 'build_outer_RF' is False, but RF geometry is provided.")


    def _build_central_RF(self) -> list[tuple[str, list[tuple[float, float]]]]:
        if not self.flags["build_central_RF"]:
            return []

        electrodes = []

        shape = central_RF(threerf_geometry=self.tm)
            
        electrodes.append(("RF_1", [shape]))

        return electrodes


    def _build_outer_RF(self) -> list[tuple[str, list[tuple[float, float]]]]:
        if not self.flags["build_outer_RF"]:
            return []

        electrodes = []

        shape = outer_RF(threerf_geometry=self.tm)
            
        electrodes.extend([
            (f"RF_2a", [shape]),
            (f"RF_2b", [[(-x, y) for x, y in reversed(shape)]]),
        ])
            
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


    def build(self, voltages=None, flags=None):
        original_flags = self.flags.copy()

        if flags is not None:
            updated_flags = original_flags.copy()
            updated_flags.update(flags)
            self.flags = updated_flags
            self._validate_flags()

        central_RF_electrodes = self._build_central_RF()
        outer_RF_electodes = self._build_outer_RF()

        all_electrodes = central_RF_electrodes + outer_RF_electodes
        system = System([PolygonPixelElectrode(name=name, paths=map(np.array, paths)) for name, paths in all_electrodes])

        electrodes_by_type = {
            "central_RF": central_RF_electrodes,
            "outer_RF": outer_RF_electodes
        }

        volt = self.default_voltages.copy()
        if voltages:
            volt.update({k: {"attr": self.default_voltages[k]["attr"], "value": v} for k, v in voltages.items()})
        self.last_voltages = volt
        self.assign_voltages(system, electrodes_by_type, voltages)
        
        self.flags = original_flags

        return system
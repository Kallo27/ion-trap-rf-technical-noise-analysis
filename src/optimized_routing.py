#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File: routing.py
# Created: 22-07-2025
# Author: Lorenzo Calandra Buonaura <lorenzocb01@gmail.com>
# Institution: University of Innsbruck - UIBK
#
# Main script for optimizing wire routing. 
#


########################
# IMPORT ZONE          #
########################

import os
import gdspy
import numpy as np

from shapely.geometry import Polygon
from electrode import polygons, System, PolygonPixelElectrode

from src.routing.pins import Pin, PinZonePlacer
from src.routing.electrodes import DCElectrode
from src.routing.states import RoutingState
from src.routing.annealing import simulated_annealing
from src.geometry.geometry_utils import check_all_ccw
from src.io.saving import convert_gdsii_to_gdspy


########################
# FUNCTIONS            #
########################
def run_routing(filename):
    print(f"Running routing for: {filename}")

    path = os.path.join("./gds_files/", filename)
    output_name = os.path.basename(filename)

    lib = gdspy.GdsLibrary(infile=path)

    rf_polys = []
    dc_polys = []

    for cell in lib.cells.values():
        labels = cell.labels
        polys = cell.polygons
        
        i=0
        for poly in polys:
            if poly.layers[0]==0:
                i+=1
                
        for poly, label in zip(polys[1:], labels):
                if "RF" not in label.text:
                    dc_polys.append(poly.polygons)
                else:
                    rf_polys.append(poly.polygons)


    polys = dc_polys + rf_polys
    system = System([PolygonPixelElectrode(name=n.text, paths=[np.array(r)[::-1] for r in p]) for n, p in zip(labels, polys)])
    check_all_ccw(system)

    # Create the DC electrodes
    polys = [Polygon(p[0]) for p in dc_polys]
    dc_electrodes = [DCElectrode(poly, name) for poly, name in zip(polys, labels)]


    # Create and add full corner pin layout
    placer = PinZonePlacer()
    pin_cell = placer.get_cell()


    # Initialize and run SA
    layer_options = [3, 4]
    initial_state = RoutingState(placer, dc_electrodes, layer_options)

    best_state = simulated_annealing(
        initial_state=initial_state,
        initial_temp=100.0,
        cooling_rate=0.995,
        iterations=3000
    )


    # Save trap to a gds file
    trap_polygons = polygons.Polygons.from_system(system)
    trap_polygons_gds = trap_polygons.to_gds(
        scale=1.,
        poly_layer=(0, 0),
        gap_layer=(1, 0),
        text_layer=(20, 0),
        via_layer=(10, 0),
        phys_unit=1.,
        name=f"trap_polygons_{output_name}",
        edge=3000,
        gap_width=5
    )

    lib = convert_gdsii_to_gdspy(trap_polygons_gds)

    # Save routing        
    cell = lib.new_cell(f"ROUTED_SA_{output_name}")

    # Add all paths
    for path in best_state.paths.values():
        cell.add(path.path)  # assuming path.path is a FlexPath

    # Add fixed pins
    fixed_pins = []
    for key in best_state.fixed_assignments.keys():
        fixed_pins.append(Pin((key.x, key.y), layer=3))
        
    for pin in fixed_pins:
        fixed_pin_cell = pin.get_cell()
        lib.add(fixed_pin_cell)                          # Add individual pin cell to library
        cell.add(gdspy.CellReference(fixed_pin_cell))    # Add reference to main cell
        
    # Add total pins cell 
    cell.add(gdspy.CellReference(pin_cell))

    # Add the pin_cell to the lib
    lib.add(cell)
    lib.add(pin_cell)

    # Create TOP cell to visualize everything
    top = lib.new_cell(f"TOP_{output_name}")
    top.add(gdspy.CellReference(lib.cells[f"trap_polygons_{output_name}"]))
    top.add(gdspy.CellReference(lib.cells[f"ROUTED_SA_{output_name}"]))
    lib.add(top)

    # Save to GDS
    path = os.path.join("./gds_files/routed_files", output_name)
    lib.write_gds(outfile=path)
    
    print(f"Saved routed GDS to: {path}")


########################
# MAIN                 #
########################

if __name__ == "__main__":
    # Example 1: Loop over a fixed list
    filenames = ["turn/layout_A/turn_A_100um.gds", "turn/layout_B/turn_B_100um.gds",
                 "linear/layout_A/fivewire_A_100um.gds", "linear/layout_B/fivewire_B_100um.gds",
                 "junction/layout_A/piecewise_junction_A_100um.gds", "junction/layout_B/piecewise_junction_B_100um.gds",
                 "junction/layout_A/spline_junction_A_100um.gds", "junction/layout_B/spline_junction_B_100um.gds",]
    for fname in filenames:
        run_routing(fname)
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File: turn_builder.py
# Created: 15-07-2025
# Author: Lorenzo Calandra Buonaura <lorenzocb01@gmail.com>
# Institution: University of Innsbruck - UIBK
#
# Main script for building and saving turn trap geometries.
#


########################
# IMPORT ZONE          #
########################


from electrode import polygons

from src.systems.curved_traps import TurnTrap
from src.io.loading import load_turn_geometry
from src.geometry.geometry_utils import check_all_ccw
from src.graphics.visualize_trap import plot_trap_with_voltages
from src.io.saving import convert_gdsii_to_gdspy


########################
# MAIN ZONE            #
########################

geometry = load_turn_geometry("src/resources/turn_parameters.json")

# Trench width
trench_width = 5

# Set segmentation flags
flags = {
    "build_RF": True,
    "build_DC": True,
    "build_C": True,
    "segment_DC": True,
    "segment_C": False
}

# Build trap
trap = TurnTrap(geometry, trench_width, flags)
system = trap.build()

check_all_ccw(system)

# Save trap to a gds file
trap_polygons = polygons.Polygons.from_system(system)

trap_polygons_gds = trap_polygons.to_gds(
    scale=1.,
    poly_layer=(0, 0),
    gap_layer=(1, 0),
    text_layer=(2, 0),
    via_layer=(10, 0),
    phys_unit=1.,
    name="trap_polygons",
    edge=3000,
    gap_width=5
)

gdspy_library = convert_gdsii_to_gdspy(trap_polygons_gds)
gdspy_library.write_gds('./gds_files/turn/layout_B/turn_B_100um.gds')

# Visulize trap
# plot_trap_with_voltages(system)
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File: saving.py
# Created: 15-04-2025
# Author: Lorenzo Calandra Buonaura <lorenzocb01@gmail.com>
# Institution: University of Innsbruck - UIBK
#
# Functions for saving trap geometries as GDS files and exporting optimization results.
#


########################
# IMPORT ZONE          #
########################

import gdsii.library
import numpy as np
import gdsii
import gdspy
import os
import json
from datetime import datetime
import shutil


########################
# FUNCTIONS            #
########################

def convert_gdsii_to_gdspy(gdsii_library: gdsii.library.Library) -> gdspy.GdsLibrary:
    gdspy_library = gdspy.GdsLibrary()
  
    # Iterate over the structures in the gdsii library and add them to gdspy library
    for structure in gdsii_library:
        structure_name = structure.name.decode('utf-8') if isinstance(structure.name, bytes) else structure.name
        gdspy_cell = gdspy.Cell(structure_name)
  
        # Add elements (polygons, paths, texts, etc.) to the gdspy cell
        for element in structure[::-1]:
            if isinstance(element, gdsii.elements.Boundary):
                gdspy_polygon = gdspy.Polygon(element.xy, layer=element.layer)
                gdspy_cell.add(gdspy_polygon)

                # Compute middle position only if polygon is non-empty
                xy_array = np.array(element.xy[:-1])
                if xy_array.size > 0:
                    middle_position = (np.mean(xy_array[:, 0]), np.mean(xy_array[:, 1]))
                          
            elif isinstance(element, gdsii.elements.Path):
                gdspy_path = gdspy.FlexPath(element.xy, width=element.width, layer=element.layer)
                gdspy_cell.add(gdspy_path)
              
            elif isinstance(element, gdsii.elements.Text):
                label_text = element.string.decode('utf-8') if isinstance(element.string, bytes) else element.string
                xy_array = np.array(element.xy)
                label_position = (np.mean(xy_array[:, 0]), np.mean(xy_array[:, 1]))

                gdspy_label = gdspy.Label(
                    text=label_text,
                    position=label_position,
                    anchor='o',
                    layer=element.layer,
                    texttype=0
                )
                gdspy_cell.add(gdspy_label)

        gdspy_library.add(gdspy_cell)
  
    return gdspy_library


def save_optimization(optimized_points, input_file, base_results_dir="optimization_results"):
    top_points, bottom_points = optimized_points

    # Create a timestamped folder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base_results_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Load original input
    with open(input_file, "r") as f:
        data = json.load(f)

    # Save original input file for reproducibility
    original_input_path = os.path.join(run_dir, "input.json")
    shutil.copy(input_file, original_input_path)

    # Update with optimized control points
    data["top_control"] = top_points.tolist()
    data["bottom_control"] = bottom_points.tolist()

    # Save updated output
    output_path = os.path.join(run_dir, "result.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Saved optimization to {run_dir}")
    return run_dir

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File: loading.py
# Created: 15-04-2025
# Author: Lorenzo Calandra Buonaura <lorenzocb01@gmail.com>
# Institution: University of Innsbruck - UIBK
#
# Functions for loading data, including trap geometries 
# and heating rate measurements.
# 


########################
# IMPORT ZONE          #
########################

import json
import pandas as pd

from src.geometry.modules import (Junction_RFdimensions,
                                  Junction_Cdimensions,
                                  Junction_DCdimensions,
                                  JunctionGeometry,
                                  ControlPoints,
                                  Linear_RFdimensions,
                                  Linear_DCdimensions,
                                  Linear_Cdimensions,
                                  FiveWireGeometry,
                                  ThreeRFGeometry,
                                  Turn_RFdimensions,
                                  Turn_DCdimensions,
                                  Turn_Cdimensions,
                                  TurnGeometry)


########################
# FUNCTIONS            #
########################

def load_junction_geometry(file_name: str):
    # Load configuration from JSON file
    with open(file_name, 'r') as file:
        config = json.load(file)

    # Extract parameters from config
    h = config['h']
    C_width = config['C_width'] * h
    C_heights = config['C_heights']
    C_radius = config['C_radius'] if 'C_radius' in config else None
    RF_length = config['RF_length']
    RF_width = config['RF_width'] * h
    X_opt = config['X_opt']
    DC_width = config['DC_width'] * C_width
    DC_height = config['DC_height'] * h
    n_DC = config['n_DC']
    top_control = config['top_control']
    bottom_control = config['bottom_control']
    
    C_dimensions = Junction_Cdimensions(width=C_width, heights=C_heights, radius=C_radius)
    RF_dimensions = Junction_RFdimensions(length=RF_length, width=RF_width, x_opt=X_opt)
    DC_dimensions = Junction_DCdimensions(width=DC_width, height=DC_height, count=n_DC)
    control_points = ControlPoints(top=top_control, bottom=bottom_control)

    geometry = JunctionGeometry(RF=RF_dimensions,
                                DC=DC_dimensions,
                                C=C_dimensions,
                                points=control_points)
    
    return geometry


def load_non_opt_junction_geometry(file_name: str):
    # Load configuration from JSON file
    with open(file_name, 'r') as file:
        config = json.load(file)

    # Extract parameters from config
    h = config['h']
    C_width = config['C_width'] * h
    C_heights = config['C_heights']
    C_radius = config['C_radius'] if 'C_radius' in config else None
    RF_length = config['RF_length']
    RF_width = config['RF_width'] * h
    DC_width = config['DC_width'] * C_width
    DC_height = config['DC_height'] * h
    n_DC = config['n_DC']
    
    C_dimensions = Junction_Cdimensions(width=C_width, heights=C_heights, radius=C_radius)
    RF_dimensions = Junction_RFdimensions(length=RF_length, width=RF_width)
    DC_dimensions = Junction_DCdimensions(width=DC_width, height=DC_height, count=n_DC)

    geometry = JunctionGeometry(RF=RF_dimensions,
                                DC=DC_dimensions,
                                C=C_dimensions)
    
    return geometry
    

def load_fivewire_geometry(file_name: str):
    # Load configuration from JSON file
    with open(file_name, 'r') as file:
        config = json.load(file)
        
    # Extract parameters from config
    h = config['h']
    C_width = config['C_width'] * h
    C_heights = config['C_heights']
    RF_length = config['RF_length']
    RF_width = config['RF_width'] * h
    DC_width = config['DC_width'] * h
    DC_height = config['DC_height'] * h
    n_DC = config['n_DC']
    
    C_dimensions = Linear_Cdimensions(width=C_width, heights=C_heights)
    RF_dimensions = Linear_RFdimensions(length=RF_length, width=RF_width)
    DC_dimensions = Linear_DCdimensions(width=DC_width, height=DC_height, count=n_DC)
    
    geometry = FiveWireGeometry(RF=RF_dimensions,
                                DC=DC_dimensions,
                                C=C_dimensions)
    
    return geometry



def load_threeRF_geometry(file_name: str):
    # Load configuration from JSON file
    with open(file_name, 'r') as file:
        config = json.load(file)
        
    # Extract parameters from config
    central_RF_length = config['central_RF_length']
    central_RF_width = config['central_RF_width']
    outer_RF_length = config['outer_RF_length']
    outer_RF_width = config['outer_RF_width']
    outer_RF_offset = config['outer_RF_offset']
    
    central_RF_dimensions = Linear_RFdimensions(length=central_RF_length, width=central_RF_width)
    outer_RF_dimensions = Linear_RFdimensions(length=outer_RF_length, width=outer_RF_width, offset=outer_RF_offset)
    
    geometry = ThreeRFGeometry(CENTRAL=central_RF_dimensions,
                               OUTER=outer_RF_dimensions)
    
    return geometry


def load_hr_carrier_data(excel_path: str):
    df = pd.read_excel(excel_path)

    csi_exp = df.iloc[:, 0].values
    h_exp = df.iloc[:, 1].values
    error_h_exp = df.iloc[:, 2].values
    freq_exp = df.iloc[:, 3].values

    return csi_exp, h_exp, error_h_exp, freq_exp




def load_turn_geometry(file_name: str):
    # Load configuration from JSON file
    with open(file_name, 'r') as file:
        config = json.load(file)
        
        
    # Extract parameters from config
    h = config['h']
    C_width = config['C_width'] * h
    n_C = config['n_C']
    RF_length = config['RF_length']
    RF_width = config['RF_width'] * h
    DC_height = config['DC_height'] * h
    n_DC = config['n_DC']
    
    C_dimensions = Turn_Cdimensions(width=C_width, count=n_C)
    RF_dimensions = Turn_RFdimensions(length=RF_length, width=RF_width)
    DC_dimensions = Turn_DCdimensions(height=DC_height, count=n_DC)
    
    geometry = TurnGeometry(RF=RF_dimensions,
                            DC=DC_dimensions,
                            C=C_dimensions)
    
    return geometry
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File: visualize_trap.py
# Created: 15-04-2025
# Author: Lorenzo Calandra Buonaura <lorenzocb01@gmail.com>
# Institution: University of Innsbruck - UIBK
#
# Visualization functions for generating plots and graphics related
# to trap geometries and electrode layouts.
#


########################
# IMPORT ZONE          #
########################

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from electrode import System


########################
# FUNCTIONS            #
########################

def plot_trap(s: System, x_lim=[-3500, 3500], y_lim=[-3500, 3500]):
    # Create the figure for the trap geometry
    fig1, ax1 = plt.subplots(figsize=(7, 7))
    ax1.set_aspect("equal")
    ax1.set_xlim(x_lim[0], x_lim[1])
    ax1.set_ylim(y_lim[0], y_lim[1])
    ax1.set_xlabel('x (μm)')
    ax1.set_ylabel('y (μm)')
    s.plot(ax1)
    ax1.set_title("Trap Geometry")
    
    # Show
    plt.show()
    
def plot_trap_with_voltages(s: System, x_lim=[-1000, 1000], y_lim=[-1000, 1000]):
    # Create the first figure for the trap geometry
    fig1, ax1 = plt.subplots(figsize=(7, 7))
    ax1.set_aspect("equal")
    ax1.set_xlim(x_lim[0], x_lim[1])
    ax1.set_ylim(y_lim[0], y_lim[1])
    ax1.set_xlabel('x (μm)')
    ax1.set_ylabel('y (μm)')
    s.plot(ax1)
    ax1.set_title("Trap Geometry")
    
    # Create the second figure for the RF voltages
    fig2, ax2 = plt.subplots(figsize=(7, 7))
    ax2.set_aspect("equal")
    ax2.set_xlim(x_lim[0], x_lim[1])
    ax2.set_ylim(y_lim[0], y_lim[1])
    ax2.set_xlabel('x (μm)')
    ax2.set_ylabel('y (μm)')
    s.plot_voltages(ax2, u=s.rfs)
    ax2.set_title("RF Voltages")
    
    # Create the third figure for the DC voltages
    fig3, ax3 = plt.subplots(figsize=(7, 7))
    ax3.set_aspect("equal")
    ax3.set_xlim(x_lim[0], x_lim[1])
    ax3.set_ylim(y_lim[0], y_lim[1])
    ax3.set_xlabel('x (μm)')
    ax3.set_ylabel('y (μm)')
    s.plot_voltages(ax3, u=s.dcs)
    ax3.set_title("DC Voltages")
    
    # Compute voltage range for colorbars
    vmin = np.min([np.min(s.rfs), np.min(s.dcs)])
    vmax = np.max([np.max(s.rfs), np.max(s.dcs)])

    # Colormap normalization
    cmap = plt.cm.RdBu_r
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # Add colorbars to the second and third figures
    cb1 = fig2.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax2, shrink=0.9, aspect=35)
    cb1.ax.tick_params(labelsize=10)
    cb1.set_label('Voltage', fontsize=10)
    
    cb2 = fig3.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax3, shrink=0.9, aspect=35)
    cb2.ax.tick_params(labelsize=10)
    cb2.set_label('Voltage', fontsize=10)

    # Show the figures
    plt.show()
    
def plot_voltages(s: System, x_lim=[3500, 3500], y_lim=[3500, 3500]):
    # Create the first figure for the RF voltages
    fig2, ax1 = plt.subplots(figsize=(7, 7))
    ax1.set_aspect("equal")
    ax1.set_xlim(x_lim[0], x_lim[1])
    ax1.set_ylim(y_lim[0], y_lim[1])
    ax1.set_xlabel('x (μm)')
    ax1.set_ylabel('y (μm)')
    s.plot_voltages(ax1, u=s.rfs)
    ax1.set_title("RF Voltages")
    
    # Create the second figure for the DC voltages
    fig3, ax2 = plt.subplots(figsize=(7, 7))
    ax2.set_aspect("equal")
    ax2.set_xlim(x_lim[0], x_lim[1])
    ax2.set_ylim(y_lim[0], y_lim[1])
    ax2.set_xlabel('x (μm)')
    ax2.set_ylabel('y (μm)')
    s.plot_voltages(ax2, u=s.dcs)
    ax2.set_title("DC Voltages")
    
    # Compute voltage range for colorbars
    vmin = np.min([np.min(s.rfs), np.min(s.dcs)])
    vmax = np.max([np.max(s.rfs), np.max(s.dcs)])

    # Colormap normalization
    cmap = plt.cm.RdBu_r
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # Add colorbars to the figures
    cb1 = fig2.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax1, shrink=0.9, aspect=35)
    cb1.ax.tick_params(labelsize=10)
    cb1.set_label('Voltage', fontsize=10)
    
    cb2 = fig3.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax2, shrink=0.9, aspect=35)
    cb2.ax.tick_params(labelsize=10)
    cb2.set_label('Voltage', fontsize=10)

    # Show the figures
    plt.show()
    

def plot_trap_nice(system, yellow_indices=[25,26,27,28],
                   blue_indices=[0,1,2,3], red_indices=None, box=True):
    
    if red_indices is None:
        red_indices = [i for i in range(len(system)) if i not in yellow_indices + blue_indices]
    
    def get_color(i):
        if i in blue_indices:
            return "#348ABD"
        elif i in yellow_indices:
            return "#CCCCCC"
        elif i in red_indices:
            return "#E24A33"
        else:
            return "#CCCCCC"

    fig = plt.figure(figsize=(8,8))
    ax = plt.gca()
    
    for i, e in enumerate(system):
        for arr in e.paths:
            arr = np.array(arr)
            if not np.all(arr[0] == arr[-1]):
                arr = np.vstack([arr, arr[0]])  # close path
            plt.fill(arr[:,0], arr[:,1], color=get_color(i), edgecolor='k', linewidth=0.5)

        # Compute polygon centroid for label
        poly = np.array(e.paths[0])
        if not np.all(poly[0] == poly[-1]):
            poly = np.vstack([poly, poly[0]])
        x = poly[:,0]
        y = poly[:,1]
        A = 0.5*np.sum(x[:-1]*y[1:] - x[1:]*y[:-1])
        Cx = np.sum((x[:-1]+x[1:])*(x[:-1]*y[1:] - x[1:]*y[:-1]))/(6*A)
        Cy = np.sum((y[:-1]+y[1:])*(x[:-1]*y[1:] - x[1:]*y[:-1]))/(6*A)

        # if i < len(system) - 4:
        #     ax.text(Cx, Cy, e.name, fontsize=10, fontfamily='STIXGeneral', ha='center', va='center', color='black')

    # Draw the red empty box
    box_coords = np.array([
        [-660, -90],
        [660, -90],
        [660, 90],
        [-660, 90],
        [-660, -90]  # close the box
    ])
    
    if box:
        # Draw a thin filled box
        plt.fill(box_coords[:,0], box_coords[:,1], color='#FBC15E', alpha=0.5, edgecolor='black', linewidth=1.5)
        ax.set_aspect('equal', 'box')
        
    plt.axis('off')
    return fig

def plot_rf_nice(system_rf, y_top=None, y_bottom=None):
    fig = plt.figure(figsize=(6,6))
    for e in system_rf:
        for arr in e.paths:
            arr = np.array(arr)
            if not np.all(arr[0] == arr[-1]):
                arr = np.vstack([arr, arr[0]])
            plt.fill(arr[:,0], arr[:,1], color="#D16002", edgecolor='k', linewidth=0.5)

    if y_top and y_bottom:
        x_top = np.linspace(-700, -241.5, 9)
        x_top = np.concatenate([x_top[1:-1], [-y_top[-1]]])
        x_bottom = np.linspace(-700, -41.5, 9)
        x_bottom = np.concatenate([x_bottom[1:-1], [-y_bottom[-1]]])

        plt.plot(x_top[:8], y_top, 'o', color='green', markersize=5)
        plt.plot(x_bottom[:8], y_bottom, 'o', color='green', markersize=5)
    plt.axis('off')
    return fig

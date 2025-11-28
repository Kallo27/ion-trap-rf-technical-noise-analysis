#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File: visualize_potential.py
# Created: 16-04-2025
# Author: Lorenzo Calandra Buonaura <lorenzocb01@gmail.com>
# Institution: University of Innsbruck - UIBK
#
# Visualization functions for generating plots and graphics related
# to pseudopotential and shims sets (waveforms) simulations.
#


########################
# IMPORT ZONE          #
########################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from labellines import labelLines
from matplotlib import rcParams
from matplotlib.ticker import ScalarFormatter

from src.simulators.trap_simulators import TrapPhysicsSimulator


# Choose matplotlib options
rcParams['font.family'] = 'STIXGeneral'
rcParams['mathtext.fontset'] = 'stix'
rcParams['axes.unicode_minus'] = False
rcParams.update({
    "font.size": 14,             # base font size
    "axes.labelsize": 15,        # x/y labels
    "axes.titlesize": 15,        # title
    "xtick.labelsize": 12,       # tick labels
    "ytick.labelsize": 12,
    "legend.fontsize": 12,       # legend
    "axes.linewidth": 1.2,
    "lines.linewidth": 1.5,      # thicker default lines
    "grid.alpha": 0.6,
    "grid.linestyle": "--",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

########################
# FUNCTIONS            #
########################

def plot_ps(x_values, results):
    # Extract pseudo-potential values
    pseudopot_values = results["pseudopot_values"]
    positions = results["positions"]
    omega_sec = results["omega_sec_values"]
    
    fig, ax = plt.subplots(4, 1, figsize=(8, 12))  # 4 subplots stacked vertically
  
    # First subplot: Pseudo-potential and its gradient
    ax[0].plot(x_values, np.array(pseudopot_values) / 1e-3, label="Pseudo-potential")
    #gradient_values = np.gradient(np.array(pseudopot_values) / 1e-3, x_values)
    #ax[0].plot(x_values, gradient_values * 10, linestyle='--', label=f'Gradient (magnified * 10) meV/μm\nIntegral = {np.sum(gradient_values):.3f}')
    ax[0].set_ylabel("Pseudo-potential (meV)")
    ax[0].set_title("Pseudo-potential along x")
    ax[0].grid(True, linestyle="--", alpha=0.6)
    ax[0].legend(loc='upper left')
  
    # Second subplot: Position components
    ax[1].plot(x_values, np.array(positions)[:, 0], label='X Position')
    ax[1].plot(x_values, np.array(positions)[:, 1], label='Y Position')
    ax[1].plot(x_values, np.array(positions)[:, 2], label='Ion-trap distance')
    ax[1].set_ylabel("Ion-trap distance (μm)")
    ax[1].set_title("Position of the minimum along x")
    ax[1].grid(True, linestyle="--", alpha=0.6)
    ax[1].legend(loc='upper left')
  
    # Third subplot: Axial secular frequency
    omega_sec = np.array(omega_sec)  # Ensure numpy array
    ax[2].plot(x_values, omega_sec[:, 0] / (2 * np.pi) / 1e6, label='Axial frequency', color='C0')
    ax[2].set_ylabel("Secular frequency (MHz)")
    ax[2].set_title("Axial secular frequency along x")
    ax[2].grid(True, linestyle="--", alpha=0.6)
    ax[2].legend(loc='upper left')
  
    # Fourth subplot: Radial secular frequencies
    var_radial_1 = np.var(omega_sec[:, 1] / (2 * np.pi) / 1e6)
    var_radial_2 = np.var(omega_sec[:, 2] / (2 * np.pi) / 1e6)
    ax[3].plot(x_values, omega_sec[:, 1] / (2 * np.pi) / 1e6, label=f'Radial 1 (Var = {var_radial_1:.3f})', color='C1')
    ax[3].plot(x_values, omega_sec[:, 2] / (2 * np.pi) / 1e6, label=f'Radial 2 (Var = {var_radial_2:.3f})', color='C2')
    ax[3].set_ylabel("Secular frequency (MHz)")
    ax[3].set_xlabel("Position along x (μm)")
    ax[3].set_title("Radial secular frequencies along x")
    ax[3].grid(True, linestyle="--", alpha=0.6)
    ax[3].legend(loc='lower left')
    
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


def plot_joint_ps(x_values, non_opt_params, opt_params):
    pseudopot_values_1 = non_opt_params["pseudopot_values"]
    positions_1 = non_opt_params["positions"]
    omega_sec_1 = non_opt_params["omega_sec_values"]
    
    pseudopot_values_2 = opt_params["pseudopot_values"]
    positions_2 = opt_params["positions"]
    omega_sec_2 = opt_params["omega_sec_values"]
    
    fig, ax = plt.subplots(4, 1, figsize=(8, 12))  # 4 subplots stacked vertically
    
    # First subplot: Pseudo-potential and its gradient
    ax[0].plot(x_values, np.array(pseudopot_values_1) / 1e-3, label="Pseudo-potential non opt", linestyle='--')
    ax[0].plot(x_values, np.array(pseudopot_values_2) / 1e-3, label="Pseudo-potential opt")
    ax[0].set_ylabel("Pseudo-potential (meV)")
    ax[0].set_title("Pseudo-potential along x")
    ax[0].grid(True, linestyle="--", alpha=0.6)
    ax[0].legend(loc='upper left')
    
    # Second subplot: Position components
    ax[1].plot(x_values, np.array(positions_1)[:, 2], label='Ion-trap distance non opt', linestyle='--')
    ax[1].plot(x_values, np.array(positions_2)[:, 2], label='Ion-trap distance opt')
    ax[1].set_ylabel("Ion-trap distance (μm)")
    ax[1].set_title("Position of the minimum along x")
    ax[1].grid(True, linestyle="--", alpha=0.6)
    ax[1].legend(loc='upper left')
  
    # Third subplot: Axial secular frequency
    omega_sec_1, omega_sec_2 = np.array(omega_sec_1), np.array(omega_sec_2)  # Ensure numpy array
    ax[2].plot(x_values, omega_sec_1[:, 0] / (2 * np.pi) / 1e6, label='Axial frequency non opt', color='C0', linestyle='--')
    ax[2].plot(x_values, omega_sec_2[:, 0] / (2 * np.pi) / 1e6, label='Axial frequency opt', color='C1')
    ax[2].set_ylabel("Frequency (MHz)")
    ax[2].set_title("Axial secular frequency along x")
    ax[2].grid(True, linestyle="--", alpha=0.6)
    ax[2].legend(loc='upper left')
  
    # Fourth subplot: Radial secular frequencies
    var_radial_1 = np.var(omega_sec_2[:, 1] / (2 * np.pi) / 1e6)
    var_radial_2 = np.var(omega_sec_2[:, 2] / (2 * np.pi) / 1e6)
    ax[3].plot(x_values, omega_sec_1[:, 1] / (2 * np.pi) / 1e6, label=f'Radial 1 non opt', color='C1', linestyle='--')
    ax[3].plot(x_values, omega_sec_1[:, 2] / (2 * np.pi) / 1e6, label=f'Radial 2 non opt', color='C2', linestyle='--')
    ax[3].plot(x_values, omega_sec_2[:, 1] / (2 * np.pi) / 1e6, label=f'Radial 1 (Var = {var_radial_1:.3f})', color='C1')
    ax[3].plot(x_values, omega_sec_2[:, 2] / (2 * np.pi) / 1e6, label=f'Radial 2 (Var = {var_radial_2:.3f})', color='C2')
    ax[3].set_ylabel("Frequency (MHz)")
    ax[3].set_xlabel("Position along x (μm)")
    ax[3].set_title("Radial secular frequencies along x")
    ax[3].grid(True, linestyle="--", alpha=0.6)
    ax[3].legend(loc='lower left')
    
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()
    
    
def plot_electric_field(E_field, x_values):
    E_field = np.array(E_field)  # Convert list to numpy array for easier indexing
    
    fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True)  # Create subplots with shared x-axis
    
    components = ['Ex', 'Ey', 'Ez']
    for i in range(3):  # Loop over the three components
        ax[i].plot(x_values, E_field[:, i], label=f'{components[i]} component', color=f'C{i}')
        ax[i].set_ylabel('Electric field (eV/μm)')
        ax[i].legend()
        ax[i].grid(True)

    ax[-1].set_xlabel('x (μm)')  # Set x label on the last subplot
    fig.suptitle('Electric field components (along x)')

    plt.tight_layout()
    plt.show()
    
    
def plot_waveforms(x_values, waveforms, threshold, names, cmap_name='tab10'):
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()

    # Choose a color map
    cmap = plt.get_cmap(cmap_name)
    n_lines = waveforms.shape[1]
    colors = [cmap(i / n_lines) for i in range(n_lines)]

    lines = []
    for i, wf in enumerate(waveforms.T):
        if np.max(np.abs(wf)) > threshold:
            line, = ax.plot(x_values, wf, color=colors[i], label=names[i])
            lines.append(line)

    labelLines(lines, align=True, fontsize=8, outline_color='white', outline_width=2)

    # Slight vertical lift for labels
    y_min, y_max = ax.get_ylim()
    for txt in ax.texts:
        txt.set_y(txt.get_position()[1] + 0.03 * (y_max - y_min))
    
    x_shade_start = -41.5
    x_shade_end = 0
    
    ax.axvspan(
            -x_shade_start,
            x_shade_start,
            color="#EEE8AC",
            alpha=0.5  # transparency
        )
    ax.axvspan(
            x_shade_start - 199,
            x_shade_start,
            color="#FBC901",
            alpha=0.5
        )
    
    ax.axvspan(
            -(x_shade_start - 199),
            -x_shade_start,
            color="#FBC901",
            alpha=0.5
        )
        
    ax.set_xlabel(r"$x \ (\mathrm{\mu m})$")
    ax.set_ylabel(r"Voltage $ (\mathrm{V})$")
    ax.grid(True)    
    return fig

def plot_waveforms_2x2(x_values, waveforms, threshold, names, cmap_name='tab10'):
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
    axes = axes.flatten()

    cmap = plt.get_cmap(cmap_name)
    n_lines = waveforms.shape[1]
    colors = plt.get_cmap(cmap_name).colors

    def group_from_name(name: str) -> int:
        n = name.upper()
        if "DC" in n:
            return 0
        elif "C" in n or "U" in n or "D" in n:
            return 1
        elif "L" in n:
            return 2
        elif "R" in n:
            return 3
        else:
            return 3  # fallback

    group_color_index = {0: 0, 1: 0, 2: 0, 3: 0}

    for i in range(n_lines):
        wf = waveforms[:, i]
        if np.max(np.abs(wf)) <= threshold:
            continue

        group = group_from_name(names[i])
        ax = axes[group]
        
        # --- Skip every second DC line in a pair ---
        if "DC" in names[i].upper():
            # Only plot even-indexed lines of each pair (0,2,4,…)
            if i % 2 != 0:
                continue

        j = group_color_index[group] % len(colors)
        ax.plot(x_values, wf, color=colors[j], label=names[i])
        group_color_index[group] += 1

    # Decorations per subplot
    for i, ax in enumerate(axes):
        ax.grid(True)

        x_shade_start = -41.5
        ax.axvspan(-x_shade_start, x_shade_start, color="#EEE8AC", alpha=0.5)
        ax.axvspan(x_shade_start - 199, x_shade_start, color="#FBC901", alpha=0.5)
        ax.axvspan(-(x_shade_start - 199), -x_shade_start, color="#FBC901", alpha=0.5)

        # Custom legend for first subplot
        if i == 0:
            ax.legend(ncol=4, loc='lower center', fontsize=10)
        else:
            ax.legend()


    axes[2].set_xlabel(r"$x \ (\mathrm{\mu m})$")
    axes[3].set_xlabel(r"$x \ (\mathrm{\mu m})$")
    axes[0].set_ylabel(r"Voltage $ (\mathrm{V})$")
    axes[2].set_ylabel(r"Voltage $ (\mathrm{V})$")
    plt.tight_layout()

    return fig


def plot_ps_analysis(x_values, ps_results, nonopt_ps_results=None):
    x_values = np.array(x_values)
    sci_formatter = ScalarFormatter(useMathText=True)
    sci_formatter.set_powerlimits((-3, 3))

    # --- Extract optimized quantities ---
    pseudopot_values = np.array(ps_results["pseudopot_values"]) / 1e-3
    grad_values = np.array(ps_results["grad_values"]) / 1e-3
    positions = np.array(ps_results["positions"])
    omega_sec = np.array(ps_results["omega_sec_values"])

    # --- Extract non-optimized quantities if provided ---
    if nonopt_ps_results is not None:
        pseudopot_values_no = np.array(nonopt_ps_results["pseudopot_values"]) / 1e-3
        grad_values_no = np.array(nonopt_ps_results["grad_values"]) / 1e-3
        positions_no = np.array(nonopt_ps_results["positions"])
        omega_sec_no = np.array(nonopt_ps_results["omega_sec_values"])

    # === Figure 1: Pseudo-potential, gradient, Z minimum ===
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 6), sharex=True)

    # Pseudo-potential
    ax1.plot(x_values, pseudopot_values, color="#003366", label="Optimized")
    if nonopt_ps_results is not None:
        ax1.plot(x_values, pseudopot_values_no, color="#003366", linestyle="--", linewidth=1.2, alpha=0.7, label="Non-optimized")
    ax1.set_ylabel(r"$\Phi_{PS} \ (\mathrm{meV})$")
    ax1.grid(True)
    ax1.yaxis.set_major_formatter(sci_formatter)

    # Gradient
    ax2.plot(x_values, grad_values, color="#8B0000", label="Optimized")
    if nonopt_ps_results is not None:
        ax2.plot(x_values, grad_values_no, color="#8B0000", linestyle="--", linewidth=1.2, alpha=0.7, label="Non-optimized")
    ax2.set_ylabel(r"$\nabla_x \Phi_{PS} \ (\mathrm{meV} / \mathrm{\mu m})$")
    ax2.grid(True)
    ax2.xaxis.set_major_formatter(sci_formatter)
    ax2.yaxis.set_major_formatter(sci_formatter)

    # Z minimum
    ax3.plot(x_values, positions[:, 2], color="#006400", label="Optimized")
    if nonopt_ps_results is not None:
        ax3.plot(x_values, positions_no[:, 2], color="#006400", linestyle="--", linewidth=1.2, alpha=0.7, label="Non-optimized")
    ax3.plot(x_values, [100]*len(x_values), color="black", linestyle="--", linewidth=1.2, alpha=0.8, label="Target ion-surface distance")
    ax3.set_xlabel(r"$x \ (\mathrm{\mu m})$")
    ax3.set_ylabel(r"$h \ (\mathrm{\mu m})$")
    ax3.grid(True)
    ax3.xaxis.set_major_formatter(sci_formatter)
    ax3.yaxis.set_major_formatter(sci_formatter)
    
    if nonopt_ps_results is not None:
        ax1.legend()
        ax2.legend()
        ax3.legend()


    # Shaded regions
    if nonopt_ps_results is None:
        x_shade_start = -41.5
        x_shade_end = 0
        for ax in [ax1, ax2, ax3]:
            ax.axvspan(x_shade_start, x_shade_end, color="#EEE8AC", alpha=0.5)
            ax.axvspan(x_shade_start - 199, x_shade_start, color="#FBC901", alpha=0.5)

    # === Figure 2: Secular frequencies ===
    fig2, ax4 = plt.subplots(figsize=(6, 4))
    colors = {"axial": "#003366", "radial_y": "#006400", "radial_z": "#8B0000"}

    # Optimized
    ax4.plot(x_values, omega_sec[:, 0] / (2*np.pi)/1e6, color=colors["axial"], label=r"$\omega_x$ (axial)")
    ax4.plot(x_values, omega_sec[:, 1] / (2*np.pi)/1e6, color=colors["radial_y"], label=r"$\omega_y$ (radial)")
    ax4.plot(x_values, omega_sec[:, 2] / (2*np.pi)/1e6, color=colors["radial_z"], label=r"$\omega_z$ (radial)")

    # Non-optimized
    if nonopt_ps_results is not None:
        ax4.plot(x_values, omega_sec_no[:, 0] / (2*np.pi)/1e6, color=colors["axial"], linestyle="--", linewidth=1.2, alpha=0.7)
        ax4.plot(x_values, omega_sec_no[:, 1] / (2*np.pi)/1e6, color=colors["radial_y"], linestyle="--", linewidth=1.2, alpha=0.7)
        ax4.plot(x_values, omega_sec_no[:, 2] / (2*np.pi)/1e6, color=colors["radial_z"], linestyle="--", linewidth=1.2, alpha=0.7)

        # === Custom legend handles ===
        line_opt = mlines.Line2D([], [], color='k', linestyle='-', label='Optimized')
        line_nonopt = mlines.Line2D([], [], color='k', linestyle='--', label='Non-optimized')

        # --- Create the two legends ---
        handles, labels = ax4.get_legend_handles_labels()

        # First legend: mode colors
        leg1 = ax4.legend(handles, labels, title="Modes", loc="center left", bbox_to_anchor=(+0.02, 0.65))
        ax4.add_artist(leg1)

        # Second legend: geometry (below the first one)
        leg2 = ax4.legend([line_opt, line_nonopt], ["Optimized", "Non-optimized"],
                        title="Geometry", loc="center left", bbox_to_anchor=(+0.02, 0.30))

        # --- Optional grid & labels ---
        ax4.set_ylabel(r"Frequency $\ (\mathrm{MHz})$")
        ax4.set_xlabel(r"$x \ (\mathrm{\mu m})$")
        ax4.grid(True)

    ax4.set_ylabel(r"Frequency $\ (\mathrm{MHz})$")
    ax4.set_xlabel(r"$x \ (\mathrm{\mu m})$")
    
    if nonopt_ps_results is None:
        ax4.legend(loc="best")
        
    ax4.grid(True)
    ax4.xaxis.set_major_formatter(sci_formatter)
    ax4.yaxis.set_major_formatter(sci_formatter)

    # Shaded regions
    if nonopt_ps_results is None:
        ax4.axvspan(x_shade_start, x_shade_end, color="#EEE8AC", alpha=0.5)
        ax4.axvspan(x_shade_start - 199, x_shade_start, color="#FBC901", alpha=0.5)

    fig1.align_ylabels([ax1, ax2, ax3])
    fig2.align_ylabels([ax1, ax2, ax3])
    plt.tight_layout()
    return fig1, fig2


def plot_sec_freq_comparison(x_values, ps_results, nonopt_ps_results=None):
    # --- Extract frequencies ---
    omega_sec = np.array(ps_results["omega_sec_values"])
    if nonopt_ps_results is not None:
        omega_sec_no = np.array(nonopt_ps_results["omega_sec_values"])

    x_values = np.array(x_values)
    sci_formatter = ScalarFormatter(useMathText=True)
    sci_formatter.set_powerlimits((-3, 3))

    # === Figure setup ===
    fig, ax = plt.subplots(figsize=(6, 4))

    # --- Colors for the three modes ---
    colors = {
        "axial": "#003366",   # dark blue
        "radial_y": "#006400",  # dark green
        "radial_z": "#8B0000"   # deep red
    }

    # --- Optimized ---
    ax.plot(
        x_values,
        omega_sec[:, 0] / (2 * np.pi) / 1e6,
        color=colors["axial"],
        label=r"$\omega_x$ (axial)"
    )
    ax.plot(
        x_values,
        omega_sec[:, 1] / (2 * np.pi) / 1e6,
        color=colors["radial_y"],
        label=r"$\omega_y$ (radial)"
    )
    ax.plot(
        x_values,
        omega_sec[:, 2] / (2 * np.pi) / 1e6,
        color=colors["radial_z"],
        label=r"$\omega_z$ (radial)"
    )

    # --- Non-optimized (if provided) ---
    if nonopt_ps_results is not None:
        ax.plot(
            x_values,
            omega_sec_no[:, 0] / (2 * np.pi) / 1e6,
            color=colors["axial"],
            linestyle="--",
            linewidth=1.2,
            alpha=0.7
        )
        ax.plot(
            x_values,
            omega_sec_no[:, 1] / (2 * np.pi) / 1e6,
            color=colors["radial_y"],
            linestyle="--",
            linewidth=1.2,
            alpha=0.7
        )
        ax.plot(
            x_values,
            omega_sec_no[:, 2] / (2 * np.pi) / 1e6,
            color=colors["radial_z"],
            linestyle="--",
            linewidth=1.2,
            alpha=0.7
        )

    # === Labels, grid, and formatting ===
    ax.set_ylabel(r"Frequency $\ [\mathrm{MHz}]$")
    ax.set_xlabel(r"$x \ [\mathrm{\mu m}]$")
    ax.grid(True)
    ax.xaxis.set_major_formatter(sci_formatter)
    ax.yaxis.set_major_formatter(sci_formatter)

    # === Shaded regions ===
    x_shade_start = -41.5
    x_shade_end = 0
    ax.axvspan(x_shade_start, x_shade_end, color="#EEE8AC", alpha=0.5)
    ax.axvspan(x_shade_start - 199, x_shade_start, color="#FBC901", alpha=0.5)

    return fig



def plot_minima_analysis(pot_simulator: TrapPhysicsSimulator, mode: str = "both"):
    csi_values = pot_simulator.csi_values
    min_pos_0 = pot_simulator.minimum_positions["Up"]
    min_pos_1 = pot_simulator.minimum_positions["Down"]
    secular_freqs = pot_simulator.get_ordered_secular_frequencies(in_mhz=True)

    # --- MINIMA PLOT ---
    fig_min, ax_min = plt.subplots(figsize=(6, 4))

    mask = csi_values > 0.16
    ax_min.plot(csi_values[mask], min_pos_1[:, 2], label='y1', color='orange', linestyle='-')
    ax_min.plot(csi_values[mask], min_pos_1[:, 0], label='x1', color='blue', linestyle='-')

    ax_min.plot(csi_values, min_pos_0[:, 2], label='y0', color='green', linestyle='-.')
    ax_min.plot(csi_values, min_pos_0[:, 0], label='x0', color='red', linestyle='-.')

    ax_min.set_xlabel(r"RF ratio $(\xi_{{RF}})$", fontsize=18)
    ax_min.set_ylabel('Position (μm)', fontsize=18)
    ax_min.set_title('Coordinates of the minima vs RF ratio', fontsize=18)
    ax_min.grid(True)
    ax_min.legend(fontsize=14)
    ax_min.tick_params(axis='both', which='major', labelsize=14)

    fig_min.tight_layout()
    plt.show(block=False)

    # --- FREQUENCIES PLOT ---
    fig_freq, ax_freq = plt.subplots(figsize=(6, 4))

    sec_0, sec_1, sec_2 = secular_freqs[:, 0], secular_freqs[:, 1], secular_freqs[:, 2]

    ax_freq.plot(csi_values, sec_0, label='Radial 1', color='red', linestyle='-')
    ax_freq.plot(csi_values, sec_1, label='Axial', color='green', linestyle='-.')
    ax_freq.plot(csi_values, sec_2, label='Radial 2', color='blue', linestyle='--')

    ax_freq.set_xlabel(r"RF ratio $(\xi_{{RF}})$", fontsize=18)
    ax_freq.set_ylabel('Frequencies (MHz)', fontsize=18)
    ax_freq.set_title('Secular frequencies vs RF ratio', fontsize=18)
    ax_freq.grid(True)
    ax_freq.legend(fontsize=14)
    ax_freq.tick_params(axis='both', which='major', labelsize=14)

    fig_freq.tight_layout()
    plt.show(block=False)


def plot_fields(csi_values, fields, direction):
    """Plot electric field components from inner/outer electrodes as subplots."""
    n_plots = len(fields)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]  # Ensure axes is iterable

    for ax, (key, data_array) in zip(axes, fields.items()):
        if data_array.shape != (len(csi_values), 3):
            print(f"Skipping {key}: unexpected shape {data_array.shape}")
            continue

        for i, label in enumerate(['x', 'z', 'y']):
            ax.plot(csi_values, data_array[:, i], label=label)

            ax.set_ylabel("Intensity (V/m)", fontsize=16)
            ax.set_xlabel(r"RF ratio $(\xi_{{RF}})$", fontsize=16)

            if key == "E_O":
                title = "Outer electrodes field"
            elif key == "E_I":
                title = "Inner electrode field"
            else:
                title = f"Field: {key}"
                ax.set_title(title, fontsize=16)

            ax.grid(True, linestyle='--', alpha=0.6, zorder=0)
            ax.legend(fontsize=12)
            ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
            ax.yaxis.offsetText.set_fontsize(14)
            ax.tick_params(labelsize=14)

    plt.suptitle(f"Field components when moving on the {direction.lower()} minimum", fontsize=18)
    plt.tight_layout()
    plt.show(block=False)


def plot_squared_gradients(csi_values, squared_gradients, direction):
    """Plot spatial components of squared electric field gradients."""

    if direction not in ("Right", "Left"):
        raise ValueError("direction must be either 'Right' or 'Left'")
    
    direction_label = " (L)" if direction == "Left" else " (R)"

    key_to_title = {
        'grad_E_O_squared': r'Gradient of $E_O^2$',
        'grad_E_I_squared': r'Gradient of $E_I^2$',
        'grad_E_O_dot_E_I': r'Gradient of $E_O \cdot E_I$'
    }

    for key in ['grad_E_O_squared', 'grad_E_I_squared', 'grad_E_O_dot_E_I']:
        if key not in squared_gradients:
            print(f"Skipping {key}: not found in dictionary")
            continue

        data_array = squared_gradients[key]
        if data_array.shape != (len(csi_values), 3):
            print(f"Skipping {key}: unexpected shape {data_array.shape}")
            continue

        plt.figure(figsize=(5, 4))

        for i, label in enumerate(['x', 'z', 'y']):
            plt.plot(csi_values, data_array[:, i], label=label)

        plt.title(f"{key_to_title[key]}{direction_label}", fontsize=16)
        plt.xlabel(r"RF ratio $(\xi_{{RF}})$", fontsize=16)
        plt.ylabel(r"Intensity $(V^2/m^3)$", fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.6, zorder=0)
        plt.legend(fontsize=12)

        ax = plt.gca()
        ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
        ax.yaxis.offsetText.set_fontsize(14)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.tight_layout()
        plt.show(block=False)


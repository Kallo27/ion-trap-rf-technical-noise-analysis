#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File: visualize_hr.py
# Created: 10-07-2025
# Author: Lorenzo Calandra Buonaura <lorenzocb01@gmail.com>
# Institution: University of Innsbruck - UIBK
#
# Visualization functions for generating plots and graphics related
# to heating rate measurements and simulations.
#


########################
# IMPORT ZONE          #
########################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib import rcParams

from src.simulators.trap_simulators import TrapPhysicsSimulator
from src.simulators.heating_rate_simulators import HeatingRateSimulator, HeatingRateSimulatorTilted


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

def plot_hr_carrier_data(data_exp, pot_simulator: TrapPhysicsSimulator):
    csi_exp, h_exp, error_h_exp, freq_exp = data_exp
    csi_values = pot_simulator.csi_values
    sec_freqs = pot_simulator.get_ordered_secular_frequencies(in_mhz=True)
    radial_freq = sec_freqs[:, 0]    
    
    plt.figure(figsize=(6, 5))

    # Error bars in front (small errors)
    plt.errorbar(
        csi_exp[:7],
        h_exp[:7],
        yerr=error_h_exp[:7],
        fmt='none',
        ecolor='black',
        capsize=4,
        zorder=3
    )

    # Error bars behind (large errors)
    plt.errorbar(
        csi_exp[7:],
        h_exp[7:],
        yerr=error_h_exp[7:],
        fmt='none',
        ecolor='black',
        capsize=4,
        zorder=1
    )

    # Colored scatter plot
    sc = plt.scatter(
        csi_exp,
        h_exp,
        c=freq_exp,
        cmap='Spectral_r',
        s=60,
        edgecolor='black',
        zorder=2,
        label='Experimental data'
    )

    plt.xlabel(r"RF ratio $(\xi_{{RF}})$", fontsize=18)
    plt.ylabel('Heating rate (ph/s)', fontsize=18)
    plt.title('Carrier data', fontsize=18)
    plt.xlim(-1.1, 1.6)
    plt.grid(True, linestyle='--', alpha=0.6, zorder=0)

    # Add colorbar
    cbar = plt.colorbar(sc)
    cbar.set_label('Secular frequency (MHz)', fontsize=14)

    # Use scientific notation on both axes
    ax = plt.gca()
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(14)

    # Set tick label font sizes
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.yaxis.offsetText.set_fontsize(14)

    # Highlight region where radial1 is minimum
    min_index = np.argmin(radial_freq)
    csi_at_min = csi_values[min_index]
    plt.axvspan(csi_at_min, plt.xlim()[1], color='gray', alpha=0.2, zorder=0)

    plt.tight_layout()
    plt.show(block=False)



def plot_heating_rates_together(heating_rate_simulator: HeatingRateSimulator, data_exp, axes_to_plot=["x", "y"], name="Amplitude", title=None):
    
    csi_values = heating_rate_simulator.csi_values
    heating_rates_dict = heating_rate_simulator.heating_rates_results
    csi_exp, h_exp, error_h_exp, freq_exp = data_exp
    
    # Keys for the three types
    types = ["Outer", "Inner", "Mixed"]

    for heating_type in types:
        fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey='row')

        def make_panel(ax, csi_values, heating_model, fmt):
            ax.plot(csi_values, heating_model, color="#003366", label=f"{name} noise model")
            ax.errorbar(
                csi_exp, h_exp, yerr=error_h_exp, fmt=fmt,
                markersize=6, color='#8B0000',
                ecolor='black', markeredgecolor='black', markeredgewidth=0.5,
                capsize=4, label="Measured HR"
            )
            ax.grid(True, which='both')
            ax.tick_params(axis='both', which='major')

            # Scientific formatting
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((5, 5))
            ax.yaxis.set_major_formatter(formatter)
            ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
            ax.yaxis.offsetText.set_fontsize(14)

            # Add legend
            ax.legend(fontsize=12)

        for row_idx, scale in enumerate(['linear', 'log']):
            for col_idx, axis_label in enumerate(axes_to_plot):
                ax = axs[row_idx, col_idx]
                data = heating_rates_dict[axis_label][heating_type]
                fmt = 's'  # markers per axis   
                make_panel(ax, csi_values, data, fmt)
                ax.set_yscale(scale)
            
            
        axs[0, 0].set_yscale("linear")
        axs[0, 0].set_ylabel("Heating rate (phonon/s)")

        axs[1, 0].set_yscale("log")
        axs[1, 0].set_ylabel("Heating rate (phonon/s)")
        axs[1, 0].set_xlabel(r"RF ratio $(\xi_{\mathrm{RF}})$")

        axs[0, 1].set_yscale("linear")
        
        axs[1, 1].set_yscale("log")
        axs[1, 1].set_xlabel(r"RF ratio $(\xi_{\mathrm{RF}})$")

        # ---------------- Set row-specific y-limits ----------------
        top_ylim = (-0.15e4, 4.15e4)     # linear top row
        bottom_ylim = (1e1, 1e5)         # log bottom row

        axs[0, 0].set_ylim(top_ylim)
        axs[0, 1].set_ylim(top_ylim)

        axs[1, 0].set_ylim(bottom_ylim)
        axs[1, 1].set_ylim(bottom_ylim)
        
        axs[0, 0].set_title(r"X mode ($\dot\bar{n}_x$)")
        axs[0, 1].set_title(r"Y mode ($\dot\bar{n}_y$)")
        for ax in [axs[0, 0], axs[0, 1]]:
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(4, 4))  # force 1e4
            ax.yaxis.offsetText.set_fontsize(14)

        if title:
            fig.suptitle(title, fontsize=20)

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.07, wspace=0.1)
        plt.show(block=False)
        
    return fig



def plot_heating_rates(heating_rate_simulator: HeatingRateSimulator, data_exp, axes_to_plot=["x", "y"], log=False):
    csi_values = heating_rate_simulator.csi_values
    heating_rates_dict = heating_rate_simulator.heating_rates_results
    csi_exp, h_exp, error_h_exp, freq_exp = data_exp
    
    # Keys for the three types
    types = ["Outer", "Inner", "Mixed"]

    for heating_type in types:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
        
        for i, axis_label in enumerate(axes_to_plot):
            heating_rate = heating_rates_dict[axis_label][heating_type]
            
            axs[i].plot(csi_values, heating_rate, label="Simulation", color="darkblue")
            axs[i].errorbar(
                csi_exp, h_exp, yerr=error_h_exp, fmt='o',
                markersize=6, color='tab:red', ecolor='tab:grey',
                capsize=4, label='Data'
            )
            
            axs[i].set_ylabel("Heating rate", fontsize=14)
            axs[i].set_xlabel(r"RF ratio $(\xi_{{RF}})$", fontsize=14)
            axs[i].set_title(f"{axis_label.upper()} axis", fontsize=16)
            axs[i].grid(True)
            axs[i].tick_params(axis='both', which='major', labelsize=12)
            axs[i].legend(fontsize=12)

        # Scientific formatting
        for ax in axs:
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((6, 6))
            ax.yaxis.set_major_formatter(formatter)
            ax.ticklabel_format(style='scientific', axis='y', scilimits=(4, 4))
            ax.yaxis.offsetText.set_fontsize(12)

        if heating_type == "Outer":
            fig.suptitle(f"Noise only on outer electrodes", fontsize=18)
        elif heating_type == "Inner":
            fig.suptitle(f"Noise only on inner electrode", fontsize=18)
        elif heating_type == "Mixed":
            fig.suptitle(f"Noise on both electrodes", fontsize=18)
        
        if log:
            for ax in axs:
                ax.set_yscale("log")
                ax.set_ylim(1e0, 1e6)
        else:
            for ax in axs:
                ax.set_ylim(-1e3, 4.1e4)

        plt.tight_layout()
        plt.subplots_adjust(top=0.88, wspace=0.3)
        plt.show(block=False)
        
        

def plot_heating_rates_tilted(
    heating_rate_simulator: HeatingRateSimulatorTilted,
    data_exp,
    axes_to_plot="x"
):
    csi_values = heating_rate_simulator.csi_values
    heating_rates_dict = heating_rate_simulator.heating_rates_results
    csi_exp, h_exp, error_h_exp, freq_exp = data_exp

    axis_label = axes_to_plot  # single character: "x", "y", or "z"
    types = ["Outer", "Inner", "Mixed"]

    for heating_type in types:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

        # Heating rate from simulation
        heating_rate = heating_rates_dict[axis_label][heating_type]

        # Linear scale plot
        axs[0].plot(csi_values, heating_rate, label="Simulation", color="darkblue")
        axs[0].errorbar(
            csi_exp, h_exp, yerr=error_h_exp, fmt='o',
            markersize=6, color='tab:red', ecolor='tab:grey',
            capsize=4, label='Data'
        )
        axs[0].set_ylabel("Heating rate", fontsize=14)
        axs[0].set_xlabel(r"RF ratio $(\xi_{{RF}})$", fontsize=14)
        axs[0].set_title("Linear scale", fontsize=16)
        axs[0].grid(True)
        axs[0].tick_params(axis='both', which='major', labelsize=12)
        axs[0].legend(fontsize=12)
        axs[0].set_ylim(-1e3, 4.1e4)

        # Log scale plot
        axs[1].plot(csi_values, heating_rate, label="Simulation", color="darkblue")
        axs[1].errorbar(
            csi_exp, h_exp, yerr=error_h_exp, fmt='o',
            markersize=6, color='tab:red', ecolor='tab:grey',
            capsize=4, label='Data'
        )
        axs[1].set_ylabel("Heating rate", fontsize=14)
        axs[1].set_xlabel(r"RF ratio $(\xi_{{RF}})$", fontsize=14)
        axs[1].set_title("Log scale", fontsize=16)
        axs[1].grid(True)
        axs[1].tick_params(axis='both', which='major', labelsize=12)
        axs[1].legend(fontsize=12)

        # Scientific formatting
        for ax in axs:
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((6, 6))
            ax.yaxis.set_major_formatter(formatter)
            ax.ticklabel_format(style='scientific', axis='y', scilimits=(4, 4))
            ax.yaxis.offsetText.set_fontsize(12)
            
        axs[1].set_yscale("log")
        axs[1].set_ylim(1e0, 1e6)

        # Title
        if heating_type == "Outer":
            prefix = "Noise only on outer electrodes"
        elif heating_type == "Inner":
            prefix = "Noise only on inner electrode"
        else:
            prefix = "Noise on both electrodes"
            
        fig.suptitle(
            f"Tilted gradients ({axis_label.upper()} axis): {prefix}\n"
            f"Angles: {np.rad2deg(heating_rate_simulator.angles_rad[0]):.2f}°, {np.rad2deg(heating_rate_simulator.angles_rad[1]):.2f}°",
            fontsize=18
        )
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.83, wspace=0.3)
        plt.show(block=False)
        
        
def plot_heating_rates_junction(x_values, hr_values, ps_results, indices=None):
    x_values = np.array(x_values)
    hr_values = np.array(hr_values)
    pseudopot_values = np.array(ps_results["pseudopot_values"]) / 1e-3  # in mV

    # --- Apply indices selection ---
    if indices is not None:
        x_values = x_values[indices]
        hr_values = hr_values[indices, :]
        pseudopot_values = pseudopot_values[indices]

    # === Figure setup ===
    sci_formatter = ScalarFormatter(useMathText=True)
    sci_formatter.set_powerlimits((-3, 3))
    sci_formatter.set_useOffset(False)
    sci_formatter.set_scientific(True)

    sci_formatter_2 = ScalarFormatter(useMathText=True)
    sci_formatter_2.set_powerlimits((-3, 3))
    sci_formatter_2.set_useOffset(False)
    sci_formatter_2.set_scientific(True)

    fig, ax1 = plt.subplots(figsize=(6, 4))

    # --- Right y-axis: pseudopotential (drawn first, so it's behind) ---
    ax2 = ax1.twinx()
    ax2.fill_between(
        x_values, pseudopot_values,
        color="#FBC901", alpha=0.3,
        label="Pseudopotential", zorder=1
    )
    ax2.set_ylabel(r"$\Phi_{PS}$ (meV)", labelpad=10, rotation=270, va='bottom')
    ax2.yaxis.set_major_formatter(sci_formatter_2)

    # --- Left y-axis: heating rates ---
    ax1.plot(x_values, hr_values[:, 0], color="#8B0000", label=r"$\dot{\bar{n}}_x$", zorder=2.5)
    ax1.plot(x_values, hr_values[:, 1], color="#003366", label=r"$\dot{\bar{n}}_y$", zorder=2.5)
    ax1.plot(x_values, hr_values[:, 2], color="#006400", label=r"$\dot{\bar{n}}_z$", zorder=2.5)
    ax1.set_ylabel(r"$\dot{\bar{n}}$ / $S_{V_N}(\Omega_{\mathrm{RF}} + \omega_x) \ \ $ (phonon/s) / ($\mathrm{V}^2$/Hz)", labelpad=10)
    ax1.set_xlabel(r"$x \ (\mathrm{\mu m})$")
    ax1.set_ylim(-500, 20500)
    ax1.xaxis.set_major_formatter(sci_formatter)
    ax1.yaxis.set_major_formatter(sci_formatter)

    # === Align zeros visually ===
    y1_min, y1_max = ax1.get_ylim()
    y2_min, y2_max = pseudopot_values.min(), pseudopot_values.max()
    scale = (y1_max - y1_min) / (y2_max - y2_min)
    shift = y1_min / scale - y2_min
    ax2.set_ylim(y2_min + shift, y2_max - shift)

    # --- Combine legends ---
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    
    return fig


def plot_heating_rates_junction_2(x_values, spline_hr, piecewise_hr, nonopt_hr, indices=None):
    spline_hr, piecewise_hr, nonopt_hr
    x_values = np.array(x_values)
    sci_formatter = ScalarFormatter(useMathText=True)
    sci_formatter.set_powerlimits((-3, 3))

    # --- Apply indices selection ---
    if indices is not None:
        x_values = x_values[indices]
        spline_hr = spline_hr[indices, :]
        piecewise_hr = piecewise_hr[indices, :]
        nonopt_hr = nonopt_hr[indices, :]
    
    # --- Select only x components ---
    spline_hr_x = spline_hr[:, 0]
    piecewise_hr_x = piecewise_hr[:, 0]
    nonopt_hr_x = nonopt_hr[:, 0]
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

    # Piecewise parametrization
    ax1.plot(x_values, piecewise_hr_x, color="#003366", label="Piecewise parametrization")
    ax1.plot(x_values, nonopt_hr_x, color="#2C2C2C", linestyle="--", linewidth=1.2, alpha=0.7, label="Non-optimized")
    ax1.set_ylabel(r"$\dot{\bar{n}}$ / $S_{V_N}(\Omega_{\mathrm{RF}} + \omega_x)$")
    ax1.grid(True)
    ax1.legend(loc="upper left")
    ax1.yaxis.set_major_formatter(sci_formatter)

    # Spline parametrization
    ax2.plot(x_values, spline_hr_x, color="#006400", label="Spline parametrization")
    ax2.plot(x_values, nonopt_hr_x, color="#2C2C2C", linestyle="--", linewidth=1.2, alpha=0.7, label="Non-optimized")
    ax2.set_xlabel(r"$x \ (\mathrm{\mu m})$")
    ax2.set_ylabel(r"$\dot{\bar{n}}$ / $S_{V_N}(\Omega_{\mathrm{RF}} + \omega_x)$")
    ax2.grid(True)
    ax2.legend(loc="upper left")
    ax2.xaxis.set_major_formatter(sci_formatter)
    ax2.yaxis.set_major_formatter(sci_formatter)

    return fig
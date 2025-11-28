# RF technical noise analysis in trapped-ion quantum computing architectures

This repository contains the code developed for the Master’s thesis **"Trapped-ion quantum architectures: RF electrode optimization and effects of RF technical noise"**, completed as part of the Physics of Data program at the University of Padua.The work was carried out in collaboration with the Cryo team of the Quantum Optics and Spectroscopy group at the University of Innsbruck. The code implements theoretical models and simulations to study the effects of radio-frequency (RF) technical noise on the motional heating of trapped ions in surface-electrode Paul traps. It also provides tools for electrode layout design and optimization.

---

## Overview

Ion traps are among the most promising platforms for quantum computing due to their high gate fidelities and long coherence times. Scaling trapped-ion processors using surface-electrode Paul traps requires minimizing the ion–surface distance, which makes ions more susceptible to technical noise from trap electrodes. While DC voltage noise has been extensively studied, RF technical noise remains less explored.

This repository implements the theoretical framework developed in the thesis to characterize how RF amplitude and phase noise affects ion secular motion, leading to heating. The code can be used to:

- Model RF phase and amplitude noise in terms of ion heating.
- Simulate heating rates for different trap geometries and shuttling protocols.
- Optimize RF electrode edges in X-junction and other trap designs to reduce heating.
- Compare simulation results with experimental data.

The simulations use **electrode modeling** and the **gapless plane approximation**.

---

## Repository Structure

- `src/` contains all scripts and modules implementing the models and simulations.
- `gds_files/` contains the GDS files of the trap modules studied in the thesis.
- `optimization_results/` contains the optimized parameters for the X-junction geometries at varying ion--surface distance.
- `notebooks/` contains example notebooks demonstrating how to use the code; these are unpolished and intended for reference.

---

## Getting Started

### Requirements

- Python 3.9+ 
- `numpy`, `scipy`, `matplotlib`
- `electrode`, needed for electrode calculations
- Other packages for visualization or additional analysis (check individual scripts for specifics)

### Running the Code

The scripts in `src/` can be run as modules. For example:

```bash
python3 -m src.junction_hr
python3 -m src.shuttling_simulation
````

This executes the main simulations for heating rate calculations and shuttling.

### Using the Notebooks

The `notebooks/` folder contains examples demonstrating how to use the code. These notebooks are **unpolished** and intended as reference rather than finalized analysis. Some simulations included here were not included in the thesis, as they were not studied in sufficient detail.

---

## License

This repository is released under the [MIT License](LICENSE).

---

## Citation

If you use this code in your work, please cite the thesis and this repository:

```
Lorenzo Calandra Buonaura, RF technical noise analysis in trapped-ion quantum computing architectures, GitHub repository, 2025.
```
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.17753565-blue)](https://doi.org/10.5281/zenodo.17753565)

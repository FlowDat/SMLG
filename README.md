## NOTE
**Before running "python run_analysis1.py" copy the contents from 10.5281/zenodo.21234013 directly into the data directory**

## Data Description

This repository contains the datasets accompanying the manuscript:

**SPECTRE: A scale-aware deep-learning model for accelerated turbulence emulation**

Syed M. Usama, Nadeem A. Malik, Umair Umer, Amjad Shaikh, Md Zishan Akhter, Syed Adnan Ali, Zhigang Sun

## Data Files

The dataset includes flow field data from direct numerical simulations of turbulent mixing layers at seven Reynolds numbers.

| File | Description |
|------|-------------|
| Re2500.npz | Reynolds number 2500 |
| Re5000.npz | Reynolds number 5000 |
| Re10000.npz | Reynolds number 10000 |
| Re15000.npz | Reynolds number 15000 |
| Re20000.npz | Reynolds number 20000 |
| Re25000.npz | Reynolds number 25000 |
| Re30000.npz | Reynolds number 30000 |
| benchmark_data.npz | Benchmark comparison results |

## Data Variables

Each Reynolds number file contains the following variables:

- coordinates: Spatial coordinates of the flow field
- variables: List of variable names (U:0, U:1, p, s)
- original_data: Original simulation data
- reconstructed_data: SPECTRE predicted data
- mean_field: Time-averaged mean field
- temporal_coefficients: Temporal coefficients from reduced-order model
- time: Time array

### Variable Definitions

- U:0 - Streamwise velocity component
- U:1 - Wall-normal velocity component
- p - Pressure field
- s - Scalar concentration field

## Visualization

A Python script is provided for data visualization and analysis:

- run_analysis.py

This script generates all figures presented in the manuscript, including:

- Global reconstruction error vs Reynolds number
- Diagnostic dashboards
- Spectral and temporal analysis
- Mean flow profiles
- Turbulent intensities
- Transport and stresses
- Coherent structures
- Layer thickness evolution
- Model validation metrics
- Mixing performance analysis

## Dependencies

The visualization code requires the following Python packages:

- Python 3.8 or higher
- numpy >= 1.21.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0

## Usage

To reproduce the figures:

```bash
python run_analysis.py
```

The script will generate all figures in the analysis_output directory.

## Citation

If you use this dataset in your research, please cite:

```
Usama, S.M., Malik, N.A., Umer, U., Shaikh, A., Akhter, M.Z., Ali, S.A., Sun, Z.
SPECTRE: A scale-aware deep-learning model for accelerated turbulence emulation.
Scientific Reports (2026).
```

## Contact

Syed M. Usama: usama@dicp.ac.cn

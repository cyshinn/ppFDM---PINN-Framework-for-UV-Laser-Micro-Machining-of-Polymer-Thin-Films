# ppFDM — PINN Framework for UV Laser Micro-Machining of Polymer Thin Films

Code and reference data supporting the manuscript: 
A Physics-Informed Neural Network Framework for Surrogate Modeling, Inverse Design, and Inverse PINN-Based Material Identification in UV Laser Micro-Machining of Polymer Thin Films 
(submitted)

The framework couples a finite-difference (FDM) thermal solver with a physics-informed neural network (PINN) for (i) temperature-field surrogate modeling, (ii) reference-curve inverse design of pulse energy for target crater diameters, and (iii) inverse identification of temperature-dependent thermal conductivity k(T) with independent cross-code verification, for polypropylene (PP) and polyethylene (PE) films.

This repository contains the computational source code only. 

## Repository structure
```
.
├── README.md
├── requirements.txt
├── references.bib
└── code/
    ├── export_training_data.py   # Explicit FDM solver — generates PP/PE thermal snapshots
    ├── train_pinn.py             # Full PINN (8×64 tanh) — paper baseline + ablation surrogates
    ├── train_fast.py             # Fast/prototype training variant (reduced epochs & collocation)
    ├── cn_fdm_reference.py       # Independent Crank–Nicolson FDM solver (cross-code check)
    ├── inverse_pinn_kT.py        # Inverse PINN — recovers k(T) jointly with the T-field
    ├── wu_comparison.py          # Direct FDM comparison with Wu et al. (2022) laser parameters
    ├── run_experiments.py        # Multi-seed reproducibility / loss-weight sweep driver
    └── supplementary_analyses.py # Diagnostics behind SI §1.4, §12, §13 (architecture/seed/loss sweeps)
```

## Requirements
```bash
pip install -r requirements.txt
```
Dependencies: `numpy`, `pandas`, `scipy`, `matplotlib`, `torch>=2.0`, `scikit-learn`.
All scripts run on CPU with PyTorch 2.x. Random seeds are fixed (`seed = 42`) for reproducibility.

## Reproducing the results (run order)

The scripts have a dependency order: the FDM reference data must be generated first, because the PINN training, inverse identification, and diagnostics all read from it.

```bash
# 1. Generate FDM reference data (PP & PE thermal snapshots, multi-Ep ablation curves)
#    Writes: fdm_output/{PP,PE}_snapshots.npz, {PP,PE}_multiEp.csv      (~3–8 min, CPU)
python code/export_training_data.py

# 2. Train the temperature surrogate PINN + ablation-diameter surrogates
#    Writes: fdm_output/pinn_results.npz                                (~6–12 min, CPU)
python code/train_pinn.py
#    (optional fast prototype, same architecture, fewer epochs/collocation points)
# python code/train_fast.py

# 3. Inverse PINN k(T) identification, then independent cross-code verification
#    Writes: fdm_output/inverse_kT_results.npz, cn_*.npy
python code/inverse_pinn_kT.py
python code/cn_fdm_reference.py

# 4. Experimental-parameter comparison (Wu et al. 2022)
#    Writes: fdm_output/wu_comparison.csv
python code/wu_comparison.py

# 5. Supplementary diagnostics — architecture / 10-seed / loss-weight sweeps (long; CPU)
#    Writes: fdm_output/supplementary_results.json
python code/supplementary_analyses.py

# (optional) Multi-seed reproducibility / loss-weight sweep driver
#    Writes: fdm_output/experiment_results.json
python code/run_experiments.py
```

## Notes
# `train_pinn.py` vs `train_fast.py`
`train_pinn.py` is the paper baseline: 8 hidden layers × 64 neurons (tanh), Xavier-normal initialization (gain 0.7), initial learning rate 1.5×10⁻³, gradient clipping 1.5, 6,000 epochs, 8,000 collocation points. `train_fast.py`shares the same architecture and hyperparameters but uses 3,000 epochs and 2,000 collocation points for quick checks — use it for prototyping, not for paper-exact results.

# Cross-code verification
`cn_fdm_reference.py` implements an implicit Crank–Nicolson solver with harmonic-mean interface conductivity, deliberately different in numerical structure from the explicit forward-Euler generator in `export_training_data.py`. 
Running it on the inverse-recovered *k\*(T)* provides an independent check that is not circular with the training-data solver.

# Material properties
PP and PE thermal/mechanical property data are drawn from the Polymer Handbook and van Krevelen; 355 nm absorption coefficients from Serafetinides et al. 
See `references.bib`.

## Citation
Please cite the manuscript above if you use this code.

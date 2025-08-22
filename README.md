# Modelling lung health with CF home-monitoring data

This repository contains code for Bayesian network modeling and inference, with a focus on data analysis of CF home-monitoring data (e.g., O2 saturation, FEV1, etc.). The codebase includes model construction, CPT calculation, inference routines, and plotting utilities, etc

## Features

- Construction of Bayesian networks for medical variables
- Efficient CPT (Conditional Probability Table) calculation
- Inference using [pgmpy](https://pgmpy.org/)
- Visualization of results using Plotly
- Profiling and optimization scripts

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd phd
   ```

2. **Create and activate a Python environment**
   ```bash
   conda create -n phd python=3.10
   conda activate phd
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *Typical dependencies include:*
   - `pgmpy`
   - `numpy`
   - `scipy`
   - `plotly`
   - `matplotlib` (optional, for some plots)

4. **(Optional) Install Jupyter for interactive analysis**
   ```bash
   pip install jupyter
   ```

## Running Analysis

1. **Run the main analysis script**
   ```bash
   python src/app_optimisation/app_as_script.py
   ```
   This will perform model construction, inference, and generate plots.

2. **Profiling (Performance Analysis)**
   To profile the code and analyze bottlenecks:
   ```bash
   python -m cProfile -o output.prof src/app_optimisation/app_as_script.py
   ```
   Then use tools like `snakeviz` or `gprof2dot` to visualize the profile.

3. **Custom Analysis**
   - Modify or add scripts in `src/` for custom models or experiments.
   - Use Jupyter notebooks for interactive exploration.

## Repository Structure

The repository is organized as follows:

- `src/`  
  Main source code directory containing all modules and scripts.
  - `app/`
    Contains the code used to develop the web platform.
  - `app_optimisation/`  
    Contains the main analysis script (`app_as_script.py`) and related optimization code.
  - `data/`  
    Contains the data processing pipeline and related data exploration.
  - `inf_cutset_conditioning/`  
    Modules to perform exact inference on loopy graphs with the cutset conditioning approach. This contains the scripts to perform longitudinal inference.
  - `inference/`  
    Helper functions and utilities for performing inference.
  - `model_validation/`  
    Early development of modules for model validation.
  - `modelling firepower/`  
    Preliminary work and exercises on statistical inference methods (Belief Propagation, Gibbs Sampling)
  - `modelling_ar/`  
    Modules focused on modeling airway resistance (AR) and related physiological processes.
  - `modelling_fef2575/`  
    Modules for modeling the forced expiratory flow at 25-75% of pulmonary volume (FEF25-75), including relevant computations and helper functions.
  - `modelling_fev1/`  
    Modules for modeling forced expiratory volume in 1 second (FEV1), including variable construction and associated helper functions.
  - `modelling_o2/`  
    Specialized modules for O2 saturation (O2Sat) modeling.
  - `models/`  
    Modules for model definitions and general helper functions.
  - `o2_fev1_analysis/`  
    Contains the analysis of the relationship O2 saturation and FEV1 signals.
  - `tests/`  
    Test suite for the validation of the inference infrastructure, mainly cutset conditioning.
  - `tromso/`  
    Replication of the thesis results on an external dataset from the Tromso study in Norway.
  - `viz/`  
    Notebooks to generate the final thesis visualisations, mainly for the longitudinal model.
- `README.md`  
  This file. Provides setup, usage, and documentation.
- `requirements.txt`  
  List of Python dependencies required to run the project.

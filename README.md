# Simulation-Based Digital Twin of a PMSM Drive for Reliability-Aware Fault Diagnosis

This repository contains the simulation and analysis code for a PMSM digital-twin fault diagnosis framework. The project is fully simulation-based and focuses on how plant--twin residuals can be used to support fault diagnosis in a closed-loop motor drive.

The main setup is simple: a PMSM plant and a nominal digital twin are run in parallel under the same control inputs. Faults are applied only to the plant, while the twin stays at its healthy nominal parameters. The difference between the plant and the twin is saved as residual signals, and those residuals are then used for feature extraction, machine learning classification, uncertainty estimation, and false-alarm-controlled decision logic.

No hardware interface is required.

---

## Overview

The framework was developed around a closed-loop PMSM drive, but the workflow is based on a more general digital-twin residual idea. A nominal model is used as a reference, and deviations from that reference are analyzed to support reliable diagnosis.

The repository includes:

- closed-loop PMSM simulation in the d-q reference frame
- nominal digital twin running in parallel with the plant
- stator resistance drift (`rs_drift`) fault simulation
- d-axis inductance mismatch (`ld_mismatch`) fault simulation
- residual-based steady-state and transient feature extraction
- physics-guided diagnostic features
- Random Forest baseline and physics-guided PIML-style classification
- split conformal prediction for uncertainty-aware outputs
- residual-energy and transient feature gates for false-alarm control
- severity and detectability analysis
- closed-loop masking analysis
- unseen-speed domain generalization test
- multi-split robustness validation
- article-ready tables and journal-style figures

---

## Repository Structure

The repository is organized around four main parts:

- `main.py` generates the PMSM digital-twin simulation dataset.
- `controllers/`, `models/`, and `utils/` contain the reusable simulation, motor model, controller, and residual-gate components.
- `evaluate_*.py` and `analyze_*.py` scripts run the main experiments and validation analyses.
- `paper_results/` contains the tables, figures, reports, and metadata used for the written report/article.

The final dataset used in the reported experiments is stored under `data/`, with the matching run-level signal files in `data/runs/`.

---

## Main Code Files

### `main.py`

Generates the simulation dataset. It runs the faulty PMSM plant and the nominal digital twin in parallel, saves the signal-level run files, and extracts residual-based diagnostic features.

The generated files are stored under:

```text
data/runs/
data/runs_index_<timestamp>.csv
data/features_<timestamp>.csv
```

The final experiments in this repository use:

```text
data/features_20260301_204447.csv
data/runs_index_20260301_204447.csv
```

### `models/pmsm.py`

Contains the PMSM d-q model used for both the plant and the nominal twin.

### `controllers/pi.py`

Contains the PI controller used in the speed and current control loops.

### `utils/energy_gate.py`

Implements the residual-energy gate used in the false-alarm-controlled decision layer. It uses healthy-run calibration, EWMA smoothing, hysteresis, and persistence counters.

### `utils/transient_feature_gate.py`

Implements a short-window transient residual gate based on the log-energy of `r_id` and `r_iq` after the fault activation time.

---

## Evaluation Scripts

### `evaluate_article_framework_v3.py`

Runs the main reliability-aware evaluation. It compares:

1. ML baseline
2. physics-guided PIML-style classifier
3. PIML + conformal prediction
4. full framework with residual evidence and false-alarm control

In the full framework, uncertain fault cases are not counted as healthy missed detections. They are reported separately as uncertain/inspect cases.

### `evaluate_threshold_sensitivity.py`

Tests how the confidence threshold affects false alarm rate, missed detection rate, uncertain/inspect rate, hard fault detection rate, accepted accuracy, and safe decision rate.

### `analyze_severity_detectability.py`

Analyzes how fault severity affects hard detection, uncertain/inspect decisions, and missed cases.

### `analyze_masking_effect.py`

Studies closed-loop masking by comparing output-level speed tracking error with internal residual and control-domain indicators.

### `evaluate_domain_generalization.py`

Tests whether the framework generalizes to an unseen operating speed. The model is trained and calibrated on seen speed conditions and evaluated on an unseen speed condition.

### `evaluate_piml_physics_loss.py`

Compares a Random Forest physics-guided baseline, a standard neural classifier, and a physics-informed neural classifier with a residual-consistency loss term.

### `evaluate_multisplit_robustness.py`

Repeats the reliability-aware evaluation over multiple train/calibration/test splits to check whether the main behavior remains stable.

---

## Table and Figure Scripts

### `make_article_summary_tables.py`

Creates the main article-ready tables from the reliability framework, detectability analysis, masking analysis, and decision breakdown outputs.

### `make_advanced_validation_tables.py`

Creates article-ready tables for threshold sensitivity, domain generalization, and the physics-informed loss experiment.

### `make_paper_ready_figures_v2.py`

Creates journal-style PNG and PDF figures from the processed table outputs.

---

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Running the Code

### 1. Generate a new dataset

```bash
python main.py
```

This creates a new timestamped dataset under `data/`.

### 2. Run the main framework evaluation

```bash
python evaluate_article_framework_v3.py
```

### 3. Run the additional validation analyses

```bash
python evaluate_threshold_sensitivity.py
python analyze_severity_detectability.py
python analyze_masking_effect.py
python evaluate_domain_generalization.py
python evaluate_piml_physics_loss.py
python evaluate_multisplit_robustness.py
```

### 4. Generate article-ready tables and figures

```bash
python make_article_summary_tables.py
python make_advanced_validation_tables.py
python make_paper_ready_figures_v2.py
```

---

## Output Locations

Tables are saved in:

```text
paper_results/tables/
```

Figures are saved in:

```text
paper_results/figures/
```

Detailed decision reports and metadata are saved in:

```text
paper_results/reports/
```

---

## Fault Types

| Fault Type | Description |
|---|---|
| `healthy` | Nominal operation |
| `rs_drift` | Stator resistance drift |
| `ld_mismatch` | d-axis inductance mismatch |

The dataset includes multiple operating speeds, load profiles, random seeds, and fault severity levels.

---

## Methodological Note

Although the simulation model is a PMSM drive, the methodology is not limited to PMSM systems. The core idea is to run a nominal model in parallel with the controlled plant and use the residual between measured and expected behavior for reliability-aware diagnosis.

This type of residual-based digital-twin structure can also be adapted to other robotic motor systems, including electric actuators in manipulators, mobile robots, and servo-driven mechatronic platforms, as long as a nominal dynamic model and meaningful residual signals are available.

---

## Reproducibility

The reported results are based on the final dataset files:

```text
data/features_20260301_204447.csv
data/runs_index_20260301_204447.csv
```

For scripts that use run-level signals, the matching folders under `data/runs/` should also be included.

If the run-level signal files are not included, they can be regenerated with:

```bash
python main.py
```

Regenerated datasets will receive new timestamped file names. To reproduce the exact reported results, keep the final timestamped CSV files and their matching run folders together.

---

## License

This project is released under the MIT License. See the `LICENSE` file for details.

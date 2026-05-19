# MFMC Reproducibility Package

This repository contains the reproducible code package for the MFMC method. It includes preprocessing scripts, main experiment notebooks, supplementary analyses, dependency specifications, and example commands.

Raw and processed DEAP, CEAP, and MAHNOB-HCI data are not distributed in this repository because the original datasets may be subject to redistribution restrictions. Users should obtain the datasets from the official sources, place them in the directory structure described below, and run the provided preprocessing scripts to regenerate the processed arrays.

## Repository Structure

```text
.
├── DEAP/
│   ├── DEAP_Preprocess.py
│   ├── DEAP_data/                  # user-provided raw DEAP .dat files, not tracked
│   ├── Data_processed/             # generated .npy files, not tracked
│   ├── MFMC_Fusion_MLP/            # MLP-fusion MFMC notebooks
│   ├── MFMC_Fusion_Attention/      # attention-fusion MFMC notebooks
│   └── MFMC_Fusion_Attention_v2/   # additional attention-fusion runs
├── CEAP/
│   ├── CEAP_Preprocess.py
│   ├── CEAP/                       # user-provided raw CEAP files, not tracked
│   ├── CEAP_Processed/             # generated .npy files, not tracked
│   ├── MFMC_Fusion_MLP/
│   ├── MFMC_Fusion_Attention/
│   ├── MFMC_Fusion_Attention_v2/
│   └── MFMC_InfoNCE_LogDet_Comparison/
├── HCI/
│   ├── HCI_Preprocess.py
│   ├── HCI/                        # user-provided raw MAHNOB-HCI sessions, not tracked
│   ├── Data_processed/             # generated .npy files, not tracked
│   └── MFMC_Fusion_MLP/            # MLP-fusion MFMC notebooks
└── Supplement/
    ├── Dependence_Contribution_Analysis/
    ├── Preprocess_sensitivity/
    ├── Fusion_Ablation/
    ├── DTC_ablation/
    ├── epsilon_sensitivity_trace_fmca/
    ├── scalability_experiment/
    └── Sgnal_selection/
```

## Environment

The experiments were run with Python 3.10 and PyTorch/CUDA. To recreate the environment:

```bash
conda env create -f environment.yml
conda activate MFMC
python -m ipykernel install --user --name MFMC --display-name "Python (MFMC)"
```

The pinned package versions include Python 3.10.20, PyTorch 2.6.0+cu124, NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, SciPy 1.15.3, Matplotlib 3.10.9, MNE 1.12.1, seaborn 0.13.2, tqdm 4.67.3, IPython 8.39.0, and ipykernel 7.2.0.

If you prefer pip-style installation, you can also use:

```bash
pip install -r requirements.txt
```

## Path Setup

The notebooks use `TAFFC_MFMC_ROOT/MFMC/...` as the project layout. If this repository is cloned as the standalone `MFMC` folder, run the following commands from the repository root before preprocessing or opening notebooks:

```bash
export TAFFC_MFMC_ROOT="$(dirname "$PWD")"
export DEAP_BASE_PATH="$PWD/DEAP"
export CEAP_BASE_PATH="$PWD/CEAP"
export HCI_BASE_PATH="$PWD/HCI"
```

If raw datasets are stored outside the repository, also set the raw-data variables shown in the Notes section.

## Data Preparation

### DEAP

Place the official DEAP `.dat` files in:

```text
DEAP/DEAP_data/
```

The expected file names are `s01.dat` to `s32.dat`. Then run:

```bash
python DEAP/DEAP_Preprocess.py
```

This generates `DEAP/Data_processed/` with windowed signals, labels, subject IDs, and channel metadata. The main DEAP MFMC notebooks use EEG, EOG, and temperature/SKT inputs. The preprocessing script also exposes additional peripheral modalities for supplementary analyses.

### CEAP

Place the official CEAP files in:

```text
CEAP/CEAP/
```

The preprocessing script expects the CEAP frame-level folders `Annotation_Frame/` and `Physio_Frame/` under this directory. Then run:

```bash
python CEAP/CEAP_Preprocess.py
```

This generates `CEAP/CEAP_Processed/` with ACC, BVP, EDA, SKT, HR, IBI, labels, subject IDs, and video IDs.

### MAHNOB-HCI

Place the official MAHNOB-HCI session folders in:

```text
HCI/HCI/
```

Then run:

```bash
python HCI/HCI_Preprocess.py
```

This generates `HCI/Data_processed/`.

## Example Runs

After preprocessing, run the notebooks from the repository root. The main reproducibility entry points are:

```text
DEAP/MFMC_Fusion_MLP/DEAP_MFMC_Fusion_MLP_subject_dep.ipynb
DEAP/MFMC_Fusion_MLP/DEAP_MFMC_Fusion_MLP_subject_indep.ipynb
DEAP/MFMC_Fusion_Attention/DEAP_MFMC_Fusion_Attention_subject_dep.ipynb
DEAP/MFMC_Fusion_Attention/DEAP_MFMC_Fusion_Attention_subject_indep.ipynb
CEAP/MFMC_Fusion_MLP/CEAP_MFMC_Fusion_MLP_subject_dep.ipynb
CEAP/MFMC_Fusion_MLP/CEAP_MFMC_Fusion_MLP_subject_indep.ipynb
CEAP/MFMC_Fusion_Attention/CEAP_MFMC_Fusion_Attention_subject_dep.ipynb
CEAP/MFMC_Fusion_Attention/CEAP_MFMC_Fusion_Attention_subject_indep.ipynb
HCI/MFMC_Fusion_MLP/HCI_MFMC_Fusion_MLP_subject_dep.ipynb
HCI/MFMC_Fusion_MLP/HCI_MFMC_Fusion_MLP_subject_indep.ipynb
HCI/MFMC_Fusion_MLP/HCI_MFMC_Fusion_MLP_subject_dep_one_fold_test.ipynb
```

For command-line execution with Jupyter installed:

```bash
jupyter nbconvert --to notebook --execute \
  DEAP/MFMC_Fusion_MLP/DEAP_MFMC_Fusion_MLP_subject_dep.ipynb \
  --output executed_DEAP_MFMC_Fusion_MLP_subject_dep.ipynb
```

The notebooks use fixed random seeds where applicable and save fold-level outputs under their local result folders.

## Supplementary Analyses

The `Supplement/` directory contains the code used for additional analyses reported in the supplementary material:

- `Preprocess_sensitivity/`: DEAP preprocessing parameter sensitivity.
- `Dependence_Contribution_Analysis/`: dependence contribution analysis for DEAP and CEAP.
- `Fusion_Ablation/`: MLP fusion vs attention fusion figure generation.
- `DTC_ablation/`: dependence term contribution ablation.
- `epsilon_sensitivity_trace_fmca/`: trace-FMCA epsilon sensitivity.
- `scalability_experiment/`: MFMC scalability experiments with multiple modality settings.
- `Sgnal_selection/`: learnable DEAP modality-selection demonstration.

## Notes

The repository intentionally excludes raw datasets and generated processed arrays. If paths differ from the default layout, set the corresponding environment variables before preprocessing:

```bash
export DEAP_RAW_DATA_DIR=/path/to/DEAP_data
export DEAP_PROCESSED_DIR=/path/to/Data_processed
export CEAP_RAW_DATA_DIR=/path/to/CEAP
export CEAP_PROCESSED_DIR=/path/to/CEAP_Processed
export HCI_SESSIONS_PATH=/path/to/HCI
export HCI_PROCESSED_DIR=/path/to/Data_processed
```

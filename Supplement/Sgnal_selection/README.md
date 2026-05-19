# DEAP Signal Selection Supplement

This folder contains a DEAP-only supplementary notebook for a learnable modality selection demo. It trains a supervised classifier over all DEAP physiological modality groups and learns one global softmax attention weight per modality.

The notebook expects processed files in:

`/home/zhengdeyang/TAFFC_MFMC/MFMC/DEAP/Data_processed`

Run `/home/zhengdeyang/TAFFC_MFMC/MFMC/DEAP/DEAP_Preprocess.py` first if `emg_data.npy` or `pleth_data.npy` is missing. The preprocessing script skips existing outputs and only adds missing modality files.

`signal_selection.ipynb` displays learned attention weights, top-3 retained modalities, and figures inline. It does not save local outputs, figures, CSV files, or checkpoints.

This is a supervised DEAP modality-selection demo using all DEAP modalities. It is not the main MFMC tri-modal self-supervised learning experiment.

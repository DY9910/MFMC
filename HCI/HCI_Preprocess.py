# MAHNOB_Preprocess.py
#
# This script preprocesses the Mahnob-HCI dataset, inspired by the
# provided scripts for DEAP and CEAP. It performs the following steps:
# 1. Locates all raw BDF and XML files.
# 2. Reads physiological signals and emotion labels for each trial.
# 3. Applies a sliding window to segment the data.
# 4. Separates data into different modalities (EEG, ECG, GSR, etc.).
# 5. Normalizes the data and filters outliers.
# 6. Saves the processed data into separate .npy files for each modality.
#
# Before running, ensure you have the required libraries:
# pip install mne numpy torch

import os
import xml.etree.ElementTree as ET
import mne
import torch
import numpy as np
import re
from tqdm import tqdm

# --- Configuration ---
def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

# Define base paths.
# Reproducibility note: keep code paths relative to this HCI folder by default.
# Set HCI_SESSIONS_PATH if the raw MAHNOB-HCI sessions live outside this repository.
BASE_PATH = os.environ.get('HCI_BASE_PATH', '/home/zhengdeyang/TAFFC_MFMC/MFMC/HCI')
SESSIONS_PATH = os.environ.get('HCI_SESSIONS_PATH', f'{BASE_PATH}/HCI')
DATA_DIR = os.environ.get('HCI_PROCESSED_DIR', f'{BASE_PATH}/Data_processed')

create_folder_if_not_exists(DATA_DIR)

# --- Windowing Parameters ---
SAMPLE_RATE = 256  # Hz, confirmed from data inspection
WINDOW_SIZE_SECONDS = 10  # 10 seconds
WINDOW_STRIDE_SECONDS = 0.5  # 0.5 second stride
WINDOW_SIZE_SAMPLES = int(WINDOW_SIZE_SECONDS * SAMPLE_RATE)  # 10s * 256Hz = 2560 samples
WINDOW_STRIDE_SAMPLES = int(WINDOW_STRIDE_SECONDS * SAMPLE_RATE) # 0.5s * 256Hz = 128 samples

# Skip the first 30 seconds of each trial, which is a neutral baseline
WINDOW_OFFSET_SECONDS = 30
WINDOW_OFFSET_SAMPLES = int(WINDOW_OFFSET_SECONDS * SAMPLE_RATE)

# --- Filtering ---
# Change this to True to enable outlier filtering
FILTER_OUTLIERS = True

# --- Channel Definitions ---
# These will be dynamically located, but we define the patterns to search for.
# CORRECTED: EEG channel names do not have the 'EEG ' prefix.
CHANNEL_PATTERNS = {
    'eeg':  ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'],
    'ecg':  ['EXG1', 'EXG2'], # Typically, ECG is captured on EXG1/2
    'gsr':  ['GSR1'],
    'resp': ['Resp'],
    'temp': ['Temp']
}

def find_channel_indices(channel_names, patterns):
    """Finds indices of channels that match given patterns."""
    indices = []
    for pattern in patterns:
        try:
            # Find exact matches
            indices.append(channel_names.index(pattern))
        except ValueError:
            # Handle cases where channel names might have extra characters
            for i, name in enumerate(channel_names):
                if pattern in name and i not in indices:
                    indices.append(i)
                    break
    return indices


def load_mahnob_data():
    """Load and apply windowing to the Mahnob-HCI dataset."""
    print("Loading and windowing Mahnob-HCI dataset...")

    all_data = {key: [] for key in CHANNEL_PATTERNS.keys()}
    all_valence = []
    all_arousal = []
    all_subject_ids = []
    all_trial_ids = []
    
    channel_indices = {}

    # Find all BDF files
    bdf_files = [os.path.join(root, file) for root, _, files in os.walk(SESSIONS_PATH) for file in files if file.endswith('.bdf')]
    bdf_files.sort()

    if not bdf_files:
        raise FileNotFoundError(f"No .bdf files found in {SESSIONS_PATH}. Please check the path.")
    
    print(f"Found {len(bdf_files)} trials to process.")

    for bdf_path in tqdm(bdf_files, desc="Processing Trials"):
        try:
            # --- Extract Subject and Trial ID from filename ---
            filename = os.path.basename(bdf_path)
            match = re.search(r'Part_(\d+)_.*_Trial(\d+)', filename)
            if not match:
                print(f"Warning: Could not parse subject/trial from '{filename}'. Skipping.")
                continue
            subject_id = int(match.group(1))
            trial_id = int(match.group(2))

            # --- Load Signal Data ---
            raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose=False)
            signal_data = raw.get_data()
            
            # --- Locate Channel Indices (if first file) ---
            if not channel_indices:
                print("Dynamically locating channel indices from first file...")
                names = raw.ch_names
                for key, patterns in CHANNEL_PATTERNS.items():
                    indices = find_channel_indices(names, patterns)
                    if not indices:
                         print(f"Warning: No channels found for modality '{key}'")
                    channel_indices[key] = indices
                    print(f"  - Found {len(indices)} channels for '{key}'.")
            
            # --- Load Emotion Labels ---
            xml_path = os.path.join(os.path.dirname(bdf_path), 'session.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()
            valence = int(root.get('feltVlnc'))
            arousal = int(root.get('feltArsl'))

            # --- Apply Sliding Window ---
            num_samples = signal_data.shape[1]
            for start in range(WINDOW_OFFSET_SAMPLES, num_samples - WINDOW_SIZE_SAMPLES + 1, WINDOW_STRIDE_SAMPLES):
                end = start + WINDOW_SIZE_SAMPLES
                
                # Extract window for each modality
                for key, indices in channel_indices.items():
                    if indices:
                        window = signal_data[indices, start:end]
                        all_data[key].append(window)

                # Append labels and IDs for the window
                all_valence.append(valence)
                all_arousal.append(arousal)
                all_subject_ids.append(subject_id - 1)  # Zero-indexed
                all_trial_ids.append(trial_id)

        except Exception as e:
            print(f"Error processing file {bdf_path}: {e}")
    
    # --- Convert lists to numpy arrays and create emotion labels ---
    processed_data = {}
    for key, data_list in all_data.items():
        if data_list:
            processed_data[key] = np.stack(data_list)
    
    valence = np.array(all_valence)
    arousal = np.array(all_arousal)
    subject_ids = np.array(all_subject_ids)
    
    # Binarize labels (high/low) using a threshold of 5
    valence_labels = (valence > 5).astype(int)
    arousal_labels = (arousal > 5).astype(int)
    # Combine into 4 quadrants: 0=sad (LVLA), 1=fear/angry (LVHA), 2=calm (HVLA), 3=happy (HVHA)
    emotion_labels = valence_labels * 2 + arousal_labels
    
    return processed_data, emotion_labels, subject_ids, valence_labels, arousal_labels

def preprocess_data(data_dict, emotion_labels, subject_ids, valence, arousal, filter_outliers=True):
    """Apply normalization and optionally remove outliers."""
    print("\nPreprocessing data...")
    
    # --- Convert to Torch Tensors ---
    tensor_dict = {key: torch.from_numpy(data).float() for key, data in data_dict.items()}
    emotion_labels = torch.from_numpy(emotion_labels)
    subject_ids = torch.from_numpy(subject_ids)
    valence = torch.from_numpy(valence)
    arousal = torch.from_numpy(arousal)
    
    # --- Normalization ---
    # Normalize signals by the magnitude of the first sample of each modality
    print("Applying magnitude normalization...")
    for key, tensor in tensor_dict.items():
        if tensor.numel() > 0:
            max_val = tensor[0].abs().max()
            if max_val > 1e-6: # Avoid division by zero
                tensor_dict[key] = tensor / max_val

    # --- Outlier Filtering ---
    if filter_outliers:
        print("Applying outlier filtering...")
        num_initial_samples = emotion_labels.shape[0]
        
        # Start with a mask of all True
        valid_mask = torch.ones(num_initial_samples, dtype=torch.bool)
        
        for key, tensor in tensor_dict.items():
            if tensor.numel() > 0:
                # Check for values greater than 5 or less than -5
                is_outlier = (tensor > 5) | (tensor < -5)
                # A sample is invalid if ANY channel at ANY time point is an outlier
                invalid_samples = is_outlier.any(dim=2).any(dim=1)
                valid_mask &= ~invalid_samples

        num_final_samples = valid_mask.sum().item()
        print(f"Removed {num_initial_samples - num_final_samples} outlier samples. "
              f"Keeping {num_final_samples} samples.")

        # Apply the final mask to all data
        filtered_dict = {key: tensor[valid_mask] for key, tensor in tensor_dict.items()}
        emotion_labels = emotion_labels[valid_mask]
        subject_ids = subject_ids[valid_mask]
        valence = valence[valid_mask]
        arousal = arousal[valid_mask]

    else:
        print("Skipping outlier filtering.")
        filtered_dict = tensor_dict

    return filtered_dict, emotion_labels, subject_ids, valence, arousal

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting Preprocessing for Mahnob-HCI Dataset ---")
    
    # 1. Load and window the data
    raw_data_dict, emotion_labels, subject_ids, valence, arousal = load_mahnob_data()
    
    # 2. Preprocess the windowed data (normalize, filter)
    processed_data_dict, emotion_labels, subject_ids, valence, arousal = preprocess_data(
        raw_data_dict, emotion_labels, subject_ids, valence, arousal, filter_outliers=FILTER_OUTLIERS
    )

    # 3. Save processed data to .npy files
    print("\nSaving processed data...")
    for key, data_tensor in processed_data_dict.items():
        if data_tensor.numel() > 0:
            filepath = f'{DATA_DIR}/{key}_data.npy'
            np.save(filepath, data_tensor.numpy())
            print(f"  - Saved {key} data to {filepath} with shape {data_tensor.shape}")
            
    np.save(f'{DATA_DIR}/subject.npy', subject_ids.numpy())
    np.save(f'{DATA_DIR}/emotion_labels.npy', emotion_labels.numpy())
    np.save(f'{DATA_DIR}/valence.npy', valence.numpy())
    np.save(f'{DATA_DIR}/arousal.npy', arousal.numpy())
    print("  - Saved labels and subject IDs.")

    print("\n--- Preprocessing Complete! ---")
    
    # --- Final Summary ---
    total_subjects = len(np.unique(subject_ids.numpy()))
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    print(f"Subjects kept in processed dataset: {total_subjects}")
    print(f"Total samples after preprocessing: {emotion_labels.shape[0]}")
    if total_subjects > 0:
      print(f"Average samples per subject: {emotion_labels.shape[0] / total_subjects:.1f}")
    print("="*50)

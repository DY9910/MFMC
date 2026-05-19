import os
import pickle
import shutil

import numpy as np


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")


# Define base paths.
# Set DEAP_RAW_DATA_DIR if the raw DEAP .dat files live outside this repository.
BASE_PATH = os.environ.get("DEAP_BASE_PATH", "/home/zhengdeyang/TAFFC_MFMC/MFMC/DEAP")
RAW_DATA_DIR = os.environ.get("DEAP_RAW_DATA_DIR", f"{BASE_PATH}/DEAP_data")
DATA_DIR = os.environ.get("DEAP_PROCESSED_DIR", f"{BASE_PATH}/Data_processed")

create_folder_if_not_exists(DATA_DIR)

# DEAP channel indices in the loaded numpy arrays are 0-based.
# The peripheral mapping follows the provided DEAP channel table:
# 32 hEOG, 33 vEOG, 34 zEMG, 35 tEMG, 36 GSR, 37 Pleth/BVP,
# 38 Resp, 39 Temp/SKT.
EEG_CHANNELS = list(range(0, 32))
EOG_CHANNELS = [32, 33]
EMG_CHANNELS = [34, 35]
GSR_CHANNELS = [36]
# Under the current repo mapping, raw channel 37 is the only peripheral channel
# not already assigned to EEG/EOG/EMG/GSR/Resp/SKT, matching DEAP Pleth/BVP.
PLETH_CHANNELS = [37]
RESP_CHANNELS = [38]
SKT_CHANNELS = [39]

_USED_PERIPHERAL_CHANNELS = (
    EOG_CHANNELS + EMG_CHANNELS + GSR_CHANNELS + PLETH_CHANNELS + RESP_CHANNELS + SKT_CHANNELS
)
if sorted(_USED_PERIPHERAL_CHANNELS) != list(range(32, 40)):
    raise ValueError(
        "DEAP peripheral channel mapping is ambiguous. Expected channels 32-39 "
        "to be covered exactly by EOG/EMG/GSR/Pleth/Resp/SKT."
    )

LEGACY_MODALITIES = {
    "eeg": EEG_CHANNELS,
    "eog": EOG_CHANNELS,
    "temp": SKT_CHANNELS,
}

NEW_MODALITIES = {
    "emg": EMG_CHANNELS,
    "gsr": GSR_CHANNELS,
    "pleth": PLETH_CHANNELS,
    "resp": RESP_CHANNELS,
}

MODALITY_OUTPUTS = {
    "eeg": "eeg_data.npy",
    "eog": "eog_data.npy",
    "temp": "temp_data.npy",
    "skt": "skt_data.npy",
    "emg": "emg_data.npy",
    "gsr": "gsr_data.npy",
    "pleth": "pleth_data.npy",
    "resp": "resp_data.npy",
}

METADATA_OUTPUTS = {
    "subject": "subject.npy",
    "emotion_labels": "emotion_labels.npy",
    "valence": "valence.npy",
    "arousal": "arousal.npy",
}

# Define windowing parameters
SAMPLE_RATE = 128
WINDOW_SIZE = 10 * SAMPLE_RATE
WINDOW_OFFSET = 3 * SAMPLE_RATE
WINDOW_STRIDE = int(0.4 * SAMPLE_RATE)

# Keep the original EEG/EOG/Temp outlier filtering behavior for alignment.
FILTER_OUTLIERS = True

# New peripheral modalities are normalized per window and clipped instead of
# causing extra sample drops. This keeps them aligned with existing outputs.
NEW_MODALITY_CLIP_Z = 8.0
EPS = 1e-6


def output_path(filename):
    return os.path.join(DATA_DIR, filename)


def file_exists(filename):
    return os.path.exists(output_path(filename))


def save_if_missing(filename, array, description):
    path = output_path(filename)
    if os.path.exists(path):
        print(f"Skipping existing {description}: {path}")
        return False

    np.save(path, array)
    print(f"Saved {description}: {path} {array.shape}")
    return True


def get_available_subject_files():
    subject_files = []
    for subject_id in range(1, 33):
        filename = os.path.join(RAW_DATA_DIR, f"s{subject_id:02d}.dat")
        if os.path.exists(filename):
            subject_files.append((subject_id, filename))
        else:
            print(f"Warning: missing raw file for subject {subject_id}: {filename}")
    return subject_files


def get_first_window_legacy_maxima(subject_files):
    """Replicate the original normalization constants from the first window."""
    if not subject_files:
        raise FileNotFoundError(f"No DEAP .dat files found in {RAW_DATA_DIR}")

    first_subject_id, first_filename = subject_files[0]
    with open(first_filename, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    trial_data = data["data"][0]
    start_idx = WINDOW_OFFSET
    end_idx = start_idx + WINDOW_SIZE

    maxima = {}
    for name, channels in LEGACY_MODALITIES.items():
        window = np.asarray(trial_data[channels, start_idx:end_idx], dtype=np.float32)
        max_value = np.abs(window).max()
        if max_value < EPS:
            raise ValueError(
                f"Cannot normalize {name}: first window max is {max_value} "
                f"in subject {first_subject_id}"
            )
        maxima[name] = max_value

    return maxima


def normalize_legacy_window(window, max_value):
    window = np.asarray(window, dtype=np.float32)
    return window / max_value


def is_valid_legacy_window(eeg_window, eog_window, temp_window):
    return (
        (eeg_window <= 5).all()
        and (eeg_window >= -5).all()
        and (eog_window <= 5).all()
        and (eog_window >= -5).all()
        and (temp_window <= 5).all()
        and (temp_window >= -5).all()
    )


def preprocess_new_peripheral_data(data):
    """Lenient per-window z-score normalization for added peripheral modalities."""
    data = np.asarray(data, dtype=np.float32)
    mean = data.mean(axis=2, keepdims=True)
    std = data.std(axis=2, keepdims=True)
    data = (data - mean) / np.maximum(std, EPS)
    data = np.nan_to_num(data, copy=False)
    data = np.clip(data, -NEW_MODALITY_CLIP_Z, NEW_MODALITY_CLIP_Z)
    return data.astype(np.float32, copy=False)


def build_processed_deap_data(collect_modalities):
    """Window DEAP data once and collect only missing output modalities."""
    print("Loading and windowing DEAP dataset...")
    print(f"Raw data directory: {RAW_DATA_DIR}")

    subject_files = get_available_subject_files()
    maxima = get_first_window_legacy_maxima(subject_files)

    collected = {name: [] for name in collect_modalities}
    metadata = {
        "subject": [],
        "emotion_labels": [],
        "valence": [],
        "arousal": [],
    }

    total_windows = 0
    kept_windows = 0

    for subject_id, filename in subject_files:
        print(f"Processing subject {subject_id:02d}...")
        try:
            with open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")

            for trial in range(len(data["data"])):
                trial_data = data["data"][trial]
                valence_score = data["labels"][trial, 0]
                arousal_score = data["labels"][trial, 1]
                valence_label = 1 if valence_score > 5 else 0
                arousal_label = 1 if arousal_score > 5 else 0
                emotion_label = valence_label * 2 + arousal_label

                for start_idx in range(
                    WINDOW_OFFSET,
                    trial_data.shape[1] - WINDOW_SIZE + 1,
                    WINDOW_STRIDE,
                ):
                    total_windows += 1
                    end_idx = start_idx + WINDOW_SIZE

                    eeg_window = normalize_legacy_window(
                        trial_data[EEG_CHANNELS, start_idx:end_idx],
                        maxima["eeg"],
                    )
                    eog_window = normalize_legacy_window(
                        trial_data[EOG_CHANNELS, start_idx:end_idx],
                        maxima["eog"],
                    )
                    temp_window = normalize_legacy_window(
                        trial_data[SKT_CHANNELS, start_idx:end_idx],
                        maxima["temp"],
                    )

                    if FILTER_OUTLIERS and not is_valid_legacy_window(
                        eeg_window,
                        eog_window,
                        temp_window,
                    ):
                        continue

                    kept_windows += 1
                    metadata["subject"].append(subject_id - 1)
                    metadata["emotion_labels"].append(emotion_label)
                    metadata["valence"].append(valence_label)
                    metadata["arousal"].append(arousal_label)

                    if "eeg" in collected:
                        collected["eeg"].append(eeg_window)
                    if "eog" in collected:
                        collected["eog"].append(eog_window)
                    if "temp" in collected:
                        collected["temp"].append(temp_window)
                    if "emg" in collected:
                        collected["emg"].append(
                            np.asarray(
                                trial_data[EMG_CHANNELS, start_idx:end_idx],
                                dtype=np.float32,
                            )
                        )
                    if "gsr" in collected:
                        collected["gsr"].append(
                            np.asarray(
                                trial_data[GSR_CHANNELS, start_idx:end_idx],
                                dtype=np.float32,
                            )
                        )
                    if "pleth" in collected:
                        collected["pleth"].append(
                            np.asarray(
                                trial_data[PLETH_CHANNELS, start_idx:end_idx],
                                dtype=np.float32,
                            )
                        )
                    if "resp" in collected:
                        collected["resp"].append(
                            np.asarray(
                                trial_data[RESP_CHANNELS, start_idx:end_idx],
                                dtype=np.float32,
                            )
                        )

        except Exception as e:
            print(f"Error loading subject {subject_id}: {e}")

    if kept_windows == 0:
        raise RuntimeError("No valid windows were produced from the DEAP raw data.")

    print(f"Total windows before filtering: {total_windows}")
    print(f"Windows kept by legacy EEG/EOG/Temp filter: {kept_windows}")

    for key, values in metadata.items():
        metadata[key] = np.asarray(values, dtype=np.int64)

    processed = {}
    for name, values in collected.items():
        if not values:
            continue
        stacked = np.stack(values).astype(np.float32, copy=False)
        if name in NEW_MODALITIES:
            stacked = preprocess_new_peripheral_data(stacked)
        processed[name] = stacked

    return processed, metadata


def validate_existing_metadata(metadata):
    for key, filename in METADATA_OUTPUTS.items():
        path = output_path(filename)
        if not os.path.exists(path):
            continue

        existing = np.load(path, mmap_mode="r")
        generated = metadata[key]
        if existing.shape != generated.shape or not np.array_equal(existing, generated):
            raise RuntimeError(
                f"Existing {filename} does not match the current DEAP raw-data "
                "windowing/filtering. Refusing to generate new modalities because "
                "they would not be aligned with existing processed data."
            )


def validate_existing_modality_shapes(metadata):
    expected_samples = len(metadata["subject"])
    for filename in MODALITY_OUTPUTS.values():
        path = output_path(filename)
        if not os.path.exists(path):
            continue

        existing = np.load(path, mmap_mode="r")
        if existing.shape[0] != expected_samples:
            raise RuntimeError(
                f"Existing {filename} has {existing.shape[0]} samples, but the "
                f"current DEAP filter keeps {expected_samples}. Refusing to mix "
                "misaligned modalities."
            )


def write_channel_names_if_missing():
    filename = "channel_names.txt"
    path = output_path(filename)
    if os.path.exists(path):
        print(f"Skipping existing channel reference: {path}")
        return

    with open(path, "w") as f:
        f.write("DEAP processed modalities:\n")
        f.write("\nEEG (eeg_data.npy): channels 0-31\n")
        f.write("EOG (eog_data.npy):\n")
        f.write("  0: hEOG (raw channel 32)\n")
        f.write("  1: vEOG (raw channel 33)\n")
        f.write("SKT/Temperature:\n")
        f.write("  temp_data.npy and skt_data.npy are raw channel 39\n")
        f.write("EMG (emg_data.npy): raw channels 34-35\n")
        f.write("GSR (gsr_data.npy): raw channel 36\n")
        f.write("Plethysmograph/BVP (pleth_data.npy): raw channel 37\n")
        f.write("Resp (resp_data.npy): raw channel 38\n")
        f.write("\nWindowing:\n")
        f.write("  sample_rate: 128 Hz\n")
        f.write("  window_size: 10 seconds / 1280 samples\n")
        f.write("  window_offset: first 3 seconds skipped\n")
        f.write("  window_stride: 0.4 seconds / 51 samples\n")
        f.write("\nPreprocessing:\n")
        f.write("  eeg/eog/temp keep the legacy max-normalization and outlier filter.\n")
        f.write("  emg/gsr/pleth/resp use per-window z-score normalization and +/-8 z clipping.\n")
    print(f"Saved channel reference: {path}")


def copy_temp_to_skt_if_needed():
    temp_path = output_path(MODALITY_OUTPUTS["temp"])
    skt_path = output_path(MODALITY_OUTPUTS["skt"])
    if os.path.exists(skt_path):
        print(f"Skipping existing SKT alias: {skt_path}")
        return True

    if os.path.exists(temp_path):
        shutil.copy2(temp_path, skt_path)
        print(f"Created SKT alias from existing temp data: {skt_path}")
        return True

    return False


def main():
    print("Starting preprocessing for DEAP dataset...")
    print(f"Outlier filtering: {'ENABLED' if FILTER_OUTLIERS else 'DISABLED'}")
    print(f"Window size: {WINDOW_SIZE} samples")
    print(f"Window stride: {WINDOW_STRIDE} samples")

    # Preserve the legacy temp_data.npy output while also exposing the requested
    # SKT modality name used by other datasets/scripts.
    skt_alias_ready = copy_temp_to_skt_if_needed()

    missing_modalities = [
        name
        for name, filename in MODALITY_OUTPUTS.items()
        if name != "skt" and not file_exists(filename)
    ]
    if not skt_alias_ready:
        missing_modalities.append("temp")

    missing_metadata = [
        key for key, filename in METADATA_OUTPUTS.items() if not file_exists(filename)
    ]

    if not missing_modalities and not missing_metadata:
        print("All requested processed files already exist. Nothing to do.")
        write_channel_names_if_missing()
        return

    # Only collect files that need to be generated. Existing modalities are read
    # only for shape checks and are never overwritten.
    collect_modalities = sorted(set(missing_modalities))
    processed, metadata = build_processed_deap_data(collect_modalities)

    validate_existing_metadata(metadata)
    validate_existing_modality_shapes(metadata)

    for key, filename in METADATA_OUTPUTS.items():
        save_if_missing(filename, metadata[key], key)

    for name in collect_modalities:
        filename = MODALITY_OUTPUTS[name]
        save_if_missing(filename, processed[name], f"{name.upper()} data")

    copy_temp_to_skt_if_needed()
    write_channel_names_if_missing()

    print("Preprocessing complete!")
    for filename in MODALITY_OUTPUTS.values():
        path = output_path(filename)
        if os.path.exists(path):
            data = np.load(path, mmap_mode="r")
            print(f"{filename}: {data.shape} {data.dtype}")
    print(f"Emotion labels: {metadata['emotion_labels'].shape}")

    unique_subjects = np.unique(metadata["subject"])
    total_subjects_kept = len(unique_subjects)
    original_subjects = 32
    subject_ids_display = sorted((unique_subjects + 1).tolist())

    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"Original subjects in DEAP dataset: {original_subjects}")
    print(f"Subjects kept in processed dataset: {total_subjects_kept}")
    print(f"Subjects removed: {original_subjects - total_subjects_kept}")
    print(f"Subject retention rate: {(total_subjects_kept / original_subjects) * 100:.1f}%")
    print(f"Total samples after preprocessing: {len(metadata['subject'])}")
    print(f"Average samples per subject: {len(metadata['subject']) / total_subjects_kept:.1f}")
    print("\nSubject IDs in processed dataset:")
    print(f"  {subject_ids_display}")

    if total_subjects_kept < original_subjects:
        all_original_subjects = list(range(1, original_subjects + 1))
        removed_subjects = [s for s in all_original_subjects if s not in subject_ids_display]
        print("\nRemoved subject IDs:")
        print(f"  {removed_subjects}")
    print("=" * 50)


if __name__ == "__main__":
    main()

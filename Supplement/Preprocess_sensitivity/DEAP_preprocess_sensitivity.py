"""Generate standalone DEAP preprocessing-sensitivity datasets.

Run this script inside the `MFMC` conda environment, for example:

    conda activate MFMC
    python MFMC/Supplement/Preprocess_sensitivity/DEAP_preprocess_sensitivity.py
"""

from __future__ import annotations

import csv
import json
import os
import pickle
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm


PROJECT_DIR = Path(
    "/home/zhengdeyang/TAFFC_MFMC/MFMC/Supplement/Preprocess_sensitivity"
)
PROCESSED_ROOT = PROJECT_DIR / "Processed_data"
MANIFEST_PATH = PROCESSED_ROOT / "manifest.csv"

DEAP_BASE_PATH = Path(
    os.environ.get("DEAP_BASE_PATH", "/home/zhengdeyang/TAFFC_MFMC/MFMC/DEAP")
)
RAW_DATA_DIR = Path(
    os.environ.get("DEAP_RAW_DATA_DIR", str(DEAP_BASE_PATH / "DEAP_data"))
)

SAMPLE_RATE = 128
EPS = 1e-6

EEG_CHANNELS = list(range(0, 32))
EOG_CHANNELS = [32, 33]
TEMP_CHANNELS = [39]

OUTPUT_FILES = {
    "eeg": "eeg_data.npy",
    "eog": "eog_data.npy",
    "temp": "temp_data.npy",
    "skt": "skt_data.npy",
    "emotion_labels": "emotion_labels.npy",
    "subject": "subject.npy",
    "valence": "valence.npy",
    "arousal": "arousal.npy",
    "meta": "preprocess_meta.json",
}


@dataclass(frozen=True)
class SubjectFile:
    subject_id: int
    subject_index: int
    path: Path
    n_trials: int


def create_folder_if_not_exists(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_available_subject_files(raw_dir: Path = RAW_DATA_DIR) -> List[SubjectFile]:
    subject_files: List[SubjectFile] = []
    for subject_id in range(1, 33):
        filename = raw_dir / f"s{subject_id:02d}.dat"
        if not filename.exists():
            print(f"Warning: missing raw DEAP file: {filename}")
            continue

        with filename.open("rb") as handle:
            payload = pickle.load(handle, encoding="latin1")

        n_trials = int(len(payload["data"]))
        subject_files.append(
            SubjectFile(
                subject_id=subject_id,
                subject_index=subject_id - 1,
                path=filename,
                n_trials=n_trials,
            )
        )
    if not subject_files:
        raise FileNotFoundError(f"No DEAP .dat files found in {raw_dir}")
    return subject_files


def get_first_window_legacy_maxima(
    subject_files: Iterable[SubjectFile],
    window_length_sec: float,
    offset_sec: float,
    sample_rate: int = SAMPLE_RATE,
) -> Dict[str, float]:
    window_size = int(window_length_sec * sample_rate)
    offset = int(offset_sec * sample_rate)

    for subject_file in subject_files:
        with subject_file.path.open("rb") as handle:
            payload = pickle.load(handle, encoding="latin1")

        for trial_data in payload["data"]:
            if trial_data.shape[1] < offset + window_size:
                continue

            start_idx = offset
            end_idx = start_idx + window_size
            maxima = {
                "eeg": float(
                    np.abs(trial_data[EEG_CHANNELS, start_idx:end_idx]).max()
                ),
                "eog": float(
                    np.abs(trial_data[EOG_CHANNELS, start_idx:end_idx]).max()
                ),
                "temp": float(
                    np.abs(trial_data[TEMP_CHANNELS, start_idx:end_idx]).max()
                ),
            }
            for name, value in maxima.items():
                if value < EPS:
                    raise ValueError(
                        f"Cannot compute legacy normalization for {name}: "
                        f"first-window max is {value}."
                    )
            return maxima

    raise RuntimeError(
        "Could not locate any DEAP trial long enough for the requested first window."
    )


def extract_trial_windows(
    trial_data: np.ndarray,
    window_length_sec: float,
    offset_sec: float,
    stride_sec: float,
    sample_rate: int = SAMPLE_RATE,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    window_size = int(window_length_sec * sample_rate)
    offset = int(offset_sec * sample_rate)
    stride = int(stride_sec * sample_rate)
    final_start = trial_data.shape[1] - window_size

    if final_start < offset:
        return

    for start_idx in range(offset, final_start + 1, stride):
        end_idx = start_idx + window_size
        eeg_window = np.asarray(
            trial_data[EEG_CHANNELS, start_idx:end_idx], dtype=np.float32
        )
        eog_window = np.asarray(
            trial_data[EOG_CHANNELS, start_idx:end_idx], dtype=np.float32
        )
        temp_window = np.asarray(
            trial_data[TEMP_CHANNELS, start_idx:end_idx], dtype=np.float32
        )
        yield eeg_window, eog_window, temp_window


def apply_normalization(
    window: np.ndarray,
    mode: str,
    legacy_maxima: Optional[float] = None,
    eps: float = EPS,
) -> np.ndarray:
    window = np.asarray(window, dtype=np.float32)

    if mode == "legacy":
        if legacy_maxima is None:
            raise ValueError("legacy_maxima is required for legacy normalization.")
        return window / float(legacy_maxima)

    if mode == "firstsample":
        scale = np.maximum(np.abs(window[:, :1]), eps)
        return window / scale

    if mode == "zscore":
        mean = window.mean(axis=1, keepdims=True)
        std = window.std(axis=1, keepdims=True)
        return (window - mean) / np.maximum(std, eps)

    raise ValueError(f"Unsupported normalization mode: {mode}")


def is_valid_window(
    eeg_window: np.ndarray,
    eog_window: np.ndarray,
    temp_window: np.ndarray,
    threshold: Optional[float],
) -> bool:
    if threshold is None:
        return True

    return (
        float(np.abs(eeg_window).max()) <= threshold
        and float(np.abs(eog_window).max()) <= threshold
        and float(np.abs(temp_window).max()) <= threshold
    )


def save_setting_outputs(
    output_dir: Path,
    arrays: Dict[str, np.ndarray],
    meta: Dict[str, object],
) -> None:
    create_folder_if_not_exists(output_dir)
    np.save(output_dir / OUTPUT_FILES["eeg"], arrays["eeg"])
    np.save(output_dir / OUTPUT_FILES["eog"], arrays["eog"])
    np.save(output_dir / OUTPUT_FILES["temp"], arrays["temp"])
    np.save(output_dir / OUTPUT_FILES["skt"], arrays["temp"])
    np.save(output_dir / OUTPUT_FILES["emotion_labels"], arrays["emotion_labels"])
    np.save(output_dir / OUTPUT_FILES["subject"], arrays["subject"])
    np.save(output_dir / OUTPUT_FILES["valence"], arrays["valence"])
    np.save(output_dir / OUTPUT_FILES["arousal"], arrays["arousal"])
    (output_dir / OUTPUT_FILES["meta"]).write_text(json.dumps(meta, indent=2))


def update_manifest(records: List[Dict[str, object]], manifest_path: Path = MANIFEST_PATH) -> None:
    create_folder_if_not_exists(manifest_path.parent)
    fieldnames = [
        "factor",
        "setting_name",
        "output_dir",
        "window_length_sec",
        "offset_sec",
        "stride_sec",
        "normalization",
        "outlier_threshold",
        "kept_windows",
        "removed_windows",
        "class_distribution_json",
        "elapsed_sec",
    ]
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def build_single_setting(
    setting_config: Dict[str, object],
    subject_files: List[SubjectFile],
) -> Dict[str, object]:
    output_dir = Path(str(setting_config["output_dir"]))
    normalization = str(setting_config["normalization"])
    threshold = setting_config["outlier_threshold"]
    threshold = None if threshold is None else float(threshold)

    legacy_maxima = None
    if normalization == "legacy":
        legacy_maxima = get_first_window_legacy_maxima(
            subject_files=subject_files,
            window_length_sec=float(setting_config["window_length_sec"]),
            offset_sec=float(setting_config["offset_sec"]),
            sample_rate=SAMPLE_RATE,
        )

    eeg_windows: List[np.ndarray] = []
    eog_windows: List[np.ndarray] = []
    temp_windows: List[np.ndarray] = []
    subjects: List[int] = []
    emotion_labels: List[int] = []
    valence_labels: List[int] = []
    arousal_labels: List[int] = []

    kept_windows = 0
    removed_windows = 0
    class_distribution = {0: 0, 1: 0, 2: 0, 3: 0}
    total_trials = sum(subject_file.n_trials for subject_file in subject_files)

    setting_start = time.time()
    with tqdm(
        total=total_trials,
        desc=f"{setting_config['setting_name']}",
        unit="trial",
        leave=False,
    ) as inner_bar:
        for subject_file in subject_files:
            with subject_file.path.open("rb") as handle:
                payload = pickle.load(handle, encoding="latin1")

            for trial_idx, trial_data in enumerate(payload["data"]):
                valence_score = float(payload["labels"][trial_idx, 0])
                arousal_score = float(payload["labels"][trial_idx, 1])
                valence_label = 1 if valence_score > 5 else 0
                arousal_label = 1 if arousal_score > 5 else 0
                emotion_label = valence_label * 2 + arousal_label

                for eeg_window, eog_window, temp_window in extract_trial_windows(
                    trial_data=trial_data,
                    window_length_sec=float(setting_config["window_length_sec"]),
                    offset_sec=float(setting_config["offset_sec"]),
                    stride_sec=float(setting_config["stride_sec"]),
                    sample_rate=SAMPLE_RATE,
                ):
                    eeg_norm = apply_normalization(
                        eeg_window, normalization, None if legacy_maxima is None else legacy_maxima["eeg"]
                    )
                    eog_norm = apply_normalization(
                        eog_window, normalization, None if legacy_maxima is None else legacy_maxima["eog"]
                    )
                    temp_norm = apply_normalization(
                        temp_window, normalization, None if legacy_maxima is None else legacy_maxima["temp"]
                    )

                    if not is_valid_window(eeg_norm, eog_norm, temp_norm, threshold):
                        removed_windows += 1
                        continue

                    eeg_windows.append(eeg_norm.astype(np.float32, copy=False))
                    eog_windows.append(eog_norm.astype(np.float32, copy=False))
                    temp_windows.append(temp_norm.astype(np.float32, copy=False))
                    subjects.append(subject_file.subject_index)
                    emotion_labels.append(emotion_label)
                    valence_labels.append(valence_label)
                    arousal_labels.append(arousal_label)
                    class_distribution[emotion_label] += 1
                    kept_windows += 1

                inner_bar.update(1)
                inner_bar.set_postfix(
                    kept=kept_windows,
                    removed=removed_windows,
                    subject=f"{subject_file.subject_id:02d}",
                    trial=trial_idx + 1,
                    refresh=False,
                )

    if kept_windows == 0:
        raise RuntimeError(f"No valid windows generated for {setting_config['setting_name']}.")

    arrays = {
        "eeg": np.stack(eeg_windows).astype(np.float32, copy=False),
        "eog": np.stack(eog_windows).astype(np.float32, copy=False),
        "temp": np.stack(temp_windows).astype(np.float32, copy=False),
        "emotion_labels": np.asarray(emotion_labels, dtype=np.int64),
        "subject": np.asarray(subjects, dtype=np.int64),
        "valence": np.asarray(valence_labels, dtype=np.int64),
        "arousal": np.asarray(arousal_labels, dtype=np.int64),
    }

    elapsed_sec = time.time() - setting_start
    meta = {
        "factor": setting_config["factor"],
        "setting_name": setting_config["setting_name"],
        "output_dir": str(output_dir),
        "raw_data_dir": str(RAW_DATA_DIR),
        "sample_rate_hz": SAMPLE_RATE,
        "window_length_sec": float(setting_config["window_length_sec"]),
        "offset_sec": float(setting_config["offset_sec"]),
        "stride_sec": float(setting_config["stride_sec"]),
        "normalization": normalization,
        "outlier_threshold": threshold,
        "legacy_maxima": legacy_maxima,
        "kept_windows": kept_windows,
        "removed_windows": removed_windows,
        "class_distribution": class_distribution,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_setting_outputs(output_dir, arrays, meta)

    record = {
        "factor": setting_config["factor"],
        "setting_name": setting_config["setting_name"],
        "output_dir": str(output_dir),
        "window_length_sec": setting_config["window_length_sec"],
        "offset_sec": setting_config["offset_sec"],
        "stride_sec": setting_config["stride_sec"],
        "normalization": normalization,
        "outlier_threshold": "" if threshold is None else threshold,
        "kept_windows": kept_windows,
        "removed_windows": removed_windows,
        "class_distribution_json": json.dumps(class_distribution, sort_keys=True),
        "elapsed_sec": round(elapsed_sec, 3),
    }

    tqdm.write(
        f"[{setting_config['setting_name']}] done in {elapsed_sec / 60:.1f} min | "
        f"kept={kept_windows} | removed={removed_windows} | saved to {output_dir}"
    )
    return record


def build_all_settings() -> List[Dict[str, object]]:
    return [
        {
            "factor": "baseline",
            "setting_name": "baseline",
            "output_dir": str(PROCESSED_ROOT / "baseline"),
            "window_length_sec": 10,
            "offset_sec": 3,
            "stride_sec": 0.4,
            "normalization": "legacy",
            "outlier_threshold": 5,
        },
        {
            "factor": "window_length",
            "setting_name": "wl_2",
            "output_dir": str(PROCESSED_ROOT / "window_length" / "wl_2"),
            "window_length_sec": 2,
            "offset_sec": 3,
            "stride_sec": 0.4,
            "normalization": "legacy",
            "outlier_threshold": 5,
        },
        {
            "factor": "window_length",
            "setting_name": "wl_5",
            "output_dir": str(PROCESSED_ROOT / "window_length" / "wl_5"),
            "window_length_sec": 5,
            "offset_sec": 3,
            "stride_sec": 0.4,
            "normalization": "legacy",
            "outlier_threshold": 5,
        },
        {
            "factor": "window_length",
            "setting_name": "wl_10",
            "output_dir": str(PROCESSED_ROOT / "window_length" / "wl_10"),
            "window_length_sec": 10,
            "offset_sec": 3,
            "stride_sec": 0.4,
            "normalization": "legacy",
            "outlier_threshold": 5,
        },
        {
            "factor": "window_length",
            "setting_name": "wl_15",
            "output_dir": str(PROCESSED_ROOT / "window_length" / "wl_15"),
            "window_length_sec": 15,
            "offset_sec": 3,
            "stride_sec": 0.4,
            "normalization": "legacy",
            "outlier_threshold": 5,
        },
        {
            "factor": "stride",
            "setting_name": "stride_0p4",
            "output_dir": str(PROCESSED_ROOT / "stride" / "stride_0p4"),
            "window_length_sec": 10,
            "offset_sec": 3,
            "stride_sec": 0.4,
            "normalization": "legacy",
            "outlier_threshold": 5,
        },
        {
            "factor": "stride",
            "setting_name": "stride_1",
            "output_dir": str(PROCESSED_ROOT / "stride" / "stride_1"),
            "window_length_sec": 10,
            "offset_sec": 3,
            "stride_sec": 1,
            "normalization": "legacy",
            "outlier_threshold": 5,
        },
        {
            "factor": "stride",
            "setting_name": "stride_2",
            "output_dir": str(PROCESSED_ROOT / "stride" / "stride_2"),
            "window_length_sec": 10,
            "offset_sec": 3,
            "stride_sec": 2,
            "normalization": "legacy",
            "outlier_threshold": 5,
        },
        {
            "factor": "stride",
            "setting_name": "stride_10",
            "output_dir": str(PROCESSED_ROOT / "stride" / "stride_10"),
            "window_length_sec": 10,
            "offset_sec": 3,
            "stride_sec": 10,
            "normalization": "legacy",
            "outlier_threshold": 5,
        },
        {
            "factor": "normalization",
            "setting_name": "norm_legacy",
            "output_dir": str(PROCESSED_ROOT / "normalization" / "norm_legacy"),
            "window_length_sec": 10,
            "offset_sec": 3,
            "stride_sec": 0.4,
            "normalization": "legacy",
            "outlier_threshold": 5,
        },
        {
            "factor": "normalization",
            "setting_name": "norm_firstsample",
            "output_dir": str(PROCESSED_ROOT / "normalization" / "norm_firstsample"),
            "window_length_sec": 10,
            "offset_sec": 3,
            "stride_sec": 0.4,
            "normalization": "firstsample",
            "outlier_threshold": 5,
        },
        {
            "factor": "normalization",
            "setting_name": "norm_zscore",
            "output_dir": str(PROCESSED_ROOT / "normalization" / "norm_zscore"),
            "window_length_sec": 10,
            "offset_sec": 3,
            "stride_sec": 0.4,
            "normalization": "zscore",
            "outlier_threshold": 5,
        },
        {
            "factor": "outlier",
            "setting_name": "thr_none",
            "output_dir": str(PROCESSED_ROOT / "outlier" / "thr_none"),
            "window_length_sec": 10,
            "offset_sec": 3,
            "stride_sec": 0.4,
            "normalization": "legacy",
            "outlier_threshold": None,
        },
        {
            "factor": "outlier",
            "setting_name": "thr_3",
            "output_dir": str(PROCESSED_ROOT / "outlier" / "thr_3"),
            "window_length_sec": 10,
            "offset_sec": 3,
            "stride_sec": 0.4,
            "normalization": "legacy",
            "outlier_threshold": 3,
        },
        {
            "factor": "outlier",
            "setting_name": "thr_5",
            "output_dir": str(PROCESSED_ROOT / "outlier" / "thr_5"),
            "window_length_sec": 10,
            "offset_sec": 3,
            "stride_sec": 0.4,
            "normalization": "legacy",
            "outlier_threshold": 5,
        },
        {
            "factor": "outlier",
            "setting_name": "thr_7",
            "output_dir": str(PROCESSED_ROOT / "outlier" / "thr_7"),
            "window_length_sec": 10,
            "offset_sec": 3,
            "stride_sec": 0.4,
            "normalization": "legacy",
            "outlier_threshold": 7,
        },
    ]


def main() -> None:
    print("DEAP preprocessing sensitivity build")
    print(f"Raw data directory: {RAW_DATA_DIR}")
    print(f"Output root: {PROCESSED_ROOT}")
    print("Expected environment: conda activate MFMC")

    create_folder_if_not_exists(PROCESSED_ROOT)
    subject_files = get_available_subject_files()
    settings = build_all_settings()
    manifest_records: List[Dict[str, object]] = []

    with tqdm(total=len(settings), desc="All settings", unit="setting") as global_bar:
        for setting_config in settings:
            global_bar.set_postfix(current=setting_config["setting_name"], refresh=False)
            record = build_single_setting(setting_config, subject_files)
            manifest_records.append(record)
            update_manifest(manifest_records)
            global_bar.update(1)

    print(f"Manifest written to {MANIFEST_PATH}")
    print(f"Completed {len(manifest_records)} preprocessing settings.")


if __name__ == "__main__":
    main()

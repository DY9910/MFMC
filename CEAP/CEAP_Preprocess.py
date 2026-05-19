import json
import torch
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

# Create output directory if it doesn't exist
def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

# Define base paths.
# Reproducibility note: keep code paths relative to this CEAP folder by default.
# Set CEAP_RAW_DATA_DIR if the raw CEAP download lives outside this repository.
BASE_PATH = os.environ.get('CEAP_BASE_PATH', '/home/zhengdeyang/TAFFC_MFMC/MFMC/CEAP')
RAW_DATA_DIR = os.environ.get('CEAP_RAW_DATA_DIR', f'{BASE_PATH}/CEAP')
DATA_DIR = os.environ.get('CEAP_PROCESSED_DIR', f'{BASE_PATH}/CEAP_Processed')

create_folder_if_not_exists(DATA_DIR)

# Define channels for CEAP dataset (Empatica E4)
# ACC (3-axis), BVP, EDA, SKT, HR, IBI = 7 channels total
PHYSIO_CHANNELS = ['acc_x', 'acc_y', 'acc_z', 'bvp', 'eda', 'skt', 'hr', 'ibi']
ANNOTATION_CHANNELS = ['valence', 'arousal']

# Define windowing parameters for CEAP dataset
# The data is already frame-based (~30fps), so we'll create windows from the frames
FRAME_RATE = 30  # Approximate frame rate
WINDOW_SIZE_SECONDS = 10  # 10 seconds
WINDOW_SIZE_FRAMES = WINDOW_SIZE_SECONDS * FRAME_RATE  # 300 frames
WINDOW_STRIDE_SECONDS = 0.5  # 0.5 second stride
WINDOW_STRIDE_FRAMES = int(WINDOW_STRIDE_SECONDS * FRAME_RATE)  # 15 frames

# Emotion threshold
EMOTION_THRESHOLD = 5.0

# Change this to True to enable outlier filtering
FILTER_OUTLIERS = True

def load_ceap_data():
    """Load the CEAP dataset files."""
    print("Loading CEAP dataset...")
    
    all_physio_data = []
    all_valence = []
    all_arousal = []
    all_emotion_labels = []
    all_subject_ids = []
    all_video_ids = []
    
    # Check available participants
    annotation_dir = f'{RAW_DATA_DIR}/Annotation_Frame'
    physio_dir = f'{RAW_DATA_DIR}/Physio_Frame'
    
    # Get available participants (P1 to P32, but some might be missing)
    available_participants = []
    for p_id in range(1, 33):  # P1 to P32
        annotation_file = f'{annotation_dir}/P{p_id}_Annotation_FrameData.json'
        physio_file = f'{physio_dir}/P{p_id}_Physio_FrameData.json'
        
        if os.path.exists(annotation_file) and os.path.exists(physio_file):
            available_participants.append(p_id)
        else:
            print(f"Warning: Missing data files for participant P{p_id}")
    
    print(f"Found data for {len(available_participants)} participants: {available_participants}")
    
    # Load data for each available participant
    for participant_id in available_participants:
        print(f"Processing participant P{participant_id}...")
        
        # Load annotation and physiological data
        annotation_filename = f'{annotation_dir}/P{participant_id}_Annotation_FrameData.json'
        physio_filename = f'{physio_dir}/P{participant_id}_Physio_FrameData.json'
        
        try:
            with open(annotation_filename, 'r') as f:
                annotation_data = json.load(f)
            
            with open(physio_filename, 'r') as f:
                physio_data = json.load(f)
            
            # Extract data for all videos for this participant
            annotation_videos = annotation_data['ContinuousAnnotation_FrameData'][0]['Video_Annotation_FrameData']
            physio_videos = physio_data['Physio_FrameData'][0]['Video_Physio_FrameData']
            
            # Process each video
            for video_idx in range(len(annotation_videos)):
                video_annotation = annotation_videos[video_idx]
                video_physio = physio_videos[video_idx]
                
                video_id = video_annotation['VideoID']
                print(f"  Processing video {video_id}...")
                
                # Extract annotation data
                annotations = video_annotation['TimeStamp_Valence_Arousal']
                annotation_timestamps = [ann['TimeStamp'] for ann in annotations]
                valence_values = [ann['Valence'] for ann in annotations]
                arousal_values = [ann['Arousal'] for ann in annotations]
                
                # Extract physiological data and align timestamps
                physio_dict = {}
                
                # ACC data (3 channels)
                acc_data = video_physio['ACC_FrameData']
                acc_timestamps = [item['TimeStamp'] for item in acc_data]
                acc_x = [item['ACC_X'] for item in acc_data]
                acc_y = [item['ACC_Y'] for item in acc_data]
                acc_z = [item['ACC_Z'] for item in acc_data]
                
                # BVP data
                bvp_data = video_physio['BVP_FrameData']
                bvp_timestamps = [item['TimeStamp'] for item in bvp_data]
                bvp_values = [item['BVP'] for item in bvp_data]
                
                # EDA data
                eda_data = video_physio['EDA_FrameData']
                eda_timestamps = [item['TimeStamp'] for item in eda_data]
                eda_values = [item['EDA'] for item in eda_data]
                
                # SKT data
                skt_data = video_physio['SKT_FrameData']
                skt_timestamps = [item['TimeStamp'] for item in skt_data]
                skt_values = [item['SKT'] for item in skt_data]
                
                # HR data
                hr_data = video_physio['HR_FrameData']
                hr_timestamps = [item['TimeStamp'] for item in hr_data]
                hr_values = [item['HR'] for item in hr_data]
                
                # IBI data (might be missing for some participants)
                ibi_timestamps = []
                ibi_values = []
                if 'IBI_FrameData' in video_physio and video_physio['IBI_FrameData']:
                    ibi_data = video_physio['IBI_FrameData']
                    ibi_timestamps = [item['TimeStamp'] for item in ibi_data]
                    ibi_values = [item['IBI'] for item in ibi_data]
                
                # Use annotation timestamps as the master timeline
                master_timestamps = annotation_timestamps
                n_frames = len(master_timestamps)
                
                # Interpolate all physiological signals to annotation timestamps
                def interpolate_signal(signal_timestamps, signal_values, target_timestamps):
                    if not signal_timestamps or not signal_values:
                        return [0.0] * len(target_timestamps)  # Return zeros if no data
                    return np.interp(target_timestamps, signal_timestamps, signal_values)
                
                # Interpolate all signals
                acc_x_interp = interpolate_signal(acc_timestamps, acc_x, master_timestamps)
                acc_y_interp = interpolate_signal(acc_timestamps, acc_y, master_timestamps)
                acc_z_interp = interpolate_signal(acc_timestamps, acc_z, master_timestamps)
                bvp_interp = interpolate_signal(bvp_timestamps, bvp_values, master_timestamps)
                eda_interp = interpolate_signal(eda_timestamps, eda_values, master_timestamps)
                skt_interp = interpolate_signal(skt_timestamps, skt_values, master_timestamps)
                hr_interp = interpolate_signal(hr_timestamps, hr_values, master_timestamps)
                ibi_interp = interpolate_signal(ibi_timestamps, ibi_values, master_timestamps)
                
                # Combine all physiological channels
                # Order: acc_x, acc_y, acc_z, bvp, eda, skt, hr, ibi
                physio_matrix = np.array([
                    acc_x_interp, acc_y_interp, acc_z_interp,
                    bvp_interp, eda_interp, skt_interp, hr_interp, ibi_interp
                ])  # Shape: (8, n_frames)
                
                # Create windows from the frame data
                max_start_idx = n_frames - WINDOW_SIZE_FRAMES
                if max_start_idx < 0:
                    print(f"    Warning: Video {video_id} too short ({n_frames} frames), skipping")
                    continue
                
                window_start_indices = range(0, max_start_idx + 1, WINDOW_STRIDE_FRAMES)
                
                for start_idx in window_start_indices:
                    end_idx = start_idx + WINDOW_SIZE_FRAMES
                    
                    # Extract physiological window
                    physio_window = physio_matrix[:, start_idx:end_idx]  # Shape: (8, window_size)
                    
                    # Extract annotation window
                    valence_window = valence_values[start_idx:end_idx]
                    arousal_window = arousal_values[start_idx:end_idx]
                    
                    # Take average of valence and arousal in this window
                    valence_avg = np.mean(valence_window)
                    arousal_avg = np.mean(arousal_window)
                    
                    # Create 4-class emotion labels based on valence and arousal
                    # Same as CASE: LAHV=0, HALV=1, LALV=2, HAHV=3
                    valence_high = valence_avg > EMOTION_THRESHOLD
                    arousal_high = arousal_avg > EMOTION_THRESHOLD
                    
                    if valence_high and arousal_high:
                        emotion_label = 3  # HAHV
                    elif valence_high and not arousal_high:
                        emotion_label = 0  # LAHV
                    elif not valence_high and arousal_high:
                        emotion_label = 1  # HALV
                    else:
                        emotion_label = 2  # LALV
                    
                    # Store data
                    all_physio_data.append(physio_window)  # Shape: (8, window_size)
                    all_valence.append(valence_avg)
                    all_arousal.append(arousal_avg)
                    all_emotion_labels.append(emotion_label)
                    all_subject_ids.append(participant_id - 1)  # Zero-indexed
                    all_video_ids.append(int(video_id[1:]) - 1)  # Convert "V1" -> 0, "V2" -> 1, etc.
                
        except Exception as e:
            print(f"Error loading participant P{participant_id}: {e}")
            continue
    
    # Convert lists to numpy arrays
    physio_data = np.stack(all_physio_data)  # Shape: (n_windows, 8, window_size)
    valence = np.array(all_valence)
    arousal = np.array(all_arousal)
    emotion_labels = np.array(all_emotion_labels)
    subject_ids = np.array(all_subject_ids)
    video_ids = np.array(all_video_ids)
    
    print(f"Loaded {len(physio_data)} windows from CEAP dataset")
    print(f"Physiological data shape: {physio_data.shape}")
    
    return physio_data, emotion_labels, subject_ids, valence, arousal, video_ids

def preprocess_data(physio_data, emotion_labels, subject_ids, valence, arousal, video_ids, filter_outliers=True):
    """Apply normalization and optionally remove outliers."""
    print("Preprocessing data...")
    
    # Convert to torch tensors
    physio_data = torch.from_numpy(physio_data).float()
    emotion_labels = torch.from_numpy(emotion_labels)
    subject_ids = torch.from_numpy(subject_ids)
    valence = torch.from_numpy(valence).float()
    arousal = torch.from_numpy(arousal).float()
    video_ids = torch.from_numpy(video_ids)
    
    # Separate signals by type
    # PHYSIO_CHANNELS = ['acc_x', 'acc_y', 'acc_z', 'bvp', 'eda', 'skt', 'hr', 'ibi']
    # Indices: acc_x=0, acc_y=1, acc_z=2, bvp=3, eda=4, skt=5, hr=6, ibi=7
    
    # Accelerometer signals (3-axis)
    acc_data = physio_data[:, [0, 1, 2], :]  # ACC_X, ACC_Y, ACC_Z
    
    # Blood volume pulse
    bvp_data = physio_data[:, [3], :]  # BVP
    
    # Electrodermal activity
    eda_data = physio_data[:, [4], :]  # EDA
    
    # Skin temperature
    skt_data = physio_data[:, [5], :]  # SKT
    
    # Heart rate
    hr_data = physio_data[:, [6], :]  # HR
    
    # Inter-beat interval
    ibi_data = physio_data[:, [7], :]  # IBI
    
    # Channel-wise normalization: Normalize each signal type by the magnitude of the first sample
    print("Applying channel-wise magnitude normalization...")
    
    # For CEAP, the data is already normalized per subject, but we'll apply additional normalization
    acc_max = acc_data[0].abs().max()
    bvp_max = bvp_data[0].abs().max()
    eda_max = eda_data[0].abs().max()
    skt_max = skt_data[0].abs().max()
    hr_max = hr_data[0].abs().max()
    ibi_max = ibi_data[0].abs().max()
    
    # Avoid division by zero
    acc_max = max(acc_max, 1e-6)
    bvp_max = max(bvp_max, 1e-6)
    eda_max = max(eda_max, 1e-6)
    skt_max = max(skt_max, 1e-6)
    hr_max = max(hr_max, 1e-6)
    ibi_max = max(ibi_max, 1e-6)
    
    acc_data = acc_data / acc_max
    bvp_data = bvp_data / bvp_max
    eda_data = eda_data / eda_max
    skt_data = skt_data / skt_max
    hr_data = hr_data / hr_max
    ibi_data = ibi_data / ibi_max
    
    if filter_outliers:
        print("Applying outlier filtering...")
        # Remove outlier samples that have magnitude larger than 5 or smaller than -5
        
        valid_samples = torch.arange(len(acc_data))
        
        # Check each signal type for outliers
        for data_type, data_tensor in [
            ('ACC', acc_data), ('BVP', bvp_data), ('EDA', eda_data),
            ('SKT', skt_data), ('HR', hr_data), ('IBI', ibi_data)
        ]:
            # Check for outliers > 5
            outliers_high = (data_tensor > 5).sum(1).sum(1)
            valid_high = torch.where(outliers_high < 1)[0]
            valid_samples = torch.tensor([s for s in valid_samples if s in valid_high])
            
            # Check for outliers < -5
            outliers_low = (data_tensor < -5).sum(1).sum(1)
            valid_low = torch.where(outliers_low < 1)[0]
            valid_samples = torch.tensor([s for s in valid_samples if s in valid_low])
        
        # Filter all data
        acc_data_filtered = acc_data[valid_samples]
        bvp_data_filtered = bvp_data[valid_samples]
        eda_data_filtered = eda_data[valid_samples]
        skt_data_filtered = skt_data[valid_samples]
        hr_data_filtered = hr_data[valid_samples]
        ibi_data_filtered = ibi_data[valid_samples]
        emotion_labels_filtered = emotion_labels[valid_samples]
        subject_ids_filtered = subject_ids[valid_samples]
        valence_filtered = valence[valid_samples]
        arousal_filtered = arousal[valid_samples]
        video_ids_filtered = video_ids[valid_samples]
        
        print(f"Removed {len(physio_data) - len(acc_data_filtered)} outlier samples")
    else:
        print("Skipping outlier filtering...")
        acc_data_filtered = acc_data
        bvp_data_filtered = bvp_data
        eda_data_filtered = eda_data
        skt_data_filtered = skt_data
        hr_data_filtered = hr_data
        ibi_data_filtered = ibi_data
        emotion_labels_filtered = emotion_labels
        subject_ids_filtered = subject_ids
        valence_filtered = valence
        arousal_filtered = arousal
        video_ids_filtered = video_ids
    
    return (acc_data_filtered, bvp_data_filtered, eda_data_filtered, skt_data_filtered, 
            hr_data_filtered, ibi_data_filtered, emotion_labels_filtered, subject_ids_filtered, 
            valence_filtered, arousal_filtered, video_ids_filtered)

# Main execution
if __name__ == "__main__":
    print("Starting preprocessing for CEAP dataset...")
    
    print(f"Outlier filtering: {'ENABLED' if FILTER_OUTLIERS else 'DISABLED'}")
    print(f"Window size: {WINDOW_SIZE_SECONDS}s ({WINDOW_SIZE_FRAMES} frames)")
    print(f"Window stride: {WINDOW_STRIDE_SECONDS}s ({WINDOW_STRIDE_FRAMES} frames)")
    
    # Load and window the data
    physio_data, emotion_labels, subject_ids, valence, arousal, video_ids = load_ceap_data()
    
    # Preprocess the windowed data
    (acc_data_filtered, bvp_data_filtered, eda_data_filtered, skt_data_filtered, 
     hr_data_filtered, ibi_data_filtered, emotion_labels_filtered, subject_ids_filtered, 
     valence_filtered, arousal_filtered, video_ids_filtered) = preprocess_data(
        physio_data, emotion_labels, subject_ids, valence, arousal, video_ids, filter_outliers=FILTER_OUTLIERS
    )
    
    # Save processed data
    print(f"Saving {len(acc_data_filtered)} valid samples...")
    np.save(f'{DATA_DIR}/subject.npy', subject_ids_filtered.numpy())
    np.save(f'{DATA_DIR}/video.npy', video_ids_filtered.numpy())
    np.save(f'{DATA_DIR}/emotion_labels.npy', emotion_labels_filtered.numpy())
    np.save(f'{DATA_DIR}/valence.npy', valence_filtered.numpy())
    np.save(f'{DATA_DIR}/arousal.npy', arousal_filtered.numpy())
    
    # Save different signal types to separate files
    np.save(f'{DATA_DIR}/acc_data.npy', acc_data_filtered.numpy())      # Accelerometer (3-axis)
    np.save(f'{DATA_DIR}/bvp_data.npy', bvp_data_filtered.numpy())      # Blood volume pulse
    np.save(f'{DATA_DIR}/eda_data.npy', eda_data_filtered.numpy())      # Electrodermal activity
    np.save(f'{DATA_DIR}/skt_data.npy', skt_data_filtered.numpy())      # Skin temperature
    np.save(f'{DATA_DIR}/hr_data.npy', hr_data_filtered.numpy())        # Heart rate
    np.save(f'{DATA_DIR}/ibi_data.npy', ibi_data_filtered.numpy())      # Inter-beat interval
    
    # Save channel names for reference
    with open(f'{DATA_DIR}/channel_names.txt', 'w') as f:
        f.write("Accelerometer signals (acc_data.npy):\n")
        f.write("  0: acc_x\n")
        f.write("  1: acc_y\n")
        f.write("  2: acc_z\n")
        f.write("\nBlood volume pulse (bvp_data.npy):\n")
        f.write("  0: bvp\n")
        f.write("\nElectrodermal activity (eda_data.npy):\n")
        f.write("  0: eda\n")
        f.write("\nSkin temperature (skt_data.npy):\n")
        f.write("  0: skt\n")
        f.write("\nHeart rate (hr_data.npy):\n")
        f.write("  0: hr\n")
        f.write("\nInter-beat interval (ibi_data.npy):\n")
        f.write("  0: ibi\n")
    
    print("Preprocessing complete!")
    print(f"Accelerometer data shape: {acc_data_filtered.shape}")
    print(f"BVP data shape: {bvp_data_filtered.shape}")
    print(f"EDA data shape: {eda_data_filtered.shape}")
    print(f"SKT data shape: {skt_data_filtered.shape}")
    print(f"HR data shape: {hr_data_filtered.shape}")
    print(f"IBI data shape: {ibi_data_filtered.shape}")
    print(f"Emotion labels shape: {emotion_labels_filtered.shape}")
    
    # Summary of emotion labels
    unique_emotions, emotion_counts = np.unique(emotion_labels_filtered.numpy(), return_counts=True)
    emotion_names = ['LAHV', 'HALV', 'LALV', 'HAHV']
    
    print("\n" + "="*50)
    print("EMOTION DISTRIBUTION")
    print("="*50)
    for emotion_idx, count in zip(unique_emotions, emotion_counts):
        percentage = (count / len(emotion_labels_filtered)) * 100
        print(f"{emotion_names[emotion_idx]}: {count} samples ({percentage:.1f}%)")
    
    # Summary of subjects and videos in processed dataset
    unique_subjects = np.unique(subject_ids_filtered.numpy())
    unique_videos = np.unique(video_ids_filtered.numpy())
    total_subjects_kept = len(unique_subjects)
    total_videos_kept = len(unique_videos)
    
    # Convert zero-indexed back to one-indexed for display
    subject_ids_display = sorted(unique_subjects + 1)
    video_ids_display = sorted(unique_videos + 1)
    
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    print(f"Subjects kept in processed dataset: {total_subjects_kept}")
    print(f"Videos kept in processed dataset: {total_videos_kept}")
    print(f"Total samples after preprocessing: {len(acc_data_filtered)}")
    print(f"Average samples per subject: {len(acc_data_filtered)/total_subjects_kept:.1f}")
    print(f"Average samples per video: {len(acc_data_filtered)/total_videos_kept:.1f}")
    print("\nSubject IDs in processed dataset:")
    print(f"  {subject_ids_display}")
    print("\nVideo IDs in processed dataset:")
    print(f"  {video_ids_display}")
    print("="*50) 

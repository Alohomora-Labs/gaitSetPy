"""
UrFall Dataset Example

This example demonstrates how to use the UrFall dataset loader to:
1. Load pre-extracted features from CSV files
2. Create sliding windows
3. Access different data types (depth, RGB, accelerometer, etc.)

UrFall is a fall detection dataset from the University of Rzeszow with:
- 30 fall sequences (fall-01 to fall-30)
- 20 ADL (Activities of Daily Living) sequences (adl-01 to adl-20)
- Multiple data modalities: Depth, RGB, Accelerometer, Synchronization, Video

Reference: https://fenix.ur.edu.pl/~mkepski/ds/uf.html
"""

import os
from gaitsetpy.dataset import UrFallLoader, get_dataset_manager
from gaitsetpy.dataset.utils import download_urfall_data
from gaitsetpy.features import GaitFeatureExtractor



def main():
    # Create a data directory (this would be where you download the dataset)
    data_dir = "./urfall_data"
    os.makedirs(data_dir, exist_ok=True)
    
    print("=" * 80)
    print("UrFall Dataset Loader Example")
    print("=" * 80)
    
    # Method 1: Using the UrFallLoader class directly
    print("\n1. Creating UrFall loader instance...")
    loader = UrFallLoader()
    
    # Display loader metadata
    print(f"   Dataset: {loader.name}")
    print(f"   Description: {loader.description}")
    print(f"   Supported data types: {loader.metadata['data_types']}")
    print(f"   Camera: {loader.metadata['camera']}")
    print(f"   Sampling frequency (depth/RGB): {loader.metadata['sampling_frequency']} Hz")
    
    # Display activity information
    print("\n2. Activity labels:")
    activities = loader.get_activity_info()
    for act_id, act_label in activities.items():
        print(f"   {act_id}: {act_label}")
    
    # Display feature information
    print("\n3. Pre-extracted features from depth maps:")
    features = loader.get_feature_info()
    for feature_name, description in features.items():
        print(f"   {feature_name}: {description}")
    
    # Download and load data (real, no simulation)
    print("\n4. Downloading UrFall feature CSVs (if missing)...")
    download_urfall_data(
        data_dir,
        data_types=['features'],
        use_falls=True,
        use_adls=True,
    )
    
    print("\n5. Loading features into DataFrames...")
    data, names = loader.load_data(
        data_dir,
        data_types=['features'],
        use_falls=True,
        use_adls=True,
    )
    print(f"   Loaded {len(data)} DataFrame(s): {names}")
    for idx, (df, name) in enumerate(zip(data, names), 1):
        print(f"   [{idx}] {name}: shape={df.shape}")
        # Show a small preview
        try:
            print(df.head(3).to_string(index=False))
        except Exception:
            pass
    
    # Example: Creating sliding windows on real data
    print("\n6. Creating sliding windows on loaded data...")
    windows = loader.create_sliding_windows(data, names, window_size=30, step_size=15)
    for w in windows:
        series = {entry['name']: (len(entry['data']) if hasattr(entry['data'], '__len__') else 'N/A') for entry in w['windows']}
        print(f"   Windows for {w['name']}: { {k: v for k, v in list(series.items())[:5]} } ...")
    
    # RAW DATA WORKFLOW: Download a small subset of accelerometer CSVs and extract features
    print("\n7. Downloading raw accelerometer CSVs for a small subset (fall-01, adl-01)...")
    raw_sequences = ['fall-01', 'adl-01']
    download_urfall_data(
        data_dir,
        sequences=raw_sequences,
        data_types=['accelerometer', 'synchronization'],
        use_falls=True,
        use_adls=True,
    )
    
    print("\n8. Loading raw accelerometer data into DataFrames...")
    acc_data, acc_names = loader.load_data(
        data_dir,
        data_types=['accelerometer'],
        sequences=raw_sequences,
        use_falls=True,
        use_adls=True,
    )
    print(f"   Loaded {len(acc_data)} accelerometer DataFrame(s): {acc_names}")
    for idx, (df, name) in enumerate(zip(acc_data, acc_names), 1):
        print(f"   [{idx}] {name}: shape={df.shape}")
        try:
            print(df.head(3).to_string(index=False))
        except Exception:
            pass
    
    print("\n9. Creating sliding windows for accelerometer data...")
    acc_windows = loader.create_sliding_windows(acc_data, acc_names, window_size=100, step_size=50)
    # Determine sampling frequency for accelerometer
    fs_acc = loader.metadata.get('accelerometer_frequency', 100)
    print(f"   Using accelerometer sampling frequency: {fs_acc} Hz")
    for w in acc_windows:
        series = {entry['name']: (len(entry['data']) if hasattr(entry['data'], '__len__') else 'N/A') for entry in w['windows']}
        print(f"   Windows for {w['name']}: { {k: v for k, v in list(series.items())[:5]} } ...")
    
    print("\n10. Extracting features from accelerometer windows (GaitFeatureExtractor)...")
    extractor = GaitFeatureExtractor(verbose=False)
    # Flatten windows across DataFrames for feature extraction
    flattened_sensor_windows = []
    for w in acc_windows:
        for entry in w['windows']:
            if entry['name'] in ['labels', 'activity_id']:
                continue
            flattened_sensor_windows.append(entry)
    extracted = extractor.extract_features(flattened_sensor_windows, fs=fs_acc)
    print(f"   Extracted features for {len(extracted)} sensor(s)/channels")
    if extracted:
        first = extracted[0]
        print(f"   Example feature keys: {list(first.get('features', {}).keys())[:8]}")
    
    # Method 2: Using the DatasetManager (real usage)
    print("\n11. Using the DatasetManager:")
    manager = get_dataset_manager()
    print(f"   Available datasets: {manager.get_available_components()}")
    urfall_loader = manager.create_instance("urfall")
    print(f"   Created loader via manager: {urfall_loader.name}")
    # Use the manager-created loader to load a subset (e.g., only ADLs) to demonstrate real usage
    _subset_data, _subset_names = urfall_loader.load_data(
        data_dir,
        data_types=['features'],
        use_falls=False,
        use_adls=True,
    )
    print(f"   Subset load via manager: {len(_subset_data)} DataFrame(s): {_subset_names}")
    if _subset_data:
        print(f"   First subset DF shape: {_subset_data[0].shape}")
    
    # Feature columns
    print("\n12. Feature columns in pre-extracted CSV files:")
    for i, col in enumerate(loader.metadata['feature_columns'], 1):
        print(f"   {i}. {col}")
    
    print("\n" + "=" * 80)
    print("Example completed!")
    print("=" * 80)
    
    # Demonstrate file paths method (for modalities stored as files)
    print("\n13. Getting file paths for image/video data (if present):")
    for dtype in ['video', 'depth', 'rgb']:
        paths = loader.get_file_paths(data_dir, dtype)
        print(f"   {dtype}: {len(paths)} file(s) found")


if __name__ == "__main__":
    main()

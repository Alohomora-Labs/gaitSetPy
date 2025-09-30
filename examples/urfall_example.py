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
import zipfile
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from gaitsetpy.preprocessing import create_preprocessing_pipeline
from gaitsetpy.eda import SensorStatisticsAnalyzer



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
    
    # Download and analyze depth/RGB/video media for the same small subset
    print("\n11. Downloading raw depth/RGB/video for subset (fall-01, adl-01) with concurrency...")
    download_urfall_data(
        data_dir,
        sequences=raw_sequences,
        data_types=['depth', 'rgb', 'video'],
        use_falls=True,
        use_adls=True,
        max_workers=6,
    )
    
    print("\n12. Basic analysis of depth/RGB archives (without full extraction)...")
    depth_paths = loader.get_file_paths(data_dir, 'depth', sequences=raw_sequences)
    rgb_paths = loader.get_file_paths(data_dir, 'rgb', sequences=raw_sequences)
    for label, paths in [("depth", depth_paths), ("rgb", rgb_paths)]:
        for seq, path in list(paths.items()):
            try:
                with zipfile.ZipFile(path, 'r') as zf:
                    png_members = [n for n in zf.namelist() if n.lower().endswith('.png')]
                    print(f"   {label.upper()} {seq}: {len(png_members)} frame(s) in archive; sample: {png_members[:3]}")
                    # Try reading one frame to get dimensions
                    if png_members:
                        with zf.open(png_members[0]) as img_file:
                            import io
                            data_bytes = img_file.read()
                            arr = plt.imread(io.BytesIO(data_bytes))
                            if hasattr(arr, 'shape'):
                                print(f"      First frame shape: {arr.shape}")
            except Exception as e:
                print(f"   Warning: Could not analyze {path}: {e}")
    
    print("\n13. Basic analysis of video files (metadata if available)...")
    video_paths = loader.get_file_paths(data_dir, 'video', sequences=raw_sequences)
    for seq, path in video_paths.items():
        try:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"   VIDEO {seq}: size ~ {size_mb:.2f} MB")
            # Try to read metadata using OpenCV if installed
            try:
                import cv2  # type: ignore
                cap = cv2.VideoCapture(path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                    print(f"      {width}x{height} @ {fps:.2f} FPS, frames: {frame_count}")
                cap.release()
            except Exception:
                pass
        except Exception as e:
            print(f"   Warning: Could not analyze video {path}: {e}")
    
    # 14. Feature extraction for depth/RGB/video (quick, lightweight)
    print("\n14. Extracting simple features from depth/RGB/video (first few frames)...")
    from gaitsetpy.features import UrFallMediaFeatureExtractor
    media_extractor = UrFallMediaFeatureExtractor(verbose=False)
    
    # Build pseudo-windows by sampling a few frames from each archive / video
    def build_windows_from_zip(paths: dict, max_frames: int = 12, grayscale_hint: bool = False):
        import io
        windows = []
        for seq, path in paths.items():
            try:
                with zipfile.ZipFile(path, 'r') as zf:
                    png_members = [n for n in zf.namelist() if n.lower().endswith('.png')]
                    png_members = png_members[:max_frames]
                    frames = []
                    for name in png_members:
                        with zf.open(name) as f:
                            arr = plt.imread(io.BytesIO(f.read()))
                            frames.append(arr)
                    if frames:
                        windows.append({'name': seq, 'data': frames})
            except Exception as e:
                print(f"   Warn: could not read {path}: {e}")
        feats = media_extractor.extract_features(windows, fs=loader.metadata['sampling_frequency'], grayscale=grayscale_hint)
        # Convert to mapping seq->flat feature dict
        out = {}
        for f in feats:
            out[f['name']] = {k: v for k, v in f.get('features', {}).items()}
        return out
    
    def build_windows_from_video(paths: dict, max_frames: int = 150):
        feats = {}
        try:
            import cv2  # type: ignore
            windows = []
            for seq, path in paths.items():
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    continue
                frames = []
                count = 0
                while count < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                    count += 1
                cap.release()
                if frames:
                    windows.append({'name': seq, 'data': frames})
            feats_list = media_extractor.extract_features(windows, fs=loader.metadata['sampling_frequency'], grayscale=True)
            for f in feats_list:
                feats[f['name']] = {k: v for k, v in f.get('features', {}).items()}
        except Exception as e:
            print(f"   Warn: OpenCV not available or video processing failed: {e}")
        return feats
    
    depth_feats = build_windows_from_zip(depth_paths, grayscale_hint=True)
    rgb_feats = build_windows_from_zip(rgb_paths, grayscale_hint=False)
    video_feats = build_windows_from_video(video_paths)
    print(f"   Extracted {len(depth_feats)} depth, {len(rgb_feats)} rgb, {len(video_feats)} video feature sets")
    
    # 15. Build a simple classification dataset from accelerometer windows and media features
    print("\n15. Building classification dataset (fall vs adl)...")
    X_list = []
    y_list = []
    # From accelerometer windows: compute simple stats per window and majority label from activity_id
    for item in acc_windows:
        # Collect per-window vectors across columns for this DataFrame
        # Build a dict name->list_of_windows
        per_name = {entry['name']: entry['data'] for entry in item['windows']}
        if 'activity_id' in per_name:
            labels_windows = per_name['activity_id']  # list of arrays
            # Determine majority label per window
            window_labels = []
            for wlab in labels_windows:
                vals, counts = np.unique(wlab, return_counts=True)
                window_labels.append(int(vals[np.argmax(counts)]))
        else:
            continue
        # For each feature column (exclude metadata names)
        for name_key, windows_arr in per_name.items():
            if name_key in ['labels', 'activity_id']:
                continue
            for idx, w in enumerate(windows_arr):
                # Ensure w is a 1D sequence
                arr_w = np.ravel(np.array(w, dtype=np.float32))
                vec = [float(np.mean(arr_w)), float(np.std(arr_w)), float(np.max(arr_w)-np.min(arr_w))]
                X_list.append(vec)
                y_list.append(window_labels[idx] if idx < len(window_labels) else 0)
    # From media features: aggregate per sequence (ensure consistent vector sizes)
    def label_from_seq(seq: str) -> int:
        return 1 if seq.startswith('fall-') else 0
    for seq, feat in depth_feats.items():
        X_list.append([feat.get('mean_intensity', 0.0), feat.get('std_intensity', 0.0)])
        y_list.append(label_from_seq(seq))
    for seq, feat in rgb_feats.items():
        X_list.append([feat.get('mean_intensity', 0.0), feat.get('std_intensity', 0.0)])
        y_list.append(label_from_seq(seq))
    for seq, feat in video_feats.items():
        X_list.append([feat.get('motion_mean', 0.0), feat.get('motion_std', 0.0), feat.get('brightness_mean', 0.0)])
        y_list.append(label_from_seq(seq))
    
    # Pad/truncate to uniform dimensionality
    max_dim = max(len(v) for v in X_list) if X_list else 0
    def pad_vec(v, d):
        if len(v) == d:
            return v
        if len(v) > d:
            return v[:d]
        return v + [0.0] * (d - len(v))
    X_list = [pad_vec(list(map(float, v)), max_dim) for v in X_list]
    
    X = np.array(X_list, dtype=np.float32) if X_list else np.empty((0, 0), dtype=np.float32)
    y = np.array(y_list, dtype=np.int64) if y_list else np.empty((0,), dtype=np.int64)
    print(f"   Classification dataset: X={X.shape}, y={y.shape}, positive={(y==1).sum()} negatives={(y==0).sum()}")
    
    # 16. Preprocess features (clipping) and train a RandomForest
    print("\n16. Preprocessing features and training classifier...")
    if X.size == 0 or len(np.unique(y)) < 2 or len(y) < 4:
        print("   Not enough labeled samples for training. Skipping classification.")
    else:
        pipeline = create_preprocessing_pipeline(['clipping'], clipping={'min_val': float(np.min(X)), 'max_val': float(np.max(X))})
        X_proc = pipeline(X)
        strat = y if len(np.unique(y)) > 1 else None
        test_size = 0.3 if len(y) >= 10 else 0.25
        X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=test_size, random_state=42, stratify=strat)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred) if len(y_test) else float('nan')
        print(f"   RandomForest accuracy: {acc:.3f}")
    
    # 17. EDA: compute and print statistics on accelerometer DataFrames
    print("\n17. EDA on accelerometer data (basic statistics)...")
    try:
        analyzer = SensorStatisticsAnalyzer()
        stats = analyzer.analyze(acc_data)
        # Print a compact summary
        for k, v in list(stats.items())[:2]:
            print(f"   {k}: keys={list(v.keys())[:5]}")
    except Exception as e:
        print(f"   Warning: EDA analysis failed: {e}")
    
    # Method 2: Using the DatasetManager (real usage)
    print("\n18. Using the DatasetManager:")
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
    print("\n19. Feature columns in pre-extracted CSV files:")
    for i, col in enumerate(loader.metadata['feature_columns'], 1):
        print(f"   {i}. {col}")
    
    print("\n" + "=" * 80)
    print("Example completed!")
    print("=" * 80)
    
    # Demonstrate file paths method (for modalities stored as files)
    print("\n20. Re-checking file paths for image/video data (after download):")
    for dtype in ['video', 'depth', 'rgb']:
        paths = loader.get_file_paths(data_dir, dtype, sequences=raw_sequences)
        print(f"   {dtype}: {len(paths)} file(s) found")


if __name__ == "__main__":
    main()

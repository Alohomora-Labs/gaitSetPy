import os
from gaitsetpy.dataset.daphnet import load_daphnet_data, create_sliding_windows
from gaitsetpy.features.gait_features import extract_gait_features
from gaitsetpy.classification.models import get_classification_model
from gaitsetpy.classification.utils.preprocess import preprocess_features

# Set data directory (change as needed)
data_dir = os.path.join(os.path.dirname(__file__), '../data')

# 1. Load Daphnet data
data, names = load_daphnet_data(data_dir)
print(f"Loaded {len(data)} Daphnet subjects.")

# 2. Create sliding windows
windows = create_sliding_windows(data, names, window_size=192, step_size=32)
print(f"Created sliding windows for {len(windows)} subjects.")

# Flatten windows to a list of {name, data} dicts for all sensors across all subjects
sensor_windows = []
for subject in windows:
    for sensor in subject['windows']:
        sensor_windows.append(sensor)

# 3. Extract all features (time, frequency, statistical)
fs = 64  # Daphnet sampling frequency
features = extract_gait_features(sensor_windows, fs, time_domain=True, frequency_domain=True, statistical=True, verbose=True)
print(f"Extracted features for {len(features)} sensors.")

# 4. Prepare features for classification (group by sensor, keep lists)
feature_dicts = []
annotation_sensor = None
for sensor in features:
    if sensor['name'] == 'annotations':
        annotation_sensor = sensor
for sensor in features:
    if sensor['name'] != 'annotations':
        n_windows = len(next(iter(sensor['features'].values())))
        # Each feature is a list per window; keep as lists
        feature_dicts.append({
            'name': sensor['name'],
            'features': sensor['features'],
            'annotations': annotation_sensor['annotations'][:n_windows] if annotation_sensor else [0]*n_windows
        })

print(f"Prepared {len(feature_dicts)} feature dictionaries for classification.")

# 5. List of models to train
model_names = ['random_forest', 'mlp', 'lstm', 'bilstm', 'cnn']  # GNN requires adjacency matrix, skip for now

for model_name in model_names:
    print(f"\n=== Training model: {model_name.upper()} ===")
    model = get_classification_model(model_name)
    model.train(feature_dicts, test_size=0.2, validation_split=True)
    results = model.evaluate(feature_dicts, detailed_report=True)
    print(f"Results for {model_name}:\n", results)

# Note: GNN requires an adjacency matrix, which is not trivial to construct for this tabular data.
# If you have a graph structure, you can add 'gnn' to model_names and pass adjacency_matrix=... to train/evaluate. 
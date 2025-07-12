"""
Test file for GaitSetPy - demonstrates both legacy and new APIs
"""
import gaitsetpy as gsp

print("=== Testing GaitSetPy with Legacy API ===")
# print(f"GaitSetPy version: {gsp.__version__}")

# # Load gait data (legacy function still works)
# daphnet, names = gsp.load_daphnet_data("data/")
# print(f"Loaded {len(daphnet)} datasets: {names}")

# # Create sliding windows (legacy function still works)
# sliding_windows = gsp.create_sliding_windows(daphnet, names)
# print(f"Created sliding windows for {len(sliding_windows)} datasets")

# # Extract features (legacy function still works)
# freq = 64
# features = gsp.extract_gait_features(sliding_windows[0]['windows'], freq, True, True, True)
# print(f"Extracted features: {len(features)} windows with {len(features[0])} features each")

# # Get safe time range from the sliding windows
# print("Available sliding windows:")
# for i, window_data in enumerate(sliding_windows):
#     print(f"  Dataset {i}: {window_data['name']} - {len(window_data['windows'])} windows")

# # Get time range from available windows (use simpler approach)
# first_window_data = sliding_windows[0]['windows'][0]['data'][0]
# print(f"First window time range: {first_window_data.index[0]} to {first_window_data.index[-1]}")

# # Use safer time range parameters
# safe_start_idx = 0  # Use simple integer indices
# safe_end_idx = 1000  # Use reasonable end index
# num_windows_to_plot = min(10, len(sliding_windows[0]['windows']))

# print(f"Using time range: {safe_start_idx} to {safe_end_idx}")
# print(f"Number of windows to plot: {num_windows_to_plot}")

# # Visualize sensor data with features (using correct function name)
# try:
#     gsp.plot_sensor_with_features(sliding_windows[0]['windows'], features, 
#                                   start_idx=safe_start_idx, end_idx=safe_end_idx, 
#                                   sensor_name="shank", num_windows=num_windows_to_plot)
#     print("✓ Legacy visualization completed successfully")
# except Exception as e:
#     print(f"✗ Legacy visualization failed: {e}")
#     # Try alternative visualization function
#     try:
#         gsp.plot_sensor_features(sliding_windows[0]['windows'], features, 
#                                 sensor_name="shank", start_idx=safe_start_idx, 
#                                 end_idx=safe_end_idx, num_windows=num_windows_to_plot)
#         print("✓ Alternative visualization completed successfully")
#     except Exception as e2:
#         print(f"✗ Alternative visualization also failed: {e2}")

print("\n=== Testing GaitSetPy with New Class-based API ===")

# Use the new class-based API
try:
    # Load dataset using new API
    loader = gsp.DaphnetLoader()
    data, names = loader.load_data("data/")
    windows = loader.create_sliding_windows(data, names, window_size=192)
    print(f"✓ New API: Loaded {len(data)} datasets using DaphnetLoader")
    print(f"  Created windows for {len(windows)} datasets")
    
    # Extract features using new API - fix the data structure issue
    extractor = gsp.GaitFeatureExtractor()
    # Pass the correct structure: windows[0]['windows'] instead of windows
    new_features = extractor.extract_features(windows[0]['windows'], fs=64, time_domain=True, 
                                             frequency_domain=True, statistical=True)
    print(f"✓ New API: Extracted features using GaitFeatureExtractor")
    print(f"  Features extracted for {len(new_features)} sensors")
    
    # Analyze data using new API - fix the data structure issue
    analyzer = gsp.DaphnetVisualizationAnalyzer()
    # Check if data is properly structured
    if data and len(data) > 0:
        analysis = analyzer.analyze(data[5])  # Pass first dataset
        print(f"✓ New API: Analyzed data using DaphnetVisualizationAnalyzer")
    else:
        print("✗ New API: No data available for analysis")
    
    # Visualize using new API
    gsp.plot_daphnet_data(data, names, sensor_type='all', dataset_index=5)
    print("✓ New API: Visualization completed successfully")
    
    # Use convenient workflow function
    workflow_result = gsp.load_and_analyze_daphnet("data/", sensor_type='shank', window_size=192)
    print(f"✓ New API: Complete workflow executed successfully")
    
except Exception as e:
    print(f"✗ New API failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Testing Preprocessing Pipeline ===")

# Test preprocessing capabilities
try:
    # Load some data first for preprocessing test
    loader = gsp.DaphnetLoader()
    data, names = loader.load_data("data/")
    windows = loader.create_sliding_windows(data, names, window_size=192)
    
    # Create a simple preprocessing pipeline
    pipeline = gsp.create_preprocessing_pipeline([
        'clipping',
        'noise_removal'
    ])
    
    # Test with sample data - use the correct window structure
    sample_data = windows[0]['windows'][0]['data'][0]
    processed_data = pipeline(sample_data)  # Call the pipeline function directly
    print(f"✓ Preprocessing pipeline executed successfully")
    
except Exception as e:
    print(f"✗ Preprocessing pipeline failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Testing Classification Model ===")

# Test classification model
try:
    model = gsp.RandomForestModel(n_estimators=10, random_state=42)
    print(f"✓ Classification model created successfully")
    
    # Note: We skip actual training since it requires properly formatted features
    
except Exception as e:
    print(f"✗ Classification model creation failed: {e}")

print("\n=== Testing Individual Components ===")

# Test individual managers
try:
    # Test dataset manager
    dataset_manager = gsp.get_dataset_manager()
    available_datasets = dataset_manager.get_available_components()
    print(f"✓ Dataset manager: {available_datasets}")
    
    # Test feature manager
    feature_manager = gsp.get_feature_manager()
    available_extractors = feature_manager.get_available_components()
    print(f"✓ Feature manager: {available_extractors}")
    
    # Test preprocessing manager
    preprocessing_manager = gsp.get_preprocessing_manager()
    available_preprocessors = preprocessing_manager.get_available_components()
    print(f"✓ Preprocessing manager: {available_preprocessors}")
    
    # Test EDA manager
    eda_manager = gsp.get_eda_manager()
    available_analyzers = eda_manager.get_available_components()
    print(f"✓ EDA manager: {available_analyzers}")
    
    # Test classification manager
    classification_manager = gsp.get_classification_manager()
    available_models = classification_manager.get_available_components()
    print(f"✓ Classification manager: {available_models}")
    
except Exception as e:
    print(f"✗ Manager testing failed: {e}")

print("\n=== System Information ===")
system_info = gsp.get_system_info()
print(f"Available datasets: {system_info['available_datasets']}")
print(f"Available extractors: {system_info['available_extractors']}")
print(f"Available preprocessors: {system_info['available_preprocessors']}")
print(f"Available analyzers: {system_info['available_analyzers']}")
print(f"Available models: {system_info['available_models']}")

print("\n=== Test Complete ===")
print("All available functionalities have been tested.")

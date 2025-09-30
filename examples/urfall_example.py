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
    
    # Note: To actually load data, you would need to download the dataset first
    # Here we show the interface without actual data
    print("\n4. Loading data interface:")
    print("   To load pre-extracted features:")
    print("   data, names = loader.load_data(data_dir, data_types=['features'])")
    print("   ")
    print("   To load specific sequences:")
    print("   data, names = loader.load_data(data_dir,")
    print("                                   data_types=['features'],")
    print("                                   sequences=['fall-01', 'fall-02', 'adl-01'])")
    print("   ")
    print("   To load only fall sequences:")
    print("   data, names = loader.load_data(data_dir,")
    print("                                   data_types=['features'],")
    print("                                   use_falls=True, use_adls=False)")
    
    # Demonstrate different data types
    print("\n5. Available data types and their files:")
    print("   - features: Pre-extracted CSV features")
    print("     * urfall-cam0-falls.csv (fall sequences)")
    print("     * urfall-cam0-adls.csv (ADL sequences)")
    print("   ")
    print("   - depth: Depth camera images (ZIP files)")
    print("     * fall-XX-cam0-d.zip (where XX = 01 to 30)")
    print("     * adl-XX-cam0-d.zip (where XX = 01 to 20)")
    print("   ")
    print("   - rgb: RGB camera images (ZIP files)")
    print("     * fall-XX-cam0-rgb.zip")
    print("     * adl-XX-cam0-rgb.zip")
    print("   ")
    print("   - accelerometer: Accelerometer CSV data")
    print("     * fall-XX-cam0-acc.csv")
    print("     * adl-XX-cam0-acc.csv")
    print("   ")
    print("   - synchronization: Synchronization CSV data")
    print("     * fall-XX-cam0-sync.csv")
    print("     * adl-XX-cam0-sync.csv")
    print("   ")
    print("   - video: Video files (MP4)")
    print("     * fall-XX-cam0.mp4")
    print("     * adl-XX-cam0.mp4")
    
    # Method 2: Using the DatasetManager
    print("\n6. Using the DatasetManager:")
    manager = get_dataset_manager()
    print(f"   Available datasets: {manager.get_available_components()}")
    
    # Create instance through manager
    urfall_loader = manager.create_instance("urfall")
    print(f"   Created loader: {urfall_loader.name}")
    
    # Example: Creating sliding windows (with hypothetical data)
    print("\n7. Creating sliding windows:")
    print("   windows = loader.create_sliding_windows(data, names,")
    print("                                            window_size=30,")
    print("                                            step_size=15)")
    print("   ")
    print("   Window parameters:")
    print("   - window_size: Number of frames per window (default: 30)")
    print("   - step_size: Step between windows (default: 15)")
    
    # Downloading data
    print("\n8. Downloading data:")
    print("   The dataset can be downloaded from:")
    print("   https://fenix.ur.edu.pl/~mkepski/ds/uf.html")
    print("   ")
    print("   Example download using the utility:")
    print("   from gaitsetpy.dataset.utils import download_urfall_data")
    print("   download_urfall_data(data_dir,")
    print("                        data_types=['features'],")
    print("                        use_falls=True, use_adls=True)")
    
    # Feature columns
    print("\n9. Feature columns in pre-extracted CSV files:")
    for i, col in enumerate(loader.metadata['feature_columns'], 1):
        print(f"   {i}. {col}")
    
    print("\n" + "=" * 80)
    print("Example completed!")
    print("=" * 80)
    
    # Demonstrate file paths method
    print("\n10. Getting file paths for image/video data:")
    print("    paths = loader.get_file_paths(data_dir, 'video')")
    print("    paths = loader.get_file_paths(data_dir, 'depth', sequences=['fall-01'])")
    print("    paths = loader.get_file_paths(data_dir, 'rgb', use_falls=True, use_adls=False)")
    
    print("\nFor complete usage, see the UrFall dataset documentation.")


if __name__ == "__main__":
    main()

# UrFall Dataset Support - Implementation Summary

## Overview
This document summarizes the implementation of UrFall dataset support in GaitSetPy.

## What Was Implemented

### 1. Core Dataset Loader (`gaitsetpy/dataset/urfall.py`)
- **UrFallLoader class**: Inherits from `BaseDatasetLoader`
- Supports multiple data modalities:
  - Pre-extracted features (CSV)
  - Depth camera images (ZIP archives)
  - RGB camera images (ZIP archives)
  - Accelerometer data (CSV)
  - Synchronization data (CSV)
  - Video files (MP4)

### 2. Dataset Coverage
- **Fall sequences**: 30 sequences (fall-01 to fall-30)
- **ADL sequences**: 20 sequences (adl-01 to adl-20)
- **Total**: 50 sequences with multiple modalities each

### 3. Feature Support
The pre-extracted features from depth maps include 11 columns:
1. `sequence_name` - Camera name omitted (e.g., 'fall-01', 'adl-01')
2. `frame_number` - Frame number in sequence
3. `label` - Human posture label (-1: not lying, 0: falling, 1: lying)
4. `HeightWidthRatio` - Bounding box height to width ratio
5. `MajorMinorRatio` - Major to minor axis ratio from BLOB segmentation
6. `BoundingBoxOccupancy` - Ratio of bounding box occupied by person pixels
7. `MaxStdXZ` - Standard deviation of pixels from centroid (X and Z axes)
8. `HHmaxRatio` - Human height in frame to standing height ratio
9. `H` - Actual height in mm
10. `D` - Distance of person center to floor in mm
11. `P40` - Ratio of point clouds in 40cm cuboid to full height cuboid

### 4. Key Methods

#### `load_data(data_dir, data_types=None, sequences=None, use_falls=True, use_adls=True)`
- Loads data from the specified directory
- Configurable data types to load
- Filter by specific sequences or all sequences
- Separate control for fall vs ADL sequences

#### `create_sliding_windows(data, names, window_size=30, step_size=15)`
- Creates sliding windows from loaded data
- Default: 30 frames (1 second at 30Hz) with 50% overlap
- Supports majority voting for labels

#### `get_file_paths(data_dir, data_type, sequences=None, use_falls=True, use_adls=True)`
- Returns file paths for image/video data types
- Useful for loading raw depth, RGB, or video files

### 5. Download and Extraction Utilities (`gaitsetpy/dataset/utils.py`)

#### `download_urfall_data(data_dir, sequences=None, data_types=None, use_falls=True, use_adls=True)`
- Downloads dataset files from https://fenix.ur.edu.pl/~mkepski/ds/data/
- Supports selective download of specific sequences and data types
- Shows progress bars during download

#### `extract_urfall_data(data_dir, sequences=None, use_falls=True, use_adls=True)`
- Extracts ZIP archives for depth and RGB data
- Selective extraction based on sequences

### 6. Integration
- Registered with `DatasetManager` as "urfall"
- Accessible via `get_dataset_manager()` and `load_dataset()` functions
- Follows same patterns as other datasets (Daphnet, HAR-UP, PhysioNet)

### 7. Testing (`tests/test_urfall.py`)
Created comprehensive test suite with 13 tests covering:
- Loader instantiation and metadata
- Supported formats and sensor information
- Activity and feature information
- Sliding window creation
- Feature loading with mock CSV files
- Filtering (falls only, ADLs only)
- File path retrieval
- Invalid data type handling
- Legacy function wrappers

**All 13 tests pass successfully** ✓

### 8. Example Code (`examples/urfall_example.py`)
Comprehensive example demonstrating:
- Creating loader instance
- Accessing metadata and information
- Loading different data types
- Creating sliding windows
- Using DatasetManager
- Downloading data

### 9. Documentation
- Updated README.md with UrFall dataset information
- Changed status badge from "In Progress" to "Supported"
- Added usage examples for UrFall dataset
- Documented all features and capabilities

## Usage Examples

### Basic Usage
```python
from gaitsetpy.dataset import UrFallLoader

loader = UrFallLoader()
data, names = loader.load_data("data/urfall", data_types=['features'])
windows = loader.create_sliding_windows(data, names, window_size=30, step_size=15)
```

### Loading Specific Sequences
```python
data, names = loader.load_data(
    "data/urfall",
    data_types=['features', 'accelerometer'],
    sequences=['fall-01', 'fall-02', 'adl-01']
)
```

### Loading Only Falls
```python
fall_data, fall_names = loader.load_data(
    "data/urfall",
    data_types=['features'],
    use_falls=True,
    use_adls=False
)
```

### Getting Video File Paths
```python
video_paths = loader.get_file_paths("data/urfall", 'video')
depth_paths = loader.get_file_paths("data/urfall", 'depth', sequences=['fall-01'])
```

## File Structure
```
gaitsetpy/dataset/
├── urfall.py              # Main loader implementation
├── utils.py               # Updated with download/extract functions
└── __init__.py           # Updated with UrFall registration

tests/
└── test_urfall.py        # Comprehensive test suite

examples/
└── urfall_example.py     # Usage example

README.md                 # Updated documentation
```

## Data Types and File Naming Convention

| Data Type | File Pattern | Description |
|-----------|-------------|-------------|
| features | `urfall-cam0-falls.csv`, `urfall-cam0-adls.csv` | Pre-extracted features |
| depth | `{seq}-cam0-d.zip` | Depth camera images |
| rgb | `{seq}-cam0-rgb.zip` | RGB camera images |
| accelerometer | `{seq}-cam0-acc.csv` | Accelerometer data |
| synchronization | `{seq}-cam0-sync.csv` | Synchronization data |
| video | `{seq}-cam0.mp4` | Video files |

Where `{seq}` is `fall-01` to `fall-30` or `adl-01` to `adl-20`.

## Dataset Reference
- **Website**: https://fenix.ur.edu.pl/~mkepski/ds/uf.html
- **Name**: University of Rzeszow Fall Detection Dataset
- **Camera**: Front camera (cam0)
- **Sampling Rate**: 30 Hz (depth/RGB), ~100 Hz (accelerometer)

## Testing Results
- **Unit tests**: 13/13 passed ✓
- **Integration tests**: All existing tests still pass (256/256) ✓
- **Import verification**: Successful ✓
- **DatasetManager registration**: Successful ✓

## Future Enhancements
- Add support for actual depth/RGB image loading (currently only paths)
- Add specialized feature extractors for UrFall data
- Add integration with classification models
- Add visualization utilities for fall detection sequences
- Add data augmentation techniques for fall detection

## Compliance
- ✓ Follows BaseDatasetLoader interface
- ✓ Registered with DatasetManager
- ✓ Includes comprehensive tests
- ✓ Documented in README
- ✓ Follows repository patterns and conventions
- ✓ Backward compatible with existing code
- ✓ All existing tests still pass

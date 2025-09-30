# GaitSetPy
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15881527.svg)](https://doi.org/10.5281/zenodo.15881527) [![PyPI version](https://badge.fury.io/py/gaitsetpy.svg)](https://pypi.org/project/gaitsetpy/) [![Docs](https://img.shields.io/badge/docs-gaitsetpy-lightgrey.svg)](https://alohomora-labs.github.io/gaitSetPy/gaitsetpy.html) ![CodeRabbit Pull Request Reviews](https://img.shields.io/coderabbit/prs/github/Alohomora-Labs/gaitSetPy?utm_source=oss&utm_medium=github&utm_campaign=Alohomora-Labs%2FgaitSetPy&labelColor=171717&color=FF570A&link=https%3A%2F%2Fcoderabbit.ai&label=CodeRabbit+Reviews)



GaitSetPy is a Python package for gait analysis and recognition. This package provides tools and algorithms to process and analyze gait data, enabling researchers and developers to build applications for gait recognition and clinical gait assessment.

## Features

- Gait data preprocessing
- Feature extraction
- Gait recognition algorithms
- Visualization tools

## Supported Datasets

### IMU Sensor Based
- Daphnet: [https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait](https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait) ![Supported](https://img.shields.io/badge/status-supported-brightgreen)
- MobiFall: [https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/](https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/) ![In Progress](https://img.shields.io/badge/status-in%20progress-yellow)

- HAR-UP (formerly UPFall): [https://sites.google.com/up.edu.mx/har-up/](https://sites.google.com/up.edu.mx/har-up/) ![Supported](https://img.shields.io/badge/status-supported-brightgreen)
- UrFall: [https://fenix.ur.edu.pl/~mkepski/ds/uf.html](https://fenix.ur.edu.pl/~mkepski/ds/uf.html) ![Supported](https://img.shields.io/badge/status-supported-brightgreen)
- Activity Net - Arduous : [https://www.mad.tf.fau.de/research/activitynet/wearable-multi-sensor-gait-based-daily-activity-data/](https://www.mad.tf.fau.de/research/activitynet/wearable-multi-sensor-gait-based-daily-activity-data/) ![In Progress](https://img.shields.io/badge/status-in%20progress-yellow)

### Pressure Sensor Based
- Physionet Gait in Parkinson's Disease: [https://physionet.org/content/gaitpdb/1.0.0/](https://physionet.org/content/gaitpdb/1.0.0/) ![Completed](https://img.shields.io/badge/status-completed-green)


## Installation

From PyPI:
```bash
pip install gaitsetpy
```

From source:
```bash
git clone https://github.com/Alohomora-Labs/gaitSetPy.git
python setup.py install
```

Optionally, also install requirements
``` bash
pip install -r requirements.txt
```

## Usage

### Daphnet Dataset (class-based API)

```python
import gaitsetpy as gsp

# Load dataset via class-based loader
loader = gsp.DaphnetLoader()
data, names = loader.load_data("data/daphnet")

# Create sliding windows and extract features
windows = loader.create_sliding_windows(data, names, window_size=192, step_size=32)
extractor = gsp.GaitFeatureExtractor()
features = extractor.extract_features(windows[0]['windows'], fs=64)

# Optional: visualize using analyzer
from gaitsetpy.eda import SensorStatisticsAnalyzer
analyzer = SensorStatisticsAnalyzer()
analyzer.visualize(
    windows[0]['windows'],
    features,
    sensor_name="shank",
    start_idx=0,
    end_idx=1000,
    num_windows=15
)
```

### HAR-UP Dataset (class-based API)

```python
import gaitsetpy as gsp

# Load HAR-UP data via class-based loader
loader = gsp.HARUPLoader()
harup_data, harup_names = loader.load_data("data/harup")

# Create sliding windows
window_size = 100  # 1 second at 100Hz
step_size = 50     # 0.5 second overlap
windows = loader.create_sliding_windows(harup_data, harup_names, window_size, step_size)

# Extract features using dataset-specific method or feature extractor
features_data = loader.extract_features(windows)
```
![alt text](image.png)

``` python
# Train and evaluate a Random Forest (class-based)
rf_model = gsp.RandomForestModel(n_estimators=50, random_state=42, max_depth=10)
rf_model.train(features_data if isinstance(features_data, list) else features)
metrics = rf_model.evaluate(features_data if isinstance(features_data, list) else features)
print(metrics.get('accuracy'))

# Load a saved model (optional)
rf_model.load_model("gaitsetpy/classification/weights/random_forest_model_40_10.pkl")
```

### UrFall Dataset (class-based API)

```python
import gaitsetpy as gsp

# Load UrFall data via class-based loader
loader = gsp.UrFallLoader()

# Load pre-extracted features (default)
data, names = loader.load_data("data/urfall", data_types=['features'])

# Or load specific data types
data, names = loader.load_data("data/urfall", 
                               data_types=['features', 'accelerometer'],
                               use_falls=True, 
                               use_adls=True)

# Load only fall sequences
fall_data, fall_names = loader.load_data("data/urfall",
                                         data_types=['features'],
                                         use_falls=True,
                                         use_adls=False)

# Create sliding windows from features
window_size = 30  # 30 frames at 30Hz = 1 second
step_size = 15    # 0.5 second overlap
windows = loader.create_sliding_windows(data, names, window_size, step_size)

# Get file paths for image/video data
video_paths = loader.get_file_paths("data/urfall", 'video')
depth_paths = loader.get_file_paths("data/urfall", 'depth', sequences=['fall-01'])

# Display dataset information
print(loader.get_activity_info())
print(loader.get_feature_info())
```

**UrFall Dataset Features:**
- 30 fall sequences and 20 ADL (Activities of Daily Living) sequences
- Multiple data modalities: depth images, RGB images, accelerometer, synchronization, video
- Pre-extracted features from depth maps (11 features per frame)
- Configurable data type loading
- Front camera (cam0) data
- Labels: -1 (standing/walking), 0 (falling - transient), 1 (lying on ground)

## Documentation

For detailed documentation and API reference, please visit the [official documentation](https://alohomora-labs.github.io/gaitSetPy/gaitsetpy.html).

## Contributing

We welcome contributions! Please read our [contributing guidelines](CONTRIBUTING.md) to get started.

## License

This project is licensed under the GNU GPL License. See the [LICENSE](LICENSE) file for more details.

## Contact

For any questions or inquiries, please contact us at [jayeeta.chakrabortyfcs@kiit.ac.in](mailto:jayeeta.chakrabortyfcs@kiit.ac.in) or [aharshit123456@gmail.com](mailto:aharshit123456@gmail.com).

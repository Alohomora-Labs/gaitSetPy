# GaitSetPy
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15881527.svg)](https://doi.org/10.5281/zenodo.15881527) [![PyPI version](https://badge.fury.io/py/gaitsetpy.svg)](https://pypi.org/project/gaitsetpy/) [![Docs](https://img.shields.io/badge/docs-gaitsetpy-lightgrey.svg)](https://alohomora-labs.github.io/gaitSetPy/gaitsetpy.html) ![CodeRabbit Pull Request Reviews](https://img.shields.io/coderabbit/prs/github/Alohomora-Labs/gaitSetPy?utm_source=oss&utm_medium=github&utm_campaign=Alohomora-Labs%2FgaitSetPy&labelColor=171717&color=FF570A&link=https%3A%2F%2Fcoderabbit.ai&label=CodeRabbit+Reviews) ![Conda-forge-downloads](https://anaconda.org/conda-forge/gaitsetpy/badges/downloads.svg) ![PyPI - Downloads](https://img.shields.io/pypi/dm/gaitsetpy?label=pypi-downloads)


GaitSetPy is a Python package for gait analysis and recognition. It provides clean, modern APIs to preprocess, analyze, extract features, classify, and visualize gait data across multiple datasets and modalities.

---

## 📚 Table of Contents
- [Features](#features)
- [Supported Datasets](#supported-datasets)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Examples: Notebooks and Scripts](#examples-notebooks-and-scripts)
- [Usage by Dataset (Code Snippets)](#usage-by-dataset-code-snippets)
  - [Daphnet](#daphnet)
  - [HAR-UP](#har-up)
  - [UrFall](#urfall)
  - [PhysioNet (VGRF)](#physionet-vgrf)
  - [MobiFall](#mobifall)
  - [Arduous](#arduous)
- [Models](#models)
- [Visualization](#visualization)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Changelog](#changelog)

---

## Features
- Gait data preprocessing
- Feature extraction (generic and dataset-specific)
- Machine learning and deep learning models for recognition
- Exploratory data analysis and visualization tools
- Class-based API with legacy compatibility

## Supported Datasets

### IMU Sensor Based
- Daphnet: `https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait` ![Supported](https://img.shields.io/badge/status-supported-brightgreen)
- MobiFall: `https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/` ![In Progress](https://img.shields.io/badge/status-in%20progress-yellow)
- HAR-UP (formerly UPFall): `https://sites.google.com/up.edu.mx/har-up/` ![Supported](https://img.shields.io/badge/status-supported-brightgreen)
- UrFall: `https://fenix.ur.edu.pl/~mkepski/ds/uf.html` ![Supported](https://img.shields.io/badge/status-supported-brightgreen)
- Activity Net - Arduous: `https://www.mad.tf.fau.de/research/activitynet/wearable-multi-sensor-gait-based-daily-activity-data/` ![In Progress](https://img.shields.io/badge/status-in%20progress-yellow)

### Pressure/Force Sensor Based
- PhysioNet Gait in Parkinson's Disease: `https://physionet.org/content/gaitpdb/1.0.0/` ![Completed](https://img.shields.io/badge/status-completed-green)

---

## Installation
From PyPI:
```bash
pip install gaitsetpy
```

From Conda-Forge:
```bash
## add conda-forge
conda config --add channels conda-forge
conda config --set channel_priority strict

conda install gaitsetpy
## or mamba
mamba install gaitsetpy
```

From source:
```bash
git clone https://github.com/Alohomora-Labs/gaitSetPy.git
python setup.py install
```

Optionally, also install requirements:
```bash
pip install -r requirements.txt
```

---

## Quickstart
```python
import gaitsetpy as gsp

# Daphnet: Load, window, extract features, visualize
loader = gsp.DaphnetLoader()
data, names = loader.load_data("data/daphnet")
windows = loader.create_sliding_windows(data, names, window_size=192, step_size=32)
extractor = gsp.GaitFeatureExtractor()
features = extractor.extract_features(windows[0]['windows'], fs=64)

from gaitsetpy.eda import SensorStatisticsAnalyzer
analyzer = SensorStatisticsAnalyzer()
analyzer.visualize(windows[0]['windows'], features, sensor_name="shank", start_idx=0, end_idx=1000, num_windows=15)

# Train & evaluate a Random Forest
rf = gsp.RandomForestModel(n_estimators=50, random_state=42, max_depth=10)
rf.train(features)
metrics = rf.evaluate(features)
print(metrics.get('accuracy'))
```

---

## Examples: Notebooks and Scripts
- **Notebooks** (`examples/notebooks`):
  - `Daphnet_demo.ipynb`, `HARUP_demo.ipynb`, `PhysioNet_demo.ipynb`, `UrFall_demo.ipynb`
  - Open with Jupyter: `jupyter notebook examples/notebooks/HARUP_demo.ipynb`
- **Scripts** (`examples/scripts`):
  - `daphnet_all_models_example.py`, `harup_example.py`, `physionet_example.py`, `urfall_example.py`
  - Run via Python: `python examples/scripts/harup_example.py`

These examples mirror the latest APIs and are maintained alongside code updates.

---

## Usage by Dataset (Code Snippets)
Below are minimal, copy-pasteable examples for each supported dataset.

### Daphnet
```python
import gaitsetpy as gsp
loader = gsp.DaphnetLoader()
data, names = loader.load_data("data/daphnet")
windows = loader.create_sliding_windows(data, names, window_size=192, step_size=32)
features = gsp.GaitFeatureExtractor().extract_features(windows[0]['windows'], fs=64)
```

### HAR-UP
```python
import gaitsetpy as gsp
loader = gsp.HARUPLoader()
harup_data, harup_names = loader.load_data("data/harup")
windows = loader.create_sliding_windows(harup_data, harup_names, window_size=100, step_size=50)
# Dataset-specific feature extraction
features_data = loader.extract_features(windows)
```

### UrFall
```python
import gaitsetpy as gsp
loader = gsp.UrFallLoader()
# Load pre-extracted depth-map features
data, names = loader.load_data("data/urfall", data_types=['features'], use_falls=True, use_adls=True)
windows = loader.create_sliding_windows(data, names, window_size=30, step_size=15)
# Filepaths for media modalities
video_paths = loader.get_file_paths("data/urfall", 'video')
depth_paths = loader.get_file_paths("data/urfall", 'depth', sequences=['fall-01'])
print(loader.get_activity_info())
print(loader.get_feature_info())
```

### PhysioNet (VGRF)
```python
import gaitsetpy as gsp
loader = gsp.PhysioNetLoader()
data, names = loader.load_data("data/physionet")
windows = loader.create_sliding_windows(data, names, window_size=600, step_size=100)
features = gsp.PhysioNetFeatureExtractor().extract_features(windows[0]['windows'], fs=100)
labels = loader.get_labels()
```

### MobiFall
```python
import gaitsetpy as gsp
loader = gsp.MobiFallLoader()
# Depending on your local dataset layout
data, names = loader.load_data("data/mobifall")
windows = loader.create_sliding_windows(data, names, window_size=256, step_size=64)
# Use generic gait feature extractor
features = gsp.GaitFeatureExtractor().extract_features(windows[0]['windows'], fs=50)
```

### Arduous
```python
import gaitsetpy as gsp
loader = gsp.ArduousLoader()
# Depending on ActivityNet/Arduous export and local layout
data, names = loader.load_data("data/arduous")
windows = loader.create_sliding_windows(data, names, window_size=256, step_size=64)
features = gsp.GaitFeatureExtractor().extract_features(windows[0]['windows'], fs=50)
```

---

## Models
- `RandomForestModel` (built-in, scikit-learn)
- Optional deep learning models (LSTM, BiLSTM, CNN, GNN) if PyTorch is installed

Factory access via `gaitsetpy.classification.models.get_classification_model(name, **kwargs)`.

---

## Visualization
- `gaitsetpy.eda.SensorStatisticsAnalyzer`
- Dataset-specific visualizers for exploratory analysis

---

## Documentation
For detailed documentation and API reference, visit the official docs:
- Docs: `https://alohomora-labs.github.io/gaitSetPy/gaitsetpy.html`

---

## Contributing
We welcome contributions! Please read our [contributing guidelines](CONTRIBUTING.md) to get started.

---

## License
This project is licensed under the GNU GPL License. See the [LICENSE](LICENSE) file for details.

---

## Citation
If you use GaitSetPy in your research, please cite our work using the DOI badge above.

---

## Changelog
- 0.2.2: README overhaul (ToC, vibrant styling), per-dataset examples, notebooks/scripts documentation, version sync.
- See GitHub Releases for older entries.

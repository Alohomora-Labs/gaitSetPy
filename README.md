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
- URFall: [https://fenix.ur.edu.pl/~mkepski/ds/uf.html](https://fenix.ur.edu.pl/~mkepski/ds/uf.html) ![In Progress](https://img.shields.io/badge/status-in%20progress-yellow)
- Activity Net - Arduous : [https://www.mad.tf.fau.de/research/activitynet/wearable-multi-sensor-gait-based-daily-activity-data/](https://www.mad.tf.fau.de/research/activitynet/wearable-multi-sensor-gait-based-daily-activity-data/) ![In Progress](https://img.shields.io/badge/status-in%20progress-yellow)

### Pressure Sensor Based
- Physionet Gait in Parkinson's Disease: [https://physionet.org/content/gaitpdb/1.0.0/](https://physionet.org/content/gaitpdb/1.0.0/) ![Completed](https://img.shields.io/badge/status-completed-green)


## Installation

You can install GaitSetPy using pip:

```bash
git clone https://github.com/Alohomora-Labs/gaitSetPy.git
python setup.py install
```

Optionally, also install requirements
``` bash
pip install -r requirements.txt
```

## Usage

Here is a simple example to get you started with GaitSetPy:

### Daphnet Dataset Example

```python
import gaitsetpy as gsp

# Load gait data
data_dir = "data/daphnet"
daphnet, names = gsp.load_daphnet_data(data_dir)

# Preprocess data
sliding_windows = gsp.create_sliding_windows(daphnet, names)
freq = 64
# Extract features
features = gsp.extract_gait_features(sliding_windows[0]['windows'], freq, True, True, True)

# Visualize gait features
gsp.plot_sensor_with_features(sliding_windows[0]['windows'], features, sensor_name="shank", num_windows=15)
```

### HAR-UP Dataset Example

```python
import gaitsetpy as gsp

# Load HAR-UP data
data_dir = "data/harup"
harup_data, harup_names = gsp.load_harup_data(data_dir)

# Create sliding windows
window_size = 100  # 1 second at 100Hz
step_size = 50     # 0.5 second overlap
windows = gsp.create_harup_windows(harup_data, harup_names, window_size, step_size)

# Extract features
features_data = gsp.extract_harup_features(windows)

# For more advanced usage, see examples/harup_example.py
```
![alt text](image.png)

``` python

# Train a Random Forest
rf_model = gsp.RandomForestModel(n_estimators=50, random_state=42, max_depth=10)
rf_model.train(features)

# Load a pretrained model
rf_model.load_pretrained_weights("random_forest_model_40_10.pkl")

# Evaluate Model
gsp.evaluate_model(rf_model.model, features) # Assuming 'rf_model' is your trained RandomForestModel instance
```

## Documentation

For detailed documentation and API reference, please visit the [official documentation](https://alohomora-labs.github.io/gaitSetPy/gaitsetpy.html).

## Contributing

We welcome contributions! Please read our [contributing guidelines](CONTRIBUTING.md) to get started.

## License

This project is licensed under the GNU GPL License. See the [LICENSE](LICENSE) file for more details.

## Contact

For any questions or inquiries, please contact us at [jayeeta.chakrabortyfcs@kiit.ac.in](mailto:jayeeta.chakrabortyfcs@kiit.ac.in) or [aharshit123456@gmail.com](mailto:aharshit123456@gmail.com).

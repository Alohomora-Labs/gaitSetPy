# GaitSetPy

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

- UPFall: [https://sites.google.com/up.edu.mx/har-up/](https://sites.google.com/up.edu.mx/har-up/) ![In Progress](https://img.shields.io/badge/status-in%20progress-yellow)
- URFall: [https://fenix.ur.edu.pl/~mkepski/ds/uf.html](https://fenix.ur.edu.pl/~mkepski/ds/uf.html) ![In Progress](https://img.shields.io/badge/status-in%20progress-yellow)
- Activity Net - Arduous : [https://www.mad.tf.fau.de/research/activitynet/wearable-multi-sensor-gait-based-daily-activity-data/](https://www.mad.tf.fau.de/research/activitynet/wearable-multi-sensor-gait-based-daily-activity-data/) ![In Progress](https://img.shields.io/badge/status-in%20progress-yellow)

### Pressure Sensor Based
- Physionet Gait in Parkinson's Disease: [https://physionet.org/content/gaitpdb/1.0.0/](https://physionet.org/content/gaitpdb/1.0.0/) ![In Progress](https://img.shields.io/badge/status-in%20progress-yellow)


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

```python
import gaitsetpy as gsp

# Load gait data
daphnet, names = gsp.load_daphnet_data("")

# Preprocess data
sliding_windows = gsp.create_sliding_windows(daphnet, names)

# Extract features
features = gsp.extract_gait_features(sliding_windows[0]['windows'], 64, True, True, True)

# Visualize gait features
gsp.plot_sensor_with_features(sliding_windows[0]['windows'], features, sensor_name="shank", num_windows=15)
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

For detailed documentation and API reference, please visit the [official documentation](https://alohomora-labs.github.io/gaitSetPy/docs_gaitsetpy.html).

## Contributing

We welcome contributions! Please read our [contributing guidelines](CONTRIBUTING.md) to get started.

## License

This project is licensed under the GNU GPL License. See the [LICENSE](LICENSE) file for more details.

## Contact

For any questions or inquiries, please contact us at [jayeeta.chakrabortyfcs@kiit.ac.in](mailto:jayeeta.chakrabortyfcs@kiit.ac.in) or [aharshit123456@gmail.com](mailto:aharshit123456@gmail.com).

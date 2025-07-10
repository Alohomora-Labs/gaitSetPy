# GaitSetPy

GaitSetPy is a Python package for gait analysis and recognition. This package provides tools and algorithms to process and analyze gait data, enabling researchers and developers to build applications for gait recognition and clinical gait assessment.

## Features

- Modular and extensible class-based architecture
- Comprehensive gait data preprocessing pipeline
- Advanced feature extraction capabilities
- Multiple classification models
- Interactive visualization tools
- Support for various IMU and pressure sensor-based datasets

## Package Structure

```
gaitsetpy/
├── classification/    # Classification models and utilities
├── dataset/          # Dataset loaders and processors
├── eda/              # Exploratory data analysis tools
├── features/         # Feature extraction modules
└── preprocessing/    # Data preprocessing pipeline
```

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

Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Here's an example demonstrating the main features of GaitSetPy:

```python
from gaitsetpy.dataset import DaphnetDataset
from gaitsetpy.preprocessing import PreprocessingPipeline
from gaitsetpy.features import FeatureExtractor
from gaitsetpy.classification.models import RandomForestModel
from gaitsetpy.eda.visualization import plot_sensor_data, plot_features, plot_confusion_matrix

# Load and preprocess data
dataset = DaphnetDataset(data_path="path/to/data")
data = dataset.load_data()

# Create preprocessing pipeline
pipeline = PreprocessingPipeline(
    window_size=128,
    overlap=0.5,
    sampling_freq=64
)
processed_data = pipeline.process(data)

# Extract features
feature_extractor = FeatureExtractor(
    time_domain=True,
    frequency_domain=True,
    statistical=True
)
features = feature_extractor.extract_features(processed_data)

# Visualize data and features
plot_sensor_data(processed_data, sensor_name="shank", num_windows=15)
plot_features(features, feature_names=feature_extractor.feature_names)

# Train and evaluate model
model = RandomForestModel(n_estimators=50, max_depth=10, random_state=42)
model.train(features, labels)

# Load pretrained weights (optional)
model.load_weights("path/to/weights/random_forest_model_40_10.pkl")

# Evaluate model
metrics = model.evaluate(test_features, test_labels)
plot_confusion_matrix(metrics['confusion_matrix'], class_names=['Normal', 'FOG'])
```

## Available Models

- Random Forest Classifier
- LSTM (coming soon)
- BiLSTM (coming soon)
- Graph Neural Network (coming soon)
- Multi-Layer Perceptron (coming soon)

## Documentation

For detailed documentation and API reference, please visit the [official documentation](https://alohomora-labs.github.io/gaitSetPy/docs_gaitsetpy.html).

## Contributing

We welcome contributions! Please read our [contributing guidelines](CONTRIBUTING.md) to get started.

## License

This project is licensed under the GNU GPL License. See the [LICENSE](LICENSE) file for more details.

## Contact

For any questions or inquiries, please contact us at [jayeeta.chakrabortyfcs@kiit.ac.in](mailto:jayeeta.chakrabortyfcs@kiit.ac.in) or [aharshit123456@gmail.com](mailto:aharshit123456@gmail.com).

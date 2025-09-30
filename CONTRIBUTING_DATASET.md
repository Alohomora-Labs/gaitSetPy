# Adding Support for a New Dataset

This guide walks you through adding a new dataset to GaitSetPy. It covers creating a loader, registering it with the plugin system, exposing utilities, wiring in feature extraction, and adding tests and docs.

Use existing datasets under `gaitsetpy/dataset/` (e.g., `harup.py`, `physionet.py`, `daphnet.py`) as references.

---

## 1) Plan your dataset integration
- **Decide a canonical name**: short, lowercase, no spaces (e.g., `mydataset`). This becomes the registry key.
- **Identify raw format**: files, directory layout, required metadata (subjects, activities, labels, sampling frequency).
- **Decide labels**: class ids vs strings; provide a mapping in metadata.
- **Sampling frequency (fs)**: required for windowing and feature extraction.

---

## 2) Create the dataset loader class
Create a new file `gaitsetpy/dataset/<name>.py` and implement a class that inherits `BaseDatasetLoader`.

Minimal skeleton:
```python
from typing import List, Dict, Tuple, Optional
import os
import pandas as pd
import numpy as np
from ..core.base_classes import BaseDatasetLoader
from .utils import sliding_window

class MyDatasetLoader(BaseDatasetLoader):
    def __init__(self):
        super().__init__(name="mydataset", description="My Dataset description")
        self.metadata = {
            "sampling_frequency": 100,  # set your fs
            "activities": {  # id->label mapping
                0: "ClassA",
                1: "ClassB",
            },
            # add dataset-specific metadata as needed (sensors, components, etc.)
        }

    def load_data(self, data_dir: str, **kwargs) -> Tuple[List[pd.DataFrame], List[str]]:
        # 1) validate paths
        # 2) read raw files into DataFrames
        # 3) add columns needed downstream: subject_id, activity_id, (optional) trial_id, activity_label
        # 4) return [dataframes], [names]
        data_list: List[pd.DataFrame] = []
        names: List[str] = []
        # ...
        self.data = data_list
        return data_list, names

    def create_sliding_windows(
        self, data: List[pd.DataFrame], names: List[str], window_size: int = 192, step_size: int = 32
    ) -> List[Dict]:
        windows_data: List[Dict] = []
        for idx, df in enumerate(data):
            if df.empty:
                continue
            # choose numeric columns to window; include activity_id and labels
            sensor_columns = [
                c for c in df.columns
                if c not in ["subject_id", "activity_id", "trial_id", "activity_label", "TIME"]
                and pd.api.types.is_numeric_dtype(df[c])
            ]
            windows: List[Dict] = []
            for col in sensor_columns:
                win = sliding_window(df[col], window_size, step_size)
                windows.append({"name": col, "data": win})
            # activity windows and labels
            activity_windows = sliding_window(df["activity_id"], window_size, step_size)
            windows.append({"name": "activity_id", "data": activity_windows})
            labels = []
            for w in activity_windows:
                vals, counts = np.unique(w, return_counts=True)
                labels.append(vals[np.argmax(counts)])
            windows.append({"name": "labels", "data": np.array(labels)})
            windows_data.append({"name": names[idx], "windows": windows})
        return windows_data

    def get_supported_formats(self) -> List[str]:
        return [".csv"]  # adjust as needed
```

Notes:
- Ensure `self.metadata["sampling_frequency"]` is set correctly.
- Standardize columns: add `subject_id`, `activity_id`, `trial_id` (optional), `activity_label` for downstream use.
- Keep memory usage in mind; iterate and stream where possible for very large datasets.

---

## 3) Optional: dataset-specific helpers
If your dataset needs specialized downloading/extraction, add functions in `gaitsetpy/dataset/utils.py`:
- `download_<name>_data(data_dir: str) -> Optional[str]`
- `extract_<name>_data(data_dir: str) -> None`

Then route them via `download_dataset()` and `extract_dataset()` switch statements. Avoid interactive prompts for CI; if interactivity is needed, provide non-interactive defaults as well.

---

## 4) Register your dataset with the manager
Open `gaitsetpy/dataset/__init__.py` and:
- import your loader: `from .mydataset import MyDatasetLoader`
- in `_register_datasets()`, add: `manager.register_dataset("mydataset", MyDatasetLoader)`
- add public exports to `__all__` if you expose legacy helpers

This enables discovery via the `DatasetManager` registry and `get_available_datasets()`.

---

## 5) Wire up feature extraction (optional but recommended)
If your dataset has unique signals, consider a dataset-specific feature extractor under `gaitsetpy/features/` and register it in `gaitsetpy/features/__init__.py` with the `FeatureManager`. Follow `HARUPFeatureExtractor` for structure.

Within your loader, you may add a convenience method, e.g. `extract_features(...)`, to map window names to your feature extractor’s expected inputs, similar to `HARUPLoader.extract_features`.

---

## 6) Write tests
Add unit and integration tests under `tests/`:
- `tests/unit/test_dataset.py`: add cases ensuring your loader loads data, sets metadata, and returns names.
- `tests/integration/test_full_pipeline.py`: add a small fixture/sample that runs load → window → features.
- If adding download/extract utils, mock network and filesystem interactions.

Example unit test sketch:
```python
import pandas as pd
from gaitsetpy.dataset import get_dataset_manager

def test_mydataset_loader_smoke(tmp_path):
    # arrange: create tiny mock files & directory layout in tmp_path
    manager = get_dataset_manager()
    # act
    loader = manager.create_instance("mydataset", "mydataset", "mydataset loader")
    data, names = loader.load_data(str(tmp_path))
    # assert
    assert isinstance(data, list)
    assert isinstance(names, list)
    assert loader.metadata["sampling_frequency"] > 0
```

---

## 7) Document your dataset
- Add a short section in the top-level `README.md` if needed.
- Update or create a dataset page under `gaitsetpy/dataset/` HTML docs if you maintain rendered docs.
- Include citation/reference URLs in your loader’s module docstring.

---

## 8) Add an example (optional)
Create a small example under `examples/` showing how to:
- instantiate the loader via `DatasetManager`
- load data from a directory
- create windows and optionally extract features

---

## 9) Run tests and linters
- Run `python -m pytest -q` from the repo root.
- Ensure `tests/` pass locally and in CI.

---

## 10) Submit your PR
Checklist before opening a PR:
- [ ] New file `gaitsetpy/dataset/<name>.py` with a `BaseDatasetLoader` subclass
- [ ] Registered in `gaitsetpy/dataset/__init__.py`
- [ ] Tests added and passing
- [ ] Docs updated (this guide, README, or dataset docs)
- [ ] Optional: feature extractor registered if added

---

## Reference: Key APIs
- `BaseDatasetLoader` methods you must implement:
  - `load_data(data_dir, **kwargs) -> Tuple[List[pd.DataFrame], List[str]]`
  - `create_sliding_windows(data, names, window_size=..., step_size=...) -> List[Dict]`
  - `get_supported_formats() -> List[str]`
- Registry:
  - `DatasetManager.register_dataset(name: str, dataset_class)`
  - `get_available_datasets()`
- Helpers:
  - `dataset.utils.sliding_window(series, window_size, step_size)`
  - Add `download_<name>_data`, `extract_<name>_data` if needed and wire through switches.

---

## Tips
- Keep naming consistent: files lowercase, class names `CamelCase`, registry keys lowercase.
- Prefer pure, non-interactive code paths for tests; guard any interactive flows.
- Large datasets: expose filters (subjects, activities, trials) as kwargs to `load_data` to reduce memory.

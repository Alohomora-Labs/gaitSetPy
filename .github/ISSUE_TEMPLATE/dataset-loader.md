---
name: Dataset Loader Implementation
about: Implement a missing dataset loader (MobiFall, Arduous, etc.)
title: '[DATASET] Implement [Dataset Name] Loader'
labels: 'enhancement, dataset, good first issue'
assignees: ''
---

## Dataset Information

**Dataset Name:** [e.g., MobiFall, Arduous]  
**Dataset URL:** [Link to dataset homepage/repository]  
**Status:** Currently marked as "In Progress" in README

## Description

Implement the complete dataset loader for [Dataset Name]. The loader should follow the same pattern as existing loaders (Daphnet, HAR-UP, PhysioNet, UrFall).

## Tasks

- [ ] Implement `load_data()` method to read dataset files
- [ ] Implement `create_sliding_windows()` method
- [ ] Add proper data validation and error handling
- [ ] Update metadata dictionary with accurate information (sampling frequency, sensors, activities)
- [ ] Add comprehensive docstrings with examples
- [ ] Add unit tests in `tests/` directory
- [ ] Create example usage in `examples/[dataset_name]_example.py`
- [ ] Update README.md status from "In Progress" to "Supported"
- [ ] Update documentation

## Implementation Guidelines

### File Structure
```
gaitsetpy/dataset/[dataset_name].py
tests/test_[dataset_name].py
examples/[dataset_name]_example.py
```

### Expected Class Structure
```python
class [DatasetName]Loader(BaseDatasetLoader):
    def __init__(self):
        super().__init__(
            name="[dataset_name]",
            description="[Dataset description]"
        )
        self.metadata = {
            'sensors': [...],
            'components': [...],
            'sampling_frequency': ...,
            'activities': [...]
        }
    
    def load_data(self, data_dir: str, **kwargs) -> Tuple[List[pd.DataFrame], List[str]]:
        # Implementation
        pass
    
    def create_sliding_windows(self, data, names, window_size, step_size):
        # Implementation
        pass
```

### Testing Requirements
- Test with sample data files (if available)
- Test edge cases (empty files, missing data, incorrect formats)
- Test sliding window creation with different parameters
- Ensure all tests pass: `python run_tests.py`

### Documentation Requirements
- Add docstrings following Google style
- Include usage examples in docstrings
- Create a complete example script
- Update README.md with usage example

## Reference Implementations

See existing loaders for reference:
- `gaitsetpy/dataset/daphnet.py` - Well-documented IMU sensor dataset
- `gaitsetpy/dataset/harup.py` - Dataset with multiple data types
- `gaitsetpy/dataset/physionet.py` - Pressure sensor dataset
- `gaitsetpy/dataset/urfall.py` - Dataset with pre-extracted features

## Resources

- [CONTRIBUTING_DATASET.md](../CONTRIBUTING_DATASET.md) - Step-by-step guide for adding datasets
- [TESTING_FRAMEWORK.md](../TESTING_FRAMEWORK.md) - Testing guidelines
- Dataset documentation (add specific link)

## Definition of Done

- [ ] Code follows existing patterns and style guidelines
- [ ] All unit tests pass
- [ ] Code coverage for new code is >90%
- [ ] Example script runs without errors
- [ ] Documentation is complete and accurate
- [ ] Pre-commit hooks pass: `pre-commit run --all-files`
- [ ] Pull request is submitted with clear description

## Additional Notes

[Add any dataset-specific considerations, challenges, or notes here]

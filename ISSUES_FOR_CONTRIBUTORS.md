# GaitSetPy - Issues for Contributors

This document outlines improvement opportunities for the GaitSetPy package. Contributors are welcome to pick any of these issues and submit pull requests.

## Table of Contents
1. [Dataset Loaders - Incomplete Implementations](#1-dataset-loaders---incomplete-implementations)
2. [Deep Learning Models - Training & Evaluation](#2-deep-learning-models---training--evaluation)
3. [Code Quality & Documentation](#3-code-quality--documentation)
4. [Testing & Coverage](#4-testing--coverage)
5. [Feature Enhancements](#5-feature-enhancements)
6. [Performance & Optimization](#6-performance--optimization)
7. [CI/CD & Infrastructure](#7-cicd--infrastructure)

---

## 1. Dataset Loaders - Incomplete Implementations

### Issue 1.1: Implement MobiFall Dataset Loader
**Priority:** High  
**Difficulty:** Medium  
**Labels:** `enhancement`, `dataset`, `good first issue`

**Description:**
The MobiFall dataset loader is currently a stub implementation. Need to implement the complete data loading functionality.

**Files to modify:**
- `gaitsetpy/dataset/mobifall.py`

**Tasks:**
- [ ] Implement `load_data()` method to read MobiFall dataset files
- [ ] Implement `create_sliding_windows()` method
- [ ] Add proper data validation and error handling
- [ ] Update metadata dictionary with accurate information
- [ ] Add unit tests in `tests/` directory
- [ ] Create example usage in `examples/mobifall_example.py`

**References:**
- Dataset: https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/
- Current status in README: ![In Progress](https://img.shields.io/badge/status-in%20progress-yellow)

---

### Issue 1.2: Implement Arduous Dataset Loader
**Priority:** High  
**Difficulty:** Medium  
**Labels:** `enhancement`, `dataset`, `good first issue`

**Description:**
The Arduous (ActivityNet) dataset loader is currently a stub implementation. Need to implement the complete data loading functionality.

**Files to modify:**
- `gaitsetpy/dataset/arduous.py`

**Tasks:**
- [ ] Implement `load_data()` method to read Arduous dataset files
- [ ] Implement `create_sliding_windows()` method
- [ ] Add proper data validation and error handling
- [ ] Update metadata dictionary with accurate sampling frequency and activities
- [ ] Add unit tests in `tests/` directory
- [ ] Create example usage in `examples/arduous_example.py`

**References:**
- Dataset: https://www.mad.tf.fau.de/research/activitynet/wearable-multi-sensor-gait-based-daily-activity-data/
- Current status in README: ![In Progress](https://img.shields.io/badge/status-in%20progress-yellow)

---

## 2. Deep Learning Models - Training & Evaluation

### Issue 2.1: Validate and Test LSTM Model
**Priority:** High  
**Difficulty:** Medium  
**Labels:** `enhancement`, `deep-learning`, `testing`

**Description:**
The LSTM model is implemented but needs validation, testing, and documentation.

**Files to review/modify:**
- `gaitsetpy/classification/models/lstm.py`
- `tests/unit/test_classification.py`

**Tasks:**
- [ ] Add comprehensive unit tests for LSTM model
- [ ] Validate model training with sample dataset
- [ ] Add hyperparameter tuning examples
- [ ] Document expected input format and shapes
- [ ] Add model architecture visualization
- [ ] Create example in `examples/lstm_example.py`
- [ ] Update classification README status from TODO to Completed

---

### Issue 2.2: Validate and Test BiLSTM Model
**Priority:** High  
**Difficulty:** Medium  
**Labels:** `enhancement`, `deep-learning`, `testing`

**Description:**
The BiLSTM model is implemented but needs validation, testing, and documentation.

**Files to review/modify:**
- `gaitsetpy/classification/models/bilstm.py`
- `tests/unit/test_classification.py`

**Tasks:**
- [ ] Add comprehensive unit tests for BiLSTM model
- [ ] Validate model training with sample dataset
- [ ] Add hyperparameter tuning examples
- [ ] Document expected input format and shapes
- [ ] Add model architecture visualization
- [ ] Create example in `examples/bilstm_example.py`
- [ ] Update classification README status from TODO to Completed

---

### Issue 2.3: Validate and Test CNN Model
**Priority:** High  
**Difficulty:** Medium  
**Labels:** `enhancement`, `deep-learning`, `testing`

**Description:**
The 1D CNN model is implemented but needs validation, testing, and documentation.

**Files to review/modify:**
- `gaitsetpy/classification/models/cnn.py`
- `tests/unit/test_classification.py`

**Tasks:**
- [ ] Add comprehensive unit tests for CNN model
- [ ] Validate model training with sample dataset
- [ ] Add hyperparameter tuning examples
- [ ] Document expected input format and shapes (1D convolutions)
- [ ] Add model architecture visualization
- [ ] Create example in `examples/cnn_example.py`
- [ ] Update classification README status from TODO to Completed

---

### Issue 2.4: Validate and Test GNN Model
**Priority:** Medium  
**Difficulty:** Hard  
**Labels:** `enhancement`, `deep-learning`, `testing`, `advanced`

**Description:**
The GNN model is implemented but needs validation, testing, and proper PyTorch Geometric integration.

**Files to review/modify:**
- `gaitsetpy/classification/models/gnn.py`
- `tests/unit/test_classification.py`
- `requirements.txt`

**Tasks:**
- [ ] Add comprehensive unit tests for GNN model
- [ ] Document PyTorch Geometric installation requirements
- [ ] Validate model training with sample dataset
- [ ] Add graph construction examples from sensor data
- [ ] Document expected input format (graph structures)
- [ ] Add model architecture visualization
- [ ] Create example in `examples/gnn_example.py`
- [ ] Update classification README status from TODO to Completed

---

### Issue 2.5: Validate and Test MLP Model
**Priority:** Medium  
**Difficulty:** Easy  
**Labels:** `enhancement`, `deep-learning`, `testing`, `good first issue`

**Description:**
The MLP model is implemented but needs validation, testing, and documentation.

**Files to review/modify:**
- `gaitsetpy/classification/models/mlp.py`
- `tests/unit/test_classification.py`

**Tasks:**
- [ ] Add comprehensive unit tests for MLP model
- [ ] Validate model training with sample dataset
- [ ] Add hyperparameter tuning examples
- [ ] Document expected input format
- [ ] Create example in `examples/mlp_example.py`
- [ ] Update classification README status from TODO to Completed

---

## 3. Code Quality & Documentation

### Issue 3.1: Resolve TODO in visualization.py
**Priority:** Low  
**Difficulty:** Easy  
**Labels:** `enhancement`, `documentation`, `good first issue`

**Description:**
Complete the TODO in `gaitsetpy/eda/visualization.py` to refactor dataframe extraction functions.

**Files to modify:**
- `gaitsetpy/eda/visualization.py`

**Current TODO:**
```python
TODO : 
- Make the thigh, shank, trunk dataframe parent child extraction functions
```

**Tasks:**
- [ ] Create parent-child extraction functions for sensor data
- [ ] Refactor existing plot functions to use new extraction functions
- [ ] Add docstrings for new functions
- [ ] Add unit tests for extraction functions
- [ ] Update any related documentation

---

### Issue 3.2: Add Type Hints Across Codebase
**Priority:** Medium  
**Difficulty:** Medium  
**Labels:** `enhancement`, `code-quality`, `documentation`

**Description:**
While some functions have type hints, many are missing. Adding comprehensive type hints improves code maintainability and IDE support.

**Files to modify:**
- Multiple files across `gaitsetpy/` directory

**Tasks:**
- [ ] Add type hints to all public functions in `gaitsetpy/features/utils.py`
- [ ] Add type hints to visualization functions
- [ ] Add type hints to preprocessing utilities
- [ ] Verify mypy passes with strict mode
- [ ] Update CI to enforce type checking

**Tools:**
- Use `mypy` for validation (already in pre-commit hooks)
- Reference: PEP 484 - Type Hints

---

### Issue 3.3: Improve Docstring Coverage
**Priority:** Medium  
**Difficulty:** Easy  
**Labels:** `enhancement`, `documentation`, `good first issue`

**Description:**
Many functions have minimal or missing docstrings. Comprehensive docstrings improve API usability.

**Files to modify:**
- Multiple files across `gaitsetpy/` directory

**Tasks:**
- [ ] Add Google-style docstrings to all public functions in `gaitsetpy/features/utils.py`
- [ ] Add docstrings with examples for dataset loaders
- [ ] Add docstrings for all classification models
- [ ] Ensure all docstrings include Args, Returns, and Raises sections
- [ ] Add usage examples in docstrings

**Style Guide:**
- Use Google-style docstrings
- Include type information (compatible with type hints)
- Provide examples where helpful

---

### Issue 3.4: Add Examples for All Datasets
**Priority:** Medium  
**Difficulty:** Easy  
**Labels:** `enhancement`, `documentation`, `examples`

**Description:**
Create comprehensive examples for all supported datasets.

**Current status:**
- ✅ Daphnet: `examples/daphnet_all_models_example.py`
- ✅ HAR-UP: `examples/harup_example.py`
- ✅ PhysioNet: `examples/physionet_example.py`
- ✅ UrFall: `examples/urfall_example.py`
- ❌ MobiFall: Missing
- ❌ Arduous: Missing

**Tasks:**
- [ ] Create `examples/mobifall_example.py` (depends on Issue 1.1)
- [ ] Create `examples/arduous_example.py` (depends on Issue 1.2)
- [ ] Add end-to-end pipeline examples showing data loading → feature extraction → training → evaluation
- [ ] Add visualization examples for each dataset
- [ ] Ensure all examples run without errors

---

## 4. Testing & Coverage

### Issue 4.1: Increase Test Coverage to 95%+
**Priority:** High  
**Difficulty:** Medium  
**Labels:** `testing`, `quality`

**Description:**
Current test coverage is 86% (105/122 tests passing). Need to increase coverage and fix failing tests.

**Current Status (from TESTING_FRAMEWORK.md):**
- Total Tests: 122
- Passing: 105 (86%)
- Failing: 17 (14%)

**Test Coverage by Category:**
| Category | Tests | Passing | Coverage |
|----------|-------|---------|----------|
| Base Classes | 23 | 23 | 100% |
| Managers | 29 | 29 | 100% |
| Features | 48 | 48 | 100% |
| Classification | 25 | 20 | 80% |
| Integration | 22 | 15 | 68% |
| Dataset Tests | 18 | 18 | 100% |

**Tasks:**
- [ ] Fix failing classification tests (5 failures)
- [ ] Fix failing integration tests (7 failures)
- [ ] Add tests for visualization module
- [ ] Add tests for preprocessing pipeline
- [ ] Achieve 95%+ code coverage
- [ ] Add property-based testing using `hypothesis`

**Known Issues to Fix:**
1. Feature Preprocessing: Missing 'annotations' key in test data
2. Manager Integration: Constructor parameter mismatches
3. Edge Cases: Need refinement for actual function behavior
4. HAR-UP Pipeline: Missing sensor columns in test data

---

### Issue 4.2: Add Integration Tests for Deep Learning Models
**Priority:** Medium  
**Difficulty:** Medium  
**Labels:** `testing`, `deep-learning`

**Description:**
Add comprehensive integration tests for all PyTorch-based models (LSTM, BiLSTM, CNN, MLP, GNN).

**Files to create/modify:**
- `tests/integration/test_dl_models.py`

**Tasks:**
- [ ] Create integration tests for complete training pipeline
- [ ] Test model save/load functionality
- [ ] Test model evaluation metrics
- [ ] Test GPU/CPU compatibility
- [ ] Test with different batch sizes and epochs
- [ ] Add performance benchmarking tests
- [ ] Mark GPU tests with `@pytest.mark.requires_gpu`

---

### Issue 4.3: Add Visual Testing for Plots
**Priority:** Low  
**Difficulty:** Medium  
**Labels:** `testing`, `visualization`

**Description:**
Add tests to verify visualization outputs are generated correctly.

**Files to create/modify:**
- `tests/unit/test_visualization.py`

**Tasks:**
- [ ] Test that plot functions create figure objects
- [ ] Test that plots can be saved to files
- [ ] Add regression tests for plot outputs
- [ ] Test different plot configurations
- [ ] Use `@pytest.mark.visualization` marker

---

## 5. Feature Enhancements

### Issue 5.1: Add Noise Filters
**Priority:** Medium  
**Difficulty:** Medium  
**Labels:** `enhancement`, `feature`

**Description:**
Add noise filtering capabilities as mentioned in TODO comment in `gaitsetpy/main.py`.

**Files to modify:**
- `gaitsetpy/main.py`
- `gaitsetpy/preprocessing/preprocessors.py`

**Tasks:**
- [ ] Implement Butterworth filter for noise reduction
- [ ] Implement Kalman filter for noise reduction
- [ ] Implement median filter for outlier removal
- [ ] Add configurable filter parameters
- [ ] Add tests for noise filters
- [ ] Add examples demonstrating filter usage
- [ ] Update documentation

---

### Issue 5.2: Refactor Sliding Window Implementation
**Priority:** Medium  
**Difficulty:** Medium  
**Labels:** `enhancement`, `refactoring`

**Description:**
Refactor sliding window creation as mentioned in TODO comment in `gaitsetpy/main.py`.

**Files to modify:**
- `gaitsetpy/main.py`
- `gaitsetpy/dataset/utils.py`

**Tasks:**
- [ ] Create unified sliding window interface
- [ ] Add support for variable stride lengths
- [ ] Add support for different padding strategies
- [ ] Optimize performance for large datasets
- [ ] Add tests for edge cases
- [ ] Update all dataset loaders to use new interface

---

### Issue 5.3: Add Feature Selection Methods
**Priority:** Medium  
**Difficulty:** Medium  
**Labels:** `enhancement`, `feature`

**Description:**
Add feature selection capabilities to improve model performance and reduce dimensionality.

**Files to create/modify:**
- `gaitsetpy/features/selection.py` (new file)
- `gaitsetpy/features/__init__.py`

**Tasks:**
- [ ] Implement correlation-based feature selection
- [ ] Implement mutual information feature selection
- [ ] Implement recursive feature elimination (RFE)
- [ ] Implement feature importance from tree-based models
- [ ] Add tests for feature selection methods
- [ ] Add examples demonstrating feature selection
- [ ] Update documentation

---

### Issue 5.4: Add Cross-Validation Support
**Priority:** Medium  
**Difficulty:** Easy  
**Labels:** `enhancement`, `feature`, `good first issue`

**Description:**
Add cross-validation utilities for more robust model evaluation.

**Files to create/modify:**
- `gaitsetpy/classification/utils/eval.py`
- `gaitsetpy/classification/__init__.py`

**Tasks:**
- [ ] Add k-fold cross-validation
- [ ] Add stratified k-fold cross-validation
- [ ] Add leave-one-subject-out cross-validation (important for gait data)
- [ ] Add time-series cross-validation
- [ ] Add tests for cross-validation methods
- [ ] Add examples demonstrating cross-validation
- [ ] Update documentation

---

### Issue 5.5: Add Model Comparison Utilities
**Priority:** Low  
**Difficulty:** Easy  
**Labels:** `enhancement`, `feature`, `good first issue`

**Description:**
Add utilities to compare multiple models and generate comparison reports.

**Files to create/modify:**
- `gaitsetpy/classification/utils/comparison.py` (new file)

**Tasks:**
- [ ] Create function to train and evaluate multiple models
- [ ] Generate comparison tables (accuracy, precision, recall, F1)
- [ ] Create visualization for model comparison
- [ ] Add statistical significance tests
- [ ] Add tests for comparison utilities
- [ ] Add examples demonstrating model comparison

---

## 6. Performance & Optimization

### Issue 6.1: Optimize Feature Extraction Performance
**Priority:** Medium  
**Difficulty:** Medium  
**Labels:** `performance`, `optimization`

**Description:**
Profile and optimize feature extraction, especially for large datasets.

**Files to modify:**
- `gaitsetpy/features/gait_features.py`
- `gaitsetpy/features/utils.py`

**Tasks:**
- [ ] Profile feature extraction with large datasets
- [ ] Vectorize operations where possible
- [ ] Add caching for expensive computations
- [ ] Add parallel processing support
- [ ] Add progress bars for long operations
- [ ] Add performance tests
- [ ] Document performance characteristics

---

### Issue 6.2: Add Batch Processing Support
**Priority:** Medium  
**Difficulty:** Medium  
**Labels:** `enhancement`, `performance`

**Description:**
Add batch processing capabilities for handling large datasets efficiently.

**Files to create/modify:**
- `gaitsetpy/dataset/utils.py`
- `gaitsetpy/features/gait_features.py`

**Tasks:**
- [ ] Add batch data loading
- [ ] Add batch feature extraction
- [ ] Add memory-efficient iterators
- [ ] Add progress tracking
- [ ] Add tests for batch processing
- [ ] Add examples demonstrating batch processing

---

### Issue 6.3: Add Memory Profiling and Optimization
**Priority:** Low  
**Difficulty:** Medium  
**Labels:** `performance`, `optimization`

**Description:**
Profile memory usage and optimize for large datasets.

**Tasks:**
- [ ] Add memory profiling tools
- [ ] Identify memory bottlenecks
- [ ] Optimize data structures
- [ ] Add streaming processing for large files
- [ ] Add memory usage documentation
- [ ] Add tests for memory-constrained scenarios

---

## 7. CI/CD & Infrastructure

### Issue 7.1: Add Code Coverage Reporting
**Priority:** High  
**Difficulty:** Easy  
**Labels:** `ci-cd`, `infrastructure`, `good first issue`

**Description:**
Integrate code coverage reporting with Codecov or similar service.

**Files to modify:**
- `.github/workflows/tests.yml`
- `README.md`

**Current status:**
- Codecov is configured in workflow but token may need setup
- No coverage badge in README

**Tasks:**
- [ ] Verify Codecov token is set up
- [ ] Add coverage badge to README
- [ ] Configure coverage thresholds
- [ ] Add coverage trend tracking
- [ ] Add coverage reports to pull requests

---

### Issue 7.2: Add Multi-Platform Testing
**Priority:** Medium  
**Difficulty:** Easy  
**Labels:** `ci-cd`, `infrastructure`, `good first issue`

**Description:**
Test on multiple platforms (Linux, macOS, Windows) and Python versions.

**Files to modify:**
- `.github/workflows/tests.yml`

**Current status:**
- Only Linux with Python 3.10

**Tasks:**
- [ ] Add macOS testing
- [ ] Add Windows testing
- [ ] Test on Python 3.8, 3.9, 3.10, 3.11
- [ ] Document platform-specific issues
- [ ] Add matrix strategy to CI workflow

---

### Issue 7.3: Add Automated Release Workflow
**Priority:** Low  
**Difficulty:** Medium  
**Labels:** `ci-cd`, `infrastructure`

**Description:**
Automate release process including version bumping, changelog generation, and PyPI publishing.

**Files to create/modify:**
- `.github/workflows/release.yml` (new file)
- `CHANGELOG.md` (new file)

**Tasks:**
- [ ] Create automated release workflow
- [ ] Add semantic versioning support
- [ ] Generate changelog automatically
- [ ] Automate PyPI publishing
- [ ] Add release notes generation
- [ ] Document release process

---

### Issue 7.4: Add Documentation Build and Deployment
**Priority:** Medium  
**Difficulty:** Easy  
**Labels:** `ci-cd`, `infrastructure`, `documentation`

**Description:**
Automate documentation building and deployment.

**Files to review/modify:**
- `.github/workflows/docs.yml`
- `generate_docs.py`

**Current status:**
- Manual documentation generation with pdoc

**Tasks:**
- [ ] Verify docs workflow is working correctly
- [ ] Add documentation versioning
- [ ] Add API reference completeness checks
- [ ] Add documentation linting
- [ ] Ensure docs are deployed automatically

---

## 8. Additional Enhancements

### Issue 8.1: Add Jupyter Notebook Tutorials
**Priority:** Medium  
**Difficulty:** Easy  
**Labels:** `enhancement`, `documentation`, `tutorials`, `good first issue`

**Description:**
Create interactive Jupyter notebook tutorials for common use cases.

**Files to create:**
- `tutorials/01_getting_started.ipynb`
- `tutorials/02_daphnet_analysis.ipynb`
- `tutorials/03_feature_extraction.ipynb`
- `tutorials/04_model_training.ipynb`
- `tutorials/05_custom_datasets.ipynb`

**Tasks:**
- [ ] Create tutorial notebooks
- [ ] Add visualizations and explanations
- [ ] Test all notebooks execute without errors
- [ ] Add to documentation
- [ ] Consider using nbconvert for static HTML versions

---

### Issue 8.2: Add Command-Line Interface (CLI)
**Priority:** Low  
**Difficulty:** Medium  
**Labels:** `enhancement`, `feature`

**Description:**
Add a command-line interface for common tasks.

**Files to create:**
- `gaitsetpy/cli.py` (new file)
- `setup.py` (update entry points)

**Tasks:**
- [ ] Design CLI interface
- [ ] Implement commands: train, evaluate, extract-features, preprocess
- [ ] Add configuration file support (YAML/JSON)
- [ ] Add progress bars and logging
- [ ] Add CLI tests
- [ ] Document CLI usage

---

### Issue 8.3: Add Docker Support
**Priority:** Low  
**Difficulty:** Easy  
**Labels:** `enhancement`, `infrastructure`, `good first issue`

**Description:**
Add Docker support for reproducible environments.

**Files to create:**
- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`

**Tasks:**
- [ ] Create Dockerfile with all dependencies
- [ ] Add docker-compose for development
- [ ] Add GPU support variant
- [ ] Document Docker usage
- [ ] Test on different platforms

---

### Issue 8.4: Add Data Augmentation Techniques
**Priority:** Medium  
**Difficulty:** Medium  
**Labels:** `enhancement`, `feature`

**Description:**
Add data augmentation techniques for improving model robustness.

**Files to create/modify:**
- `gaitsetpy/preprocessing/augmentation.py` (new file)

**Tasks:**
- [ ] Implement time warping
- [ ] Implement jittering (adding noise)
- [ ] Implement scaling
- [ ] Implement rotation
- [ ] Implement time masking
- [ ] Add tests for augmentation techniques
- [ ] Add examples demonstrating augmentation

---

## Contributing

To work on any of these issues:

1. **Fork the repository** on GitHub
2. **Create a new branch** for your work: `git checkout -b feat/<issue-name>`
3. **Make your changes** following the coding standards
4. **Add tests** for your changes
5. **Run the test suite** to ensure everything passes: `python run_tests.py`
6. **Run pre-commit hooks**: `pre-commit run --all-files`
7. **Submit a pull request** with a clear description of your changes

### Resources

- [Contributing Guidelines](CONTRIBUTING.md)
- [Dataset Contributing Guide](CONTRIBUTING_DATASET.md)
- [Testing Framework Documentation](TESTING_FRAMEWORK.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)

### Questions?

If you have questions about any of these issues, please:
- Open a discussion on GitHub
- Comment on the specific issue
- Contact the maintainers at [aharshit123456@gmail.com](mailto:aharshit123456@gmail.com)

---

**Last Updated:** 2025-02-24  
**Maintainer:** @aharshit123456

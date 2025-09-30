# Quick Start Guide for Contributors

Welcome to GaitSetPy! This guide will help you get started with contributing to the project.

## 🚀 Quick Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/gaitSetPy.git
cd gaitSetPy

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -e .

# 4. Install pre-commit hooks
pre-commit install

# 5. Run tests to verify setup
python run_tests.py --fast
```

## 📋 Finding Issues to Work On

### Good First Issues
Perfect for newcomers to the project:
- [Dataset Loaders](#dataset-loaders) - Implement MobiFall or Arduous loaders
- [Documentation](#documentation) - Improve docstrings and examples
- [Testing](#testing) - Add tests to improve coverage
- [Simple Features](#simple-features) - Add utility functions

### Issue Labels
- `good first issue` - Easy issues for beginners
- `enhancement` - New features or improvements
- `bug` - Something isn't working
- `documentation` - Improvements to documentation
- `testing` - Related to test coverage
- `dataset` - Dataset loader related
- `deep-learning` - Related to DL models

## 🔧 Development Workflow

### 1. Choose an Issue
Browse [ISSUES_FOR_CONTRIBUTORS.md](ISSUES_FOR_CONTRIBUTORS.md) or GitHub Issues

### 2. Create a Branch
```bash
git checkout -b feat/your-feature-name  # For features
git checkout -b fix/bug-description     # For bug fixes
git checkout -b docs/documentation-update  # For docs
```

### 3. Make Your Changes
- Follow existing code style
- Add tests for new functionality
- Update documentation

### 4. Test Your Changes
```bash
# Run all tests
python run_tests.py

# Run specific tests
python run_tests.py --file tests/unit/test_yourmodule.py

# Run with coverage
python run_tests.py --coverage

# Run pre-commit checks
pre-commit run --all-files
```

### 5. Commit Your Changes
```bash
git add .
git commit -m "feat: add feature description"  # Or fix:, docs:, test:, etc.
```

Commit message prefixes:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test additions/changes
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `chore:` - Maintenance tasks

### 6. Push and Create PR
```bash
git push origin your-branch-name
```
Then create a Pull Request on GitHub with:
- Clear title describing the change
- Reference to related issue(s)
- Description of what changed and why
- Screenshots (for UI changes)

## 📚 Key Resources

### Documentation
- [README.md](README.md) - Project overview
- [CONTRIBUTING.md](CONTRIBUTING.md) - General contribution guidelines
- [CONTRIBUTING_DATASET.md](CONTRIBUTING_DATASET.md) - Adding new datasets
- [TESTING_FRAMEWORK.md](TESTING_FRAMEWORK.md) - Testing guidelines
- [ISSUES_FOR_CONTRIBUTORS.md](ISSUES_FOR_CONTRIBUTORS.md) - Detailed issue list

### Code Examples
- `examples/` - Example scripts for each dataset
- `tests/` - Test examples showing usage patterns

### Issue Templates
- `.github/ISSUE_TEMPLATE/` - Templates for creating issues

## 🎯 Common Tasks

### Adding a New Dataset Loader

1. Create loader class in `gaitsetpy/dataset/[dataset_name].py`
2. Inherit from `BaseDatasetLoader`
3. Implement required methods
4. Add tests in `tests/test_[dataset_name].py`
5. Create example in `examples/[dataset_name]_example.py`
6. Update README.md

See [CONTRIBUTING_DATASET.md](CONTRIBUTING_DATASET.md) for details.

### Adding Tests

1. Create/update test file in `tests/`
2. Use fixtures from `tests/conftest.py`
3. Follow existing test patterns
4. Run tests to verify: `python run_tests.py`

### Improving Documentation

1. Add/update docstrings (Google style)
2. Update examples if needed
3. Update README.md if adding features
4. Generate docs: `python generate_docs.py`

### Adding a Feature

1. Design the API
2. Implement the feature
3. Add comprehensive tests (>90% coverage)
4. Add docstrings and type hints
5. Create example usage
6. Update documentation

## 🐛 Debugging Tips

### Run Tests in Verbose Mode
```bash
python run_tests.py --verbose
```

### Run Specific Test
```bash
pytest tests/unit/test_specific.py::test_function_name -v
```

### Check Code Coverage
```bash
python run_tests.py --coverage
# Open htmlcov/index.html in browser
```

### Debug with Print Statements
- Use `print()` for quick debugging
- Use `import pdb; pdb.set_trace()` for interactive debugging

## ✅ Pre-Submission Checklist

Before submitting a PR, ensure:

- [ ] All tests pass: `python run_tests.py`
- [ ] Code coverage is maintained or improved
- [ ] Pre-commit hooks pass: `pre-commit run --all-files`
- [ ] Docstrings added for new functions/classes
- [ ] Type hints added
- [ ] Examples updated if needed
- [ ] README updated if adding features
- [ ] Commit messages are clear and descriptive

## 🤝 Getting Help

### Ask Questions
- Open a [Discussion](https://github.com/Alohomora-Labs/gaitSetPy/discussions)
- Comment on the issue you're working on
- Email maintainers: aharshit123456@gmail.com

### Report Problems
- Use appropriate issue template
- Include error messages and environment details
- Provide minimal reproducible example

## 🌟 Recognition

Contributors are acknowledged in:
- README.md
- Release notes
- GitHub contributors page

Thank you for contributing to GaitSetPy! 🎉

---

**Quick Links:**
- [Issues for Contributors](ISSUES_FOR_CONTRIBUTORS.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Documentation](https://alohomora-labs.github.io/gaitSetPy/gaitsetpy.html)

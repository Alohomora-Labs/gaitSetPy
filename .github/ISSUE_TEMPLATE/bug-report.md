---
name: Bug Report
about: Report a bug or unexpected behavior
title: '[BUG] '
labels: 'bug'
assignees: ''
---

## Bug Description

**Brief Summary:**
[A clear and concise description of the bug]

**Component:**
- [ ] Data Loading
- [ ] Feature Extraction
- [ ] Preprocessing
- [ ] Classification/Models
- [ ] Visualization
- [ ] Documentation
- [ ] Other

**Severity:**
- [ ] Critical (crashes, data loss)
- [ ] High (major functionality broken)
- [ ] Medium (feature not working as expected)
- [ ] Low (minor issue, workaround available)

## To Reproduce

Steps to reproduce the behavior:
1. [First Step]
2. [Second Step]
3. [Third Step]
4. [etc.]

**Minimal Code Example:**
```python
import gaitsetpy as gsp

# Code that reproduces the bug
...
```

## Expected Behavior

[A clear description of what you expected to happen]

## Actual Behavior

[A clear description of what actually happened]

**Error Message/Stack Trace:**
```
[Paste the full error message and stack trace here]
```

## Environment

**GaitSetPy Version:**
```bash
$ python -c "import gaitsetpy; print(gaitsetpy.__version__)"
[Output]
```

**Python Version:**
```bash
$ python --version
[Output]
```

**Operating System:**
- [ ] Linux (specify distribution and version)
- [ ] macOS (specify version)
- [ ] Windows (specify version)

**Dependencies:**
```bash
$ pip list | grep -E "numpy|pandas|scipy|torch|scikit-learn"
[Output]
```

## Additional Context

**Dataset Used (if applicable):**
- [ ] Daphnet
- [ ] MobiFall
- [ ] HAR-UP
- [ ] PhysioNet
- [ ] UrFall
- [ ] Arduous
- [ ] Custom dataset

**Related Issues:**
[Link to related issues if any]

**Workaround (if found):**
[Describe any workaround you've found]

**Screenshots/Plots:**
[If applicable, add screenshots or plots to help explain the problem]

## Possible Solution

[If you have suggestions on how to fix the bug, describe them here]

## Checklist

- [ ] I have checked that this issue hasn't already been reported
- [ ] I have provided a minimal code example that reproduces the issue
- [ ] I have included the full error message and stack trace
- [ ] I have provided information about my environment
- [ ] I have checked the documentation for relevant information

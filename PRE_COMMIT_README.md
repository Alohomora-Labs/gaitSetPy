# Pre-commit Hooks for GaitSetPy

This document explains how to set up and use pre-commit hooks for GaitSetPy development.

## What are Pre-commit Hooks?

Pre-commit hooks automatically run checks and formatters on your code before each commit, ensuring code quality and consistency across the project.

## Installation

### 1. Install pre-commit

```bash
pip install pre-commit
```

Or install all development dependencies:

```bash
pip install -r requirements.txt
```

### 2. Install the git hook scripts

```bash
pre-commit install
```

This will install the pre-commit hooks in your local `.git/hooks/` directory.

## What Gets Checked?

Our pre-commit configuration includes:

### Code Formatting
- **Black**: Automatically formats Python code to a consistent style
- **isort**: Sorts and organizes imports

### Code Quality
- **Flake8**: Lints Python code for style violations and potential bugs
- **mypy**: Performs static type checking

### Security
- **Bandit**: Scans for common security issues

### General Checks
- Trailing whitespace removal
- End-of-file fixer
- YAML/JSON validation
- Large file detection
- Merge conflict detection
- Private key detection

## Usage

### Automatic Run on Commit

Once installed, pre-commit hooks will run automatically on `git commit`:

```bash
git add .
git commit -m "Your commit message"
```

If any hooks fail, the commit will be blocked and you'll see which checks failed.

### Manual Run

Run all hooks on all files:

```bash
pre-commit run --all-files
```

Run specific hook:

```bash
pre-commit run black --all-files
pre-commit run flake8 --all-files
```

Run on staged files only:

```bash
pre-commit run
```

### Update Hooks

Update all hooks to their latest versions:

```bash
pre-commit autoupdate
```

### Skip Hooks (Emergency Only)

If you absolutely need to skip pre-commit hooks (not recommended):

```bash
git commit --no-verify -m "Your message"
```

## Configuration

The pre-commit configuration is in `.pre-commit-config.yaml`. Tool-specific settings are in `pyproject.toml`.

### Black Settings
- Line length: 100 characters
- Target Python versions: 3.8, 3.9, 3.10, 3.11

### isort Settings
- Profile: black (compatible with Black)
- Line length: 100 characters

### Flake8 Settings
- Max line length: 100 characters
- Ignored errors: E203, E501, W503 (Black compatibility)

### mypy Settings
- Ignore missing imports
- No strict optional

## Troubleshooting

### Hook fails but you can't see why

Run with verbose output:

```bash
pre-commit run --all-files --verbose
```

### Black and Flake8 conflict

Our configuration is set to avoid conflicts. If you encounter any, check:
1. Line length is set to 100 in both tools
2. Flake8 ignores E203, E501, W503

### Pre-commit is too slow

You can skip slow hooks temporarily:

```bash
SKIP=mypy git commit -m "Your message"
```

Or skip multiple hooks:

```bash
SKIP=mypy,bandit git commit -m "Your message"
```

## CI Integration

Pre-commit hooks also run in CI/CD pipelines to ensure all code follows the same standards. They run automatically on pull requests.

## Recommended Workflow

1. Make your changes
2. Run `pre-commit run --all-files` to check everything
3. Fix any issues reported
4. Commit your changes
5. Hooks will run again automatically on commit

## Getting Help

If you encounter issues with pre-commit hooks:

1. Check this README
2. Check `.pre-commit-config.yaml` for configuration
3. Check `pyproject.toml` for tool-specific settings
4. Open an issue on GitHub

## Additional Resources

- [Pre-commit Documentation](https://pre-commit.com/)
- [Black Documentation](https://black.readthedocs.io/)
- [isort Documentation](https://pycqa.github.io/isort/)
- [Flake8 Documentation](https://flake8.pycqa.org/)
- [mypy Documentation](https://mypy.readthedocs.io/)
- [Bandit Documentation](https://bandit.readthedocs.io/)

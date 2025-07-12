# Documentation Generation for gaitSetPy

This project uses **pdoc** to automatically generate HTML documentation from Python docstrings.

## Overview

The documentation system consists of:
- **`generate_docs.py`**: Python script that generates HTML documentation using pdoc
- **`.github/workflows/docs.yml`**: GitHub Actions workflow for automatic documentation generation
- **`requirements.txt`**: Includes pdoc as a dependency

## Manual Documentation Generation

To generate documentation locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Generate documentation
python generate_docs.py
```

This will create:
- `index.html` - Main documentation page (redirects to gaitsetpy.html)
- `gaitsetpy.html` - Main module documentation
- `gaitsetpy/` - Directory containing HTML files for all submodules
- `search.js` - Search index for documentation

## Automatic Documentation Generation

The GitHub Actions workflow automatically:

1. **Triggers** when:
   - Code is pushed to the `main` branch
   - Changes are made to the `gaitsetpy/` directory
   - Changes are made to `generate_docs.py` or `requirements.txt`
   - Manual trigger via GitHub Actions UI

2. **Generates** documentation using the `generate_docs.py` script

3. **Commits** updated documentation files back to the main branch

4. **Deploys** documentation to GitHub Pages at `https://username.github.io/gaitSetPy/docs/`

## Documentation Structure

The generated documentation includes:

- **Main package**: `gaitsetpy.html`
- **Classification module**: `gaitsetpy/classification.html`
  - Models: `bilstm.html`, `gnn.html`, `lstm.html`, `mlp.html`, `random_forest.html`
  - Utils: `dataset.html`, `eval.html`, `preprocess.html`, `train.html`
- **Dataset module**: `gaitsetpy/dataset.html`
  - Datasets: `arduous.html`, `daphnet.html`, `mobifall.html`
  - Utils: `utils.html`
- **EDA module**: `gaitsetpy/eda.html`
  - `statistics.html`, `visualization.html`
- **Features module**: `gaitsetpy/features.html`
  - `gait_features.html`, `utils.html`
- **Preprocessing module**: `gaitsetpy/preprocessing.html`
  - `pipeline.html`, `utils.html`
- **Utils module**: `gaitsetpy/utils.html`

## Customization

To customize the documentation:

1. **Modify docstrings** in your Python code
2. **Update `generate_docs.py`** to change pdoc options
3. **Modify the workflow** in `.github/workflows/docs.yml` to change triggers or deployment

## GitHub Pages Setup

To enable GitHub Pages deployment:

1. Go to your repository settings
2. Navigate to "Pages" in the left sidebar
3. Under "Source", select "Deploy from a branch"
4. Choose "gh-pages" as the branch
5. Set folder to "/ (root)"

Your documentation will be available at `https://username.github.io/repository-name/docs/`

## Troubleshooting

### Common Issues

1. **pdoc not found**: Make sure pdoc is installed (`pip install pdoc`)
2. **Import errors**: Ensure all dependencies are installed
3. **GitHub Pages not updating**: Check the Actions tab for workflow errors
4. **Documentation not reflecting changes**: Trigger workflow manually or push changes to main

### Manual Workflow Trigger

To manually trigger the documentation generation:

1. Go to your repository's "Actions" tab
2. Click on "Generate Documentation" workflow
3. Click "Run workflow" button
4. Select the branch (usually `main`)
5. Click "Run workflow"

## Dependencies

- **pdoc**: For generating HTML documentation
- **Python 3.11+**: Required for the workflow
- **GitHub Actions**: For automatic deployment

## File Structure

```
gaitSetPy/
├── generate_docs.py           # Documentation generation script
├── .github/workflows/docs.yml # GitHub Actions workflow
├── requirements.txt           # Dependencies (includes pdoc)
├── index.html                 # Main documentation page
├── gaitsetpy.html            # Package documentation
├── search.js                 # Search functionality
└── gaitsetpy/                # Module documentation
    ├── classification.html
    ├── dataset.html
    ├── eda.html
    ├── features.html
    ├── preprocessing.html
    └── utils.html
```

This system ensures that your documentation stays up-to-date with your code changes automatically! 
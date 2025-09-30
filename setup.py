from setuptools import setup, find_packages

# Read the README.md file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='gaitsetpy',
    version='0.2.0',
    packages=find_packages(include=["gaitsetpy", "gaitsetpy.*"]),
    description="A Python package for gait analysis using sensor data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Alohomora Labs",
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'requests',
        'statsmodels',
        'matplotlib',
        'scikit-learn',
        'joblib',
        'tqdm',
    ],
    extras_require={
        'deep-learning': [
            'torch>=1.9.0',
        ],
        'all': [
            'torch>=1.9.0',
        ],
    },
    python_requires='>=3.8',
)
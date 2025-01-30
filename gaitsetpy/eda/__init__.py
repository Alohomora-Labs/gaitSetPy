"""
eda: Provides exploratory data analysis tools for gait datasets.

Features:
- Visualization of data (scatterplots, labellograms, etc.)
- Statistical summaries and distributions.
"""

from .visualization import plot_labellogram, plot_scatter, plot_power_spectrum
from .statistics import summarize_data

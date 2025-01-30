"""
classification: Classification models and utilities.

Features:
- Model training, evaluation, and testing.
- Consistent label management.
"""

from .models import train_model, evaluate_model, predict
from .utils import load_model, save_model


from gaitsetpy import *
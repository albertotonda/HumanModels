"""
humanmodels
===========
scikit-learn compatible human-defined models for classification and regression
"""
from .humanmodels import HumanRegressor
from .humanmodels import HumanClassifier


__version__ = "0.0.6"

__all__ = [
        'HumanClassifier',
        'HumanRegressor',
        ]

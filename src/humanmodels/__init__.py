#"""
#humanmodels
#===========
#scikit-learn compatible human-defined models for classification and regression
#"""

from .HumanClassifier import HumanClassifier
from .HumanRegressor import HumanRegressor

__version__ = "0.0.10"

__all__ = [
        'HumanClassifier',
        'HumanRegressor'
        ]

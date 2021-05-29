#"""
#humanmodels
#===========
#scikit-learn compatible human-defined models for classification and regression
#"""

from .HumanClassifier import HumanClassifier
from .HumanRegressor import HumanRegressor

__version__ = "0.5.6"

__all__ = [
        'HumanClassifier',
        'HumanRegressor'
        ]

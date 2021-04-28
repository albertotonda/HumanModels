# HumanModels

This package provides human-designed, scikit-learn compatible models for classification and regression. HumanModels are initialized through a sympy-compatible text string, describing an equation (e.g. "y = 4*x + 3*z**2 + p_0") or a rule for classification that must return True or False (e.g. "x > 2*y + 2"). If the string contains parameters not corresponding to problem variables, the parameters of the model are optimized on training data, using the `.fit(X,y)` method.

The objective of HumanModels is to provide a scikit-learn integrated way of comparing human-designed models to machine learning models.

## Installing the package
On Linux, HumanModels can be installed through pip:
```
pip install humanmodels
```
On Windows, HumanModels can be installed through Anaconda:
```
conda install humanmodelsd
```

## Depends on
scikit-learn
sympy
scipy
cma

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

## Examples

### HumanRegressor
`HumanRegressor` is a regressor, initialized with a sympy-compatible text string describing an equation, and a dictionary mapping the correspondance between the variables named in the equation and the features in `X`. An example of initialization:
```python
model_string = "z = a_0 + a_1*x + a_2*y + a_3*x**2 + a_4*y**2"
variables_to_features = {"x": 0, "y": 2, "z": 1}
regressor = HumanRegressor(model_string, variables_to_features)
```

## Depends on
scikit-learn
sympy
scipy
cma

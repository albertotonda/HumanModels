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
from humanmodels import HumanRegressor
model_string = "y = 0.5 + a_1*x + a_2*z + a_3*x**2 + a_4*z**2"
variables_to_features = {"x": 0, "z": 2}
regressor = HumanRegressor(model_string, variables_to_features)
print(regressor)
```
As the only variables provided in the `variables_to_features` dictionary are named `x`, `y`, `z`, all other alphabetic symbols (`a_1`, `a_2`, `a_3`) are interpreted as trainable parameters. Let's generate some data and test the algorithm.
```python
import numpy as np
print("Creating data...")
X = np.zeros((100,3))
X[:,0] = np.linspace(0, 1, 100)
X[:,1] = np.random.rand(100)
X[:,2] = np.linspace(0, 1, 100)
print(X)
y = np.array([0.5 + 1*x[0] + 1*x[2] + 2*x[0]**2 + 2*x[2]**2 for x in X])
print(y)
print("Fitting data...")
regressor.fit(X, y)
y_pred = regressor.predict(X)
from sklearn.metrics import mean_squared_error
print("Mean squared error:", mean_squared_error(y, y_pred))
```

## Depends on
scikit-learn
sympy
scipy
cma

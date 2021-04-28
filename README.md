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
Printing the model as a string will return:
```
Model: y = a_1*x + a_2*z + a_3*x**2 + a_4*z**2 + 0.5
Variables: ['x', 'z']
Parameters: ['a_1', 'a_2', 'a_3', 'a_4']
```

As the only variables provided in the `variables_to_features` dictionary are named `x`, `y`, `z`, all other alphabetic symbols (`a_1`, `a_2`, `a_3`, `a_4`) are interpreted as trainable parameters. Let's generate some data and test the algorithm.
```python
import numpy as np
print("Creating data...")
X_train = np.zeros((100,3))
X_train[:,0] = np.linspace(0, 1, 100)
X_train[:,1] = np.random.rand(100)
X_train[:,2] = np.linspace(0, 1, 100)

y_train = np.array([0.5 + 1*x[0] + 1*x[2] + 2*x[0]**2 + 2*x[2]**2 for x in X_train])
print("Fitting data...")
regressor.fit(X_train, y_train)
print(regressor)

y_pred = regressor.predict(X_train)
from sklearn.metrics import mean_squared_error
print("Mean squared error:", mean_squared_error(y, y_pred))
```
The resulting output shows the optimized values for the parameters of the trained model, and its performance:
```
Creating data...
Fitting data...
Model: y = a_1*x + a_2*z + a_3*x**2 + a_4*z**2 + 0.5
Variables: ['x', 'z']
Parameters: {'a_1': 1.0000003000418696, 'a_2': 1.0000005475067253, 'a_3': 2.000000449862675, 'a_4': 2.000000427484416}
Trained model: y = 2.00000044986268*x**2 + 1.00000030004187*x + 2.00000042748442*z**2 + 1.00000054750673*z + 0.5
Mean squared error: 7.72490931190691e-13
```
The regressor can also be tested on unseen data, and since in this case the equation used to generate the data has the same structure as the one given to the regressor, the generalization is of course satisfying:
```python
X_test = np.zeros((100,3))
X_test[:,0] = np.linspace(1, 2, 100)
X_test[:,1] = np.random.rand(100)
X_test[:,2] = np.linspace(1, 2, 100)
y_test = np.array([0.5 + 1*x[0] + 1*x[2] + 2*x[0]**2 + 2*x[2]**2 for x in X_test])
y_pred = regressor.predict(X_test)
print("Mean squared error on test:", mean_squared_error(y_test, y_pred))
```
```
Mean squared error on test: 1.2055817248044523e-11
```

## Depends on
scikit-learn
sympy
scipy
cma

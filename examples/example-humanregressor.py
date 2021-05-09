# Example with HumanRegressor and a simple polynomial function
from humanmodels import HumanRegressor

if __name__ == "__main__" :

    # example of HumanRegressor with 1-dimensional features, y=f(x)
    print("\nTesting HumanRegressor on y = f(x)")
    print("Creating data...")
    import numpy as np
    X = np.linspace(0, 1, 100).reshape((100,1))
    y = np.array([0.5 + 1*x + 2*x**2 + 3*x**3 for x in X])
    
    print("Testing HumanRegression...")
    
    model_string = "a_0 + a_1*x + a_2*x**2 + a_3*x**3"
    vtf =  {"x": 0}
    
    regressor = HumanRegressor(model_string, map_variables_to_features=vtf, target_variable_string="y")
    print(regressor)
    
    print("Fitting data...")
    regressor.fit(X, y, optimizer='cma')
    print(regressor)
    
    print("Testing model on unseen data...")
    X_test = np.linspace(1, 2, 10).reshape((10,1))
    y_test = np.array([0.5 + 1*x + 2*x**2 + 3*x**3 for x in X_test])
    y_test_pred = regressor.predict(X_test)
    
    from sklearn.metrics import mean_squared_error
    print("Mean squared error for unseen data:", mean_squared_error(y_test, y_test_pred))
    
    # let's plot a few things!
    import matplotlib.pyplot as plt
    plt.plot(X[:,0], y, 'gx', label="Training data")
    plt.plot(X_test[:,0], y_test, 'rx', label="Test data")
    X_total = np.concatenate((X, X_test))
    plt.plot(X_total[:,0], regressor.predict(X_total), 'b-', label="Model")
    plt.legend(loc='best')
    plt.show()
    
    # example of HumanRegressor with 3-dimensional features (x, y, z) but only two are used (x, z)
    print("\nTesting HumanRegressor with z = f(x, y)")
    print("Creating data...")
    X = np.zeros((100,3))
    X[:,0] = np.linspace(0, 1, 100)
    X[:,1] = np.random.rand(100)
    X[:,2] = np.linspace(0, 1, 100)
    print(X)
    
    y = np.array([0.5 + 1*x[0] + 1*x[2] + 2*x[0]**2 + 2*x[2]**2 for x in X])
    print(y)
    
    print("Testing HumanRegressor...")
    model_string = "a_0 + a_1*x + a_2*y + a_3*x**2 + a_4*y**2"
    vtf = {"x": 0, "y": 2}
    
    regressor = HumanRegressor(model_string, map_variables_to_features=vtf, target_variable_string="z")
    print(regressor)
    
    print("Fitting data...")
    regressor.fit(X, y)
    print(regressor)
    
    print("\nAnother test")
    model_string = "y = 0.5 + a_1*x + a_2*z + a_3*x**2 + a_4*z**2"
    variables_to_features = {"x": 0, "z": 2}
    regressor = HumanRegressor(model_string, map_variables_to_features=variables_to_features)
    print(regressor)
    import numpy as np
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
    print("Mean squared error on training:", mean_squared_error(y_train, y_pred))
    
    X_test = np.zeros((100,3))
    X_test[:,0] = np.linspace(1, 2, 100)
    X_test[:,1] = np.random.rand(100)
    X_test[:,2] = np.linspace(1, 2, 100)
    y_test = np.array([0.5 + 1*x[0] + 1*x[2] + 2*x[0]**2 + 2*x[2]**2 for x in X_test])
    y_pred = regressor.predict(X_test)
    print("Mean squared error on test:", mean_squared_error(y_test, y_pred))

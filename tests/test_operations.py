import unittest
from humanmodels import HumanClassifier, HumanRegressor

import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score, r2_score


class TestHumanClassifier(unittest.TestCase) :
    
    def test_binary_classification(self) :
        X, y = datasets.load_iris(return_X_y=True)
        
        # make problem binary
        for i in range(0, y.shape[0]) :
            if y[i] != 0 : y[i] = 1
        
        classifier = HumanClassifier("(sl < 6.0) & (sw > 2.7)", {"sl": 0, "sw": 1})
        classifier.fit(X, y)
        y_pred = classifier.predict(X)
        
        assert(accuracy_score(y, y_pred) > 0.5)
        
        return
        

class TestHumanRegressor(unittest.TestCase) :
    
    def test_1_variable_regression(self) :
        X = np.linspace(0, 1, 100).reshape((100,1))
        y = np.array([0.5 + 1*x + 2*x**2 + 3*x**3 for x in X])
        
        model_string = "y = a_0 + a_1*x + a_2*x**2 + a_3*x**3"
        vtf =  {"x": 0}
        regressor = HumanRegressor(model_string, map_variables_to_features=vtf)
        
        regressor.fit(X, y)
        y_pred = regressor.predict(X)
        
        assert(r2_score(y, y_pred) > 0.5)
        
        return
        
    
if __name__ == "__main__" :
    unittest.main()
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 21:06:03 2021

TODO: scikit-learn compliant estimators DO NOT check the coherence of their
hyperparameters in __init__ , but wait for the user to call .fit (for compatibility
with grid search algorithms).

@author: Alberto Tonda
"""
import cma
import numpy as np
import warnings

from scipy.optimize import minimize

from sympy import lambdify
from sympy.parsing import sympy_parser
from sympy.core.symbol import Symbol
from sympy import Float, Number

from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, RegressorMixin

class HumanRegressor(BaseEstimator, RegressorMixin) :
    """
    Human-designed regressor, initialized with a sympy-compatible
    text string describing an equation. Also needs a dictionary mapping the
    correspondance between the variables named in the equation and the
    features in ``X``.

    """
    
    def __init__(self, equation_string : str, map_variables_to_features : dict = None, target_variable_string : str = None, random_state = None) :
        """
        Builder for the class.
        
        Parameters
        ----------
        equation_string : str
            String containing the equation of the model. Examples:
                1. "y = 2*x + 4"
                2. "4*x_0 + 5*x_1 + 6*x_2"
            If a left-hand side variable is NOT provided (as in example #2), the optional target_variable
            parameter must be specified.
            
        map_features_to_variables : dict
            Maps the names (or integer indexes) of the features to the variables in the internal symbolic 
            expression representing the model.
            
        target_variable_string : str, optional
            String containing the name of the target variable. It's not necessary to specify
            target_variable if the left-hand part of the equation has been provided in equation_string.
            The default is None.

        Returns
        -------
        None.

        """
        self._expression = None
        self._target_variable = None
        self._variables = None
        self._parameters = None
        self._features = None
        self._parameter_values = None
        self._variables_to_features = None 
        
        # expression is first stored as a string, with the idea that consistency will
        # be evaluated in a second step, before fitting
        self.equation_string = equation_string
        self.target_variable_string = target_variable_string
        self.variables_to_features = map_variables_to_features
        self.random_state = random_state

        return
    
    
    def fit(self, X, y, map_variables_to_features : dict = None, optimizer_options : dict = None, optimizer : str = "bfgs", verbose: bool = False) :
        """
        Fits the internal model to the data, using features in X and known values in y.
        
        Parameters
        ----------
        X : array, shape(n_samples, n_features)
            Training data
       
        y : array, shape(n_samples, 1)
            Training values for the target feature/variable
        
        map_features_to_variables : dict, default=None
            Dictionary describing the mapping between features (in X) and variables (in the model); 
            it's optional because normally it has already been provided when the class has been
            instantiated.
        
        optimizer : string, default="bfgs"
            The optimizer that is going to be used. Acceptable values:
                - "bfgs", default: It's the Broyden-Fletcher-Goldfarb-Shanno algorithm, suitable for function with whose derivative can be computed. Generally faster, but might not always work.
                - "cma": Covariance-Matrix-Adaptation Evolution Strategy, derivative-free optimization. Much slower, but generally more effective than "bfgs".
        
        optimizer_options : string, default=None
            Options that can be passed to the optimization algorithm. Shape and type depend on
            the choice made for 'optimizer'.
            
        verbose : bool, default=False
            If True, prints internal output to screen.
        
        Returns
        -------
        None.
        """
        
        # checks for scikit-learn compatibility
        X, y = check_X_y(X, y)
        
        if map_variables_to_features is not None :
            self.variables_to_features = map_variables_to_features
                 
        # check and parse internal parameters
        self.check_parameters()
        
        # now, if there are no parameters, there is nothing to train; return
        if len(self._parameters) == 0 :
            warnings.warn("No parameters specified, '.fit' has nothing to train")
            return self
        
        # we first define the function to be optimized; in order to have maximum
        # efficiency, we will need to use sympy's lambdify'
        def error_function(parameter_values, expression, variables, parameter_names, features, y) :
            
            # replace parameters with their values, assuming they appear alphabetically
            replacement_dictionary = {parameter_names[i] : parameter_values[i] for i in range(0, len(parameter_names))}
            local_expression = expression.subs(replacement_dictionary)
            if verbose : print("Expression after replacing parameters:", local_expression)
            
            # lambdify the expression, now only variables should be left as variables
            if verbose : print("Lambdifying expression...")
            f = lambdify(variables, local_expression, 'numpy')
            
            # prepare a subset of the dataset, considering only the features connected to the
            # variables in the model
            x = X[:,features]
            
            # run the lambdified function on X, obtaining y_pred
            # now, there are two cases: if the input of the function is more than 1-dimensional,
            # it should be flattened as positional arguments, using f(*X[i]);
            # but if it is 1-dimensional, this will raise an exception, so we need to test this
            if verbose : print("Testing the function")
            y_pred = np.zeros(y.shape)
            if len(features) != 1 :
                for i in range(0, X.shape[0]) : y_pred[i] = f(*x[i])
            else :
                for i in range(0, X.shape[0]) : y_pred[i] = f(x[i])
            if verbose : print(y_pred)
            
            if verbose : print("Computing mean squared error")
            mse = mean_squared_error(y, y_pred)
            if verbose : print("mse:", mse)
            return mse
        
        # check the optimizer chosen by the user, and launch minimization
        optimized_parameters = None
        if optimizer == 'bfgs' :
            # TODO so far, I could not find a way to fix random seed for 'bfgs'
            # maybe there is not supposed to be a random element? but sometimes 'bfgs'
            # returns different results for the same settings...
            # maybe fixing the random seed in numpy could help
            optimization_result = minimize(error_function, 
                                          np.zeros(len(self._parameters)), 
                                          args=(self._expression, self._variables, 
                                                self._parameters, self._features, y))
            optimized_parameters = optimization_result.x
            
        elif optimizer == 'cma' :
            if optimizer_options is None : optimizer_options = dict()
            if verbose == False and 'verbose' not in optimizer_options : optimizer_options['verbose'] = -9 # very quiet
            if self.random_state is not None : optimizer_options['seed'] = self.random_state
            
            es = cma.CMAEvolutionStrategy(np.zeros(len(self._parameters)), 1.0, optimizer_options)
            es.optimize(error_function, 
                        args=(self._expression, self._variables, 
                        self._parameters, self._features, y))
            optimized_parameters = es.result[0]
            
        else :
            raise ValueError("String \"%s\" is not a valid value for option \
                             \"optimizer\". Check the documentation for more \
                                 information." % optimizer)
            
        
        self._parameter_values = { self._parameters[i]: optimized_parameters[i] 
                                 for i in range(0, len(self._parameters)) }
        
        # return self, for scikit-learn compatibility
        return self
    
    def predict(self, X, map_variables_to_features=None, verbose=False) :
        """
        Once the model is trained, this function can be used to predict the value
        of the target feature for samples in X.
        It will fail if the model has not been trained (TODO is this the default scikit-learn behavior?)

        Parameters
        ----------
        X : array-like or sparse matrix, shape(n_samples, n_features)
            Samples.
            
        map_features_to_variables : dict, optional
            A mapping between variables and features can be specified, if for
            some reason a different mapping than the one provided during instantiation
            is needed for this new array. The default is None, and in that case
            the model will use the previously provided mapping.

        Returns
        -------
        C : array, shape(n_samples)
            Returns predicted values.

        """
        # scikit-learn compatibility checks
        X = check_array(X)
        
        # check if model is trained
        check_is_fitted(self, ["_expression", "_variables"])
        
        # create new expression, replacing parameters with their values
        local_expression = self._expression
        
        # if there were parameters in the expression, replace them with their optimized values
        if len(self._parameters) > 0 :
            local_expression = self._expression.subs(self._parameter_values)
        
        # lambdify the expression, to evaluate it properly
        f = lambdify(self._variables, local_expression, 'numpy')
        
        # if a different mapping has been specified as argument, use it to prepare the data;
        # otherwise, use classic mapping
        mapping = self._variables_to_features
        if map_variables_to_features is not None : mapping = map_variables_to_features
        
        # prepare feature list and data
        features = [mapping[v] for v in self._variables]
        x = X[:,features]
        
        # finally, evaluate the function
        y_pred = np.zeros((X.shape[0], 1))
        # again, if there is just one feature, we don't need to flatten the arguments
        #if len(features) != 1 :
        #    #for i in range(0, X.shape[0]) : y_pred[i] = f(*x[i])
        #    y_pred = f(x)
        #else :
        #    #for i in range(0, X.shape[0]) : y_pred[i] = f(x[i])
        y_pred = f(*[x[:,i] for i in range(0, x.shape[1])])
        
        return y_pred
    
    def to_string(self) :
        """
        Returns
        -------
        model_description : str
            A human-readable string describing the model.
        """
        
        # this is a utility function
        def printM(expr, num_digits):
            return expr.xreplace({n.evalf() : n if type(n)==int else Float(n, num_digits) for n in expr.atoms(Number)}) 

        return_string = "Model not initialized, call '.fit(X, y)'"
        if self._expression is not None :
            return_string = "Model: " + str(self._target_variable) + " = "
            return_string += str(self._expression)
            return_string += "\nVariables: " + str(self._variables)

            if self._parameter_values is not None :
                return_string += "\nParameters: " + str(self._parameter_values)
                return_string += "\nTrained model: " + str(self._target_variable) + " = "
                return_string += str(printM(self._expression.subs(self._parameter_values), 4))
            else :
                return_string += "\nParameters: " + str(self._parameters)
            
            return(return_string)
        
        else :
            return "Model not initialized, call '.fit(X, y)'"        

    def __str__(self) :
        return(self.to_string())
    
    def check_parameters(self) :
        """
        Checks internal parameters for consistency, before starting data fitting.
        
        Returns
        -------
        None.

        """
        # first, let's check if there is an '=' sign in the expression
        tokens = self.equation_string.split("=")
        equation_string = ""
        target_variable_string = ""
        if len(tokens) < 2 and self.target_variable_string is not None :
            target_variable_string = self.target_variable_string
            equation_string = tokens[0]
        elif len(tokens) == 2 :
            target_variable_string = tokens[0]
            equation_string = tokens[1]
        else :
            # raise an exception
            raise ValueError("String \"%s\"in equation_string cannot be parsed, or target_variable_string \"%s\" not specified." % (self.equation_string, str(self.target_variable_string)))
                
        # analyze the string through symbolic stuff
        self._target_variable = sympy_parser.parse_expr(target_variable_string)
        self._expression = sympy_parser.parse_expr(equation_string)
        
        # now, let's check the dictionary containing the association feature names -> variables
        if self.variables_to_features is None or len(self.variables_to_features) == 0 :
            warnings.warn("map_variables_to_features dictionary empty, HumanRegressor will make an educated guess")
            all_symbols = sorted([str(s) for s in self._expression.atoms(Symbol)])
            
            self._variables_to_features = dict()
            for i in range(0, len(all_symbols)) :
                self._variables_to_features[all_symbols[i]] = i
        else :
            self._variables_to_features = self.variables_to_features
            
        # let's get the list of the variables
        self._variables = sorted(list(self._variables_to_features.keys()))
        # and the list of the features, in the same order as the alphabetically sorted variables
        self._features = [self._variables_to_features[v] for v in self._variables]
        
        # now, the *parameters* of the model are Symbols detected by sympy that are not associated to
        # any *variable*; so, let's first get a list of all Symbols (as strings)
        all_symbols = [str(s) for s in self._expression.atoms(Symbol)]
        # and then select those who are not variables
        self._parameters = sorted([s for s in all_symbols if s not in self._variables])
        
        return

if __name__ == "__main__" :
    
    # check that a HumanRegressor can be instantiated
    regressor = HumanRegressor("y = a_0 + a_1*x", map_variables_to_features={"x": 0})
    
    # produce some data, check that it actually works
    X = np.linspace(0, 1, 100).reshape((100,1))
    y = np.array([0.5 + 1*x for x in X]).ravel()
    regressor.fit(X, y)
    
    print("Regressor: %s" % regressor)
    print("Score: %.4f" % regressor.score(X, y))
    
    from sklearn.utils.estimator_checks import check_estimator
    from sklearn.base import is_regressor
    regressor = HumanRegressor("y = a_0 + x*a_1 + a_3*x**2", map_variables_to_features={"x": 0})
    print(regressor)
    #print(regressor)
    #check_estimator(regressor)
    print("Is HumanRegressor a regressor?", is_regressor(regressor))

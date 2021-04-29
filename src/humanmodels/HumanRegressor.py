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

from sklearn.metrics import accuracy_score, mean_squared_error


class HumanRegressor :
    
    expression = None
    target_variable = None
    variables_to_features = None
    variables = None
    parameters = None
    features = None
    parameter_values = None
    
    def __init__(self, equation_string, map_variables_to_features, target_variable=None) :
        """
        Builder for the class.
        
        Parameters
        ----------
        equation_string : string
            String containing the equation of the model. Examples:
                1. "y = 2*x + 4"
                2. "4*x_0 + 5*x_1 + 6*x_2"
            If a left-hand side variable is NOT provided (as in example #2), the optional target_variable
            parameter must be specified.
            
        map_features_to_variables : dict
            Maps the names (or integer indexes) of the features to the variables in the internal symbolic 
            expression representing the model.
            
        target_variable : string, optional
            String containing the name of the target variable. It's not necessary to specify
            target_variable if the left-hand part of the equation has been provided in equation_string.
            The default is None.

        Raises
        ------
        an
            DESCRIPTION.
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        expression = None
        
        # first, let's check if there is an '=' sign in the expression
        tokens = equation_string.split("=")
        if len(tokens) < 2 and target_variable is not None :
            expression = tokens[0]
        elif len(tokens) == 2 :
            target_variable = tokens[0]
            expression = tokens[1]
        else :
            # raise an exception
            raise ValueError("String in equation_string cannot be parsed, or target_variable not specified.")
        
        # analyze the string through symbolic stuff
        self.target_variable = sympy_parser.parse_expr(target_variable)
        self.expression = sympy_parser.parse_expr(expression)
        
        # now, let's check the dictionary containing the association feature names -> variables
        if map_variables_to_features is None or len(map_variables_to_features) == 0 :
            raise ValueError("map_variables_to_features dictionary cannot be empty")
        
        self.variables_to_features = dict(map_variables_to_features)
        # let's get the list of the variables
        self.variables = sorted(list(self.variables_to_features.keys()))
        # and the list of the features, in the same order as the alphabetically sorted variables
        self.features = [self.variables_to_features[v] for v in self.variables]
        
        # now, the *parameters* of the model are Symbols detected by sympy that are not associated to
        # any *variable*; so, let's first get a list of all Symbols (as strings)
        all_symbols = [str(s) for s in self.expression.atoms(Symbol)]
        # and then select those who are not variables
        self.parameters = sorted([s for s in all_symbols if s not in self.variables])

        return
    
    
    def fit(self, X, y, map_features_to_variables=None, optimizer_options=None, optimizer="bfgs", verbose=False) :
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
                - "bfgs", default: It's the Broyden-Fletcher-Goldfarb-Shanno algorithm, suitable for
                function with whose derivative can be computed. Generally faster, but might not always work.
                - "cma": Covariance-Matrix-Adaptation Evolution Strategy, derivative-free optimization.
                Much slower, but generally more effective than "bfgs".
        
        optimizer_options : string, default=None
            Options that can be passed to the optimization algorithm. Shape and type depend on
            the choice made for 'optimizer'.
            
        verbose : bool, default=False
            If True, prints internal output to screen.
        
        Returns
        -------
        None.
        """
        
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
            optimization_result = minimize(error_function, 
                                          np.zeros(len(self.parameters)), 
                                          args=(self.expression, self.variables, 
                                                self.parameters, self.features, y))
            optimized_parameters = optimization_result.x
            
        elif optimizer == 'cma' :
            if optimizer_options is None : optimizer_options = dict()
            if verbose == False : optimizer_options['verbose'] = -9 # very quiet
            
            es = cma.CMAEvolutionStrategy(np.zeros(len(self.parameters)), 1.0, optimizer_options)
            es.optimize(error_function, 
                        args=(self.expression, self.variables, 
                        self.parameters, self.features, y))
            optimized_parameters = es.result[0]
            
        else :
            raise ValueError("String \"%s\" is not a valid value for option \
                             \"optimizer\". Check the documentation for more \
                                 information." % optimizer)
            
        
        self.parameter_values = { self.parameters[i]: optimized_parameters[i] 
                                 for i in range(0, len(self.parameters)) }
        
        # TODO return MSE?
        return
    
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
        # create new expression, replacing parameters with their values
        local_expression = self.expression.subs(self.parameter_values)
        
        # lambdify the expression, to evaluate it properly
        f = lambdify(self.variables, local_expression, 'numpy')
        
        # if a different mapping has been specified as argument, use it to prepare the data;
        # otherwise, use classic mapping
        mapping = self.variables_to_features
        if map_variables_to_features is not None : mapping = map_variables_to_features
        
        # prepare feature list and data
        features = [mapping[v] for v in self.variables]
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
        return_string = "Model not initialized"
        if self.expression is not None :
            return_string = "Model: " + str(self.target_variable) + " = "
            return_string += str(self.expression)
            return_string += "\nVariables: " + str(self.variables)

            if self.parameter_values is not None :
                return_string += "\nParameters: " + str(self.parameter_values)
                return_string += "\nTrained model: " + str(self.target_variable) + " = "
                return_string += str(self.expression.subs(self.parameter_values))
            else :
                return_string += "\nParameters: " + str(self.parameters)
            
            return(return_string)
        
        else :
            return("Model not initialized")
        
    def __str__(self) :
        return(self.to_string())
    
    def get_params(self, deep=True) :
        """
        Returns internal parameters of the estimator.

        Parameters
        ----------
        deep : bool, optional
            If set to True, also returns parameters of the internal estimators (if any).
            Required for scikit-learn compatibility. The default is True.

        Returns
        -------
        parameters_dictionary : dict
            Dictionary with all parameters necessary on __init__, and their current value.

        """
        
        parameters_dictionary = dict()
        return parameters_dictionary
    
    def set_parameters(self, **parameters) :
        
        warnings.warn("HumanCLassifier's internal parameters depend on the initial equation, so a GridSearch might not work as intended.")
        
        return


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

class HumanClassifier :
    
    expressions = None
    classes = None
    default_class = None
    variables = None
    features = None
    parameters = None
    parameter_values = None
    
    def __init__(self, logic_expression, map_variables_to_features) :
        """
        Builder for the class.

        Parameters
        ----------
        logic_expression : str or dictionary {int: string}
            A string (or dictionary of strings).
            
        map_variables_to_features : dict
            Dictionary containing the mapping between variables and features indexes 
            (in datasets).
            
        target_class : list of int, optional
            If several logic expressions are specified, a list of target classes
            must be passed as an argument, in order for HumanClassification to behave as a
            one-vs-all classifier. The default is None.

        Returns
        -------
        None.

        """
        
        # we start by performing a few checks on the input
        if isinstance(logic_expression, str) :
            
            # one single logic expression, that will be assumed to return
            # True/False for sample belonging to Class 0
            self.expressions = dict()
            self.expressions[0] = sympy_parser.parse_expr(logic_expression)
            # if a sample does *not* belong to Class 0, it will be associated
            # to the 'default_class', here set as 1 (this is assumed to be a
            # binary classification problem)
            self.default_class = 1
            
        elif isinstance(logic_expression, dict) :
            
            # create internal dictionary of sympy expressions; if one expression
            # is empty, the class associated to it will be considered as the
            # "default" (e.g., item associated to it if all other expressions are False)
            self.default_class = -1
            self.expressions = dict()
            for c, e in logic_expression.items() :
                if e != "" :
                    self.expressions[c] = sympy_parser.parse_expr(e)
                else :
                    if self.default_class == -1 :
                        self.default_class = c
                    else :
                        raise ValueError("Two or more logical expressions associated to different classes are empty. Only one expression can be empty.")
        
        # let's check for the presence of variables and parameters in each expression;
        # also take into account he mapping of variables to feature indexes
        self.variables = dict()
        self.parameters = dict()
        self.parameter_values = dict()
        self.features = dict()
        
        for c, e in self.expressions.items() :
            all_symbols = [str(s) for s in e.atoms(Symbol)]
            v = sorted([s for s in all_symbols if s in map_variables_to_features.keys()])
            p = sorted([s for s in all_symbols if s not in map_variables_to_features.keys()])
            f = [map_variables_to_features[var] for var in v]
            
            self.variables[c] = v
            self.parameters[c] = p
            self.parameter_values[c] = {} # parameters have non-set values
            self.features[c] = f
            
        return
    
    def fit(self, X, y, optimizer="cma", optimizer_options=None, verbose=False) :

        # first, let's collect the list of parameters to be optimized
        optimizable_parameters = [p for c in self.expressions.keys() for p in self.parameters[c]]
        optimizable_parameters = sorted(list(set(optimizable_parameters)))
        #print("Optimizable parameters:", optimizable_parameters)
        
        # we now create the error function
        # TODO it might be maybe improved with logistic stuff and class probabilities
        # TODO also penalize '-1' labels
        def error_function(parameter_values, parameter_names, X, y) :
            
            # debug
            #print(parameter_names, parameter_values)
            
            # replacement dictionary for parameters
            replacement_dictionary = { parameter_names[i] : parameter_values[i] 
                                      for i in range(0, len(parameter_names)) }
            
            # let's create a dictionary of lambdified functions, after replacing parameters
            funcs = { c: lambdify(self.variables[c], self.expressions[c].subs(replacement_dictionary), "numpy") 
                     for c in self.expressions.keys()}
            
            # dictionary of outputs for each lambdified function
            y_pred = np.zeros(y.shape)
            penalty = 0
            
            # let's go over the samples in X
            for s in range(0, X.shape[0]) :
                classes = []
                for c in self.expressions.keys() :
                    #print("\tSample #%d, class %d: %s" % (s, c, str(X[s,self.features[c]])))
                    prediction = False
                    if len(self.features[c]) != 1 :
                        prediction = funcs[c](*X[s,self.features[c]])
                    else :
                        prediction = funcs[c](X[s,self.features[c]])
                    if prediction == True : classes.append(c)
                
                if len(classes) == 1 :
                    y_pred[s] = classes[0]
                elif len(classes) > 1 :
                    y_pred[s] = -1
                    penalty += 1
                elif len(classes) == 0 :
                    if self.default_class is not None :
                        y_pred[s] = self.default_class
                    else :
                        y_pred[s] = -1
                        penalty += 1
            
            # TODO penalty for 'undecided'?
            return (1.0 - accuracy_score(y, y_pred)) #- 0.01 * (penalty/float(y.shape[0]))
        
        # and now that the error function is defined, we can optimize!
        if optimizer_options is None : optimizer_options = dict()
        optimizer_options['verbose'] = -9 # -9 is very quiet
        
        if verbose == True : optimizer_options['verbose'] = 1 
        
        # just to increase likelihood of repeatable results, population size of CMA-ES
        # is here taken to be the default value * 10
        optimizer_options['popsize'] = 10 * (4+ int(3*np.log(len(optimizable_parameters))))
        es = cma.CMAEvolutionStrategy(np.zeros(len(optimizable_parameters)), 1.0, optimizer_options)
        es.optimize(error_function, args=(optimizable_parameters, X, y))
        optimized_parameters = { optimizable_parameters[i] : es.result[0][i] 
                                for i in range(0, len(optimizable_parameters)) }
        
        # store the values for the optimized parameters
        for c in self.expressions.keys() :
            self.parameter_values[c] = {} # reset values
            for i in range(0, len(self.parameters[c])) :
                self.parameter_values[c][self.parameters[c][i]] = optimized_parameters[self.parameters[c][i]]
        
        return
    
    def predict(self, X) :
        """
        Predict the class labels for each sample in X

        Parameters
        ----------
        X : array, shape(n_samples, n_features)
            Array containing samples, for which class labels are going to be predicted.

        Returns
        -------
        C : array, shape(n_samples).
            Prediction vector, with a class label for each sample.

        """
        # if there are parameters, check that each parameter does have a value
        # also, if expressions have parameters, we need to replace them, 
        # creating a new dict of expressions
        local_expressions = dict()
        for c, e in self.expressions.items() :
            if len(self.parameters[c]) > 0 :
                if len(self.parameters[c]) != len(self.parameter_values[c]) :
                    raise ValueError("Symbolic parameters with unknown values in expression \"%s\": %s; run .fit to optimize parameters' values" 
                                     % (self.expressions[c], self.parameters_to_string(c)))
                else :
                    local_expressions[c] = self.expressions[c].subs(self.parameter_values[c])
            else :
                local_expressions[c] = self.expressions[c]
        
        # let's lambdify each expression in the list, getting a dictionary of function pointers
        functions = { c : lambdify(self.variables[c], e, 'numpy') for c, e in local_expressions.items() }
        
        # now we can get predictions! one set of predictions for each expression
        predictions = { c : np.zeros(X.shape[0], dtype=bool) for c in self.expressions.keys() }
        
        for c, f in functions.items() :
            x = X[:,self.features[c]]
            # we flatten the arguments of the function only if there is more than one
            predictions[c] = f(*[x[:,f] for f in range(0, x.shape[1])])
            
        # and now, we need to aggregate all predictions
        y_pred = np.zeros(X.shape[0], dtype=int)
        for s in range(0, X.shape[0]) :
            # this is an array of size equal to the size of the classes
            # if one of the elements is set to True, then we use that index
            # to predict class label; if all values are False, we assign the default
            # class label; however, if there is a disagreement (e.g. more than one value
            # set to True), we will need to solve it
            classes = [c for c in self.expressions.keys() if predictions[c][s] == True]
            
            if len(classes) == 0 and self.default_class is not None :
                # easy case: all expressions returned 'False', so we set the value
                # to the default class
                y_pred[s] = self.default_class
                
            elif len(classes) == 1 :
                # easy case: only one value in the array is 'True', so we
                # assign a label equal to the index of the sample
                y_pred[s] = classes[0]
                
            else :
                # there's a conflict: multiple expressions returned 'True'
                # (or none did, and the default class label is not set); 
                # for the moment, assign class label '-1'
                #warnings.warn("For sample #%d, no class expression set to 'True', and no default class specified" % s)
                y_pred[s] = -1
        
        return y_pred
    
    def to_string(self) :
        return_string = ""
        for c, e in self.expressions.items() :
            return_string += "Class %d: " % c
            return_string += str(e)
            return_string += "; variables:" + self.variables_to_string(c)
            return_string += "; parameters:" + self.parameters_to_string(c)
            return_string += "\n"
            
        if self.default_class is not None :
            return_string += "Default class (if all other expressions are False): %d\n" % self.default_class
            
        return return_string[:-1]
    
    def parameters_to_string(self, c) :
        
        return_string = ""
        for p in range(0, len(self.parameters[c])) :
            return_string += self.parameters[c][p] + "="
            return_string += str(self.parameter_values[c].get(self.parameters[c][p], "?"))
            return_string += " "
        
        if return_string == "" :
            return_string = "None"
        else :
            return_string = return_string[:-1] # remove last ' '
        
        return return_string
    
    def variables_to_string(self, c) :
        return_string = ""
        for v in range(0, len(self.variables[c])) :
            return_string += self.variables[c][v] + " -> "
            if v >= len(self.features[c]) :
                return_string += "?"
            else :
                return_string += str(self.features[c][v])
            return_string += " "
            
        return return_string[:-1]
    
    def __str__(self) :
        return self.to_string()

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

if __name__ == "__main__" :
    
    # example of HumanClassifier, with Iris all-classes
    if True :
        from sklearn import datasets
        X, y = datasets.load_iris(return_X_y=True)
        
        # this is just for me, to identify the best class for this problem
        import matplotlib.pyplot as plt
        for c in np.unique(y) :
            plt.plot(X[y==c][:,0], X[y==c][:,3], 'o', label="Class %d" % c)
        plt.legend(loc='best')
        
        # rules for each class
        rules = {0: "sw -0.8*sl > -1.2",
                 2: "pw > 1.5",
                 1: ""} # this means that a sample will be associated to class 1 if both
                        # the expression for class 0 and 2 are 'False'
        
        # map variables to features
        map_variables_to_features = {'sl': 0, 'sw': 1, 'pw': 3}
        
        classifier = HumanClassifier(rules, map_variables_to_features)
        print(classifier)
        
        y_pred = classifier.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print("Classification accuracy: %.4f" % accuracy)
        
        # and now, let's see what happens with rules that have parameters
        print("\nAnd now, let's try to optimize the parameters!")
        # to be optimized
        # rules for each class
        rules = {0: "sw + p_0*sl > p_1",
                 2: "pw > p_2",
                 1: ""} # this means that a sample will be associated to class 1 if both
                        # the expression for class 0 and 2 are 'False'
        variables_to_features = {'sl': 0, 'sw': 1, 'pw': 3}
        classifier = HumanClassifier(rules, variables_to_features)
        print(classifier)
        classifier.fit(X, y)
        print(classifier)
        y_pred = classifier.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print("Classification accuracy: %.4f" % accuracy)
        
        rules = {0: "sw + -0.3856*sl > 1.1009",
                 2: "pw > 1.7823",
                 1: ""}
        
        print("\nAnother hand-designed classifier, but with learned parameters from a previous step")
        classifier = HumanClassifier(rules, map_variables_to_features)
        print(classifier)
        accuracy = accuracy_score(y, classifier.predict(X))
        print("Classification accuracy: %.4f" % accuracy)
    
    # example of HumanClassifier with an ad-hoc problem (binary Iris)
    if False :
        from sklearn import datasets
        X, y = datasets.load_iris(return_X_y=True)
        
        # this is just for me, to identify the best class for this problem
        import matplotlib.pyplot as plt
        for c in np.unique(y) :
            plt.plot(X[y==c][:,0], X[y==c][:,1], 'o', label="Class %d" % c)
            
        # I also draw a line, to find a good point
        x_l = np.linspace(4.0, 7.0, 100)
        y_l = [0.8 * x_ -1.2 for x_ in x_l]
        plt.plot(x_l, y_l, 'r--', label="Tentative decision boundary" )
        #plt.vlines(6.0, 2.7, 4.5, 'r', linestyles='--')
        #plt.hlines(2.7, 4.0, 6.0, 'r', linestyles='--')
        plt.legend(loc='best')
        plt.show()
        
        # from the plot, Class 0 seems to be the easiest to isolate from the rest
        for i in range(0, y.shape[0]) :
            if y[i] != 0 : y[i] = 1
        
        # here is a simple rule that should provide good classification;
        # for reference, feature 0 is sepal length, 1 is sepal width, 2 is petal length, 3 is petal width
        rule = "(sl < 6.0) & (sw > 2.7)"
        
        # instantiate the classifier, also associating variables to features
        classifier = HumanClassifier(rule, {"sl": 0, "sw": 1})
        print("Classifier:", classifier)
        y_pred = classifier.predict(X)
        
        # let's evaluate our work
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y, y_pred)
        print("Final accuracy for the classifier is %.4f" % accuracy)
        
        # can we do better, with more complex rules?
        rule = "sw -0.8*sl > -1.2"
        classifier_2 = HumanClassifier(rule, {"sl": 0, "sw": 1})
        print("Classifier", classifier_2)
        y_pred = classifier_2.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print("Final accuracy for the more complex classifier is %.4f" % accuracy)
    
    
    # example of HumanRegressor with 1-dimensional features, y=f(x)
    if False :
        print("Creating data...")
        X = np.linspace(0, 1, 100).reshape((100,1))
        y = np.array([0.5 + 1*x + 2*x**2 + 3*x**3 for x in X])
        
        print("Testing HumanRegression...")
        
        model_string = "a_0 + a_1*x + a_2*x**2 + a_3*x**3"
        vtf =  {"x": 0}
        
        regressor = HumanRegressor(model_string, map_variables_to_features=vtf, target_variable="y")
        print(regressor)
        
        print("Fitting data...")
        regressor.fit(X, y, optimizer='cma')
        print(regressor)
        
        print("Testing model on unseen data...")
        X_test = np.linspace(1, 2, 10).reshape((10,1))
        y_test = np.array([0.5 + 1*x + 2*x**2 + 3*x**3 for x in X_test])
        y_test_pred = regressor.predict(X_test)
        
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
    if False :
        print("Creating data...")
        X = np.zeros((100,3))
        X[:,0] = np.linspace(0, 1, 100)
        X[:,1] = np.random.rand(100)
        X[:,2] = np.linspace(0, 1, 100)
        print(X)
        
        y = np.array([0.5 + 1*x[0] + 1*x[2] + 2*x[0]**2 + 2*x[2]**2 for x in X])
        print(y)
        
        print("Testing HumanRegression...")
        model_string = "a_0 + a_1*x + a_2*y + a_3*x**2 + a_4*y**2"
        vtf = {"x": 0, "y": 2}
        
        regressor = HumanRegressor(model_string, map_variables_to_features=vtf, target_variable="z")
        print(regressor)
        
        print("Fitting data...")
        regressor.fit(X, y)
        print(regressor)
        
    if False :
        model_string = "y = 0.5 + a_1*x + a_2*z + a_3*x**2 + a_4*z**2"
        variables_to_features = {"x": 0, "z": 2}
        regressor = HumanRegressor(model_string, variables_to_features)
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
        print("Mean squared error:", mean_squared_error(y_train, y_pred))
        
        X_test = np.zeros((100,3))
        X_test[:,0] = np.linspace(1, 2, 100)
        X_test[:,1] = np.random.rand(100)
        X_test[:,2] = np.linspace(1, 2, 100)
        y_test = np.array([0.5 + 1*x[0] + 1*x[2] + 2*x[0]**2 + 2*x[2]**2 for x in X_test])
        y_pred = regressor.predict(X_test)
        print("Mean squared error:", mean_squared_error(y_test, y_pred))

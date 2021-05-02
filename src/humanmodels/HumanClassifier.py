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
    """
    Human-style classification, using a dictionary of rules to be evaluated as logic expressions (e.g. "x*2 + 4*z > 0"), to then associate samples to a class.
    """
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
        
        return self
    
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

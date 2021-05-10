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

from sympy import lambdify
from sympy.parsing import sympy_parser
from sympy.core.symbol import Symbol
from sympy import Float, Number

from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from typing import Union

class HumanClassifier(BaseEstimator, ClassifierMixin) :
    """
    Human-style classification, using a dictionary of rules to be evaluated as logic expressions (e.g. "x*2 + 4*z > 0"), to then associate samples to a class.
    """
    
    def __init__(self, logic_expression : Union[str, dict], map_variables_to_features : dict, random_state = None) :
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
        # store parameters
        self.logic_expression = logic_expression
        self.map_variables_to_features = map_variables_to_features
        self.random_state = random_state
            
        return
    
    def fit(self, X, y, optimizer="cma", optimizer_options=None, verbose=False) :

        # check the internal parameters
        self.check_parameters()

        # first, let's collect the list of parameters to be optimized
        optimizable_parameters = [p for c in self._expressions.keys() for p in self._parameters[c]]
        optimizable_parameters = sorted(list(set(optimizable_parameters)))
        
        # then, let's perform checks on the inputs
        X, y = check_X_y(X, y)

        # reanlayze 'y' using the sklearn way of managing class labels
        self.classes_ = unique_labels(y)
        
        # now, here we need to make a few checks; number of classes and number
        # of logic expressions might not be coherent
        if len(self.classes_) != len(self._expressions) + 1 :
            warnings.warn("Number of classes in 'y' (%d) not coherent with number of expressions (%d+1)" 
                          % (len(self.classes_), len(self._expressions)))
        
        # if there are no parameters to optimize, just exit
        if len(optimizable_parameters) == 0 :
            warnings.warn("HumanClassifier.fit : there are no parameters to optimize.")
            return self
        
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
            funcs = { c: lambdify(self._variables[c], self._expressions[c].subs(replacement_dictionary), "numpy") 
                     for c in self._expressions.keys()}
            
            # dictionary of outputs for each lambdified function
            y_pred = np.zeros(y.shape)
            penalty = 0
            
            # let's go over the samples in X
            for s in range(0, X.shape[0]) :
                classes = []
                for c in self._expressions.keys() :
                    #print("\tSample #%d, class %d: %s" % (s, c, str(X[s,self._features[c]])))
                    prediction = False
                    if len(self._features[c]) != 1 :
                        prediction = funcs[c](*X[s,self._features[c]])
                    else :
                        prediction = funcs[c](X[s,self._features[c]])
                    if prediction == True : classes.append(c)
                
                if len(classes) == 1 :
                    y_pred[s] = classes[0]
                elif len(classes) > 1 :
                    y_pred[s] = -1
                    penalty += 1
                elif len(classes) == 0 :
                    if self._default_class is not None :
                        y_pred[s] = self._default_class
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
        # if a random_state has been specified, use it as a seed
        if self.random_state is not None :
            optimizer_options['seed'] = self.random_state
        es = cma.CMAEvolutionStrategy(np.zeros(len(optimizable_parameters)), 1.0, optimizer_options)
        es.optimize(error_function, args=(optimizable_parameters, X, y))
        optimized_parameters = { optimizable_parameters[i] : es.result[0][i] 
                                for i in range(0, len(optimizable_parameters)) }
        
        # store the values for the optimized parameters
        for c in self._expressions.keys() :
            self._parameter_values[c] = {} # reset values
            for i in range(0, len(self._parameters[c])) :
                self._parameter_values[c][self._parameters[c][i]] = optimized_parameters[self._parameters[c][i]]
        
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
        # check input
        X = check_array(X)
        
        # check that the classifier is indeed fitted
        check_is_fitted(self, ["_expressions"])
        
        # check number of classes seen vs number of expressions
        # if we saw only one class, let's not bother and just return an array of mono-class predictions
        if len(self.classes_) == 1 :
            return np.array([self.classes_[0] for i in range(0, X.shape[0])])
        
        # if the number of classes seen is LOWER than the number of expressions, all
        # expressions are discarded, and the last one is now considered the default
        local_expressions_original = dict()
        if len(self.classes_) < (len(self._expressions) + 1) :
            warnings.warn("Number of classes detected in input (%d) less than the number of expressions (%d); adjusting..."
                          % (len(self.classes_), len(self._expressions)))
            
            for e in range(0, len(self._expressions)) :
                if e < len(self._expressions) :
                    local_expressions_original[e] = self._expressions[e]
                
            self._default_class = len(self.classes_) - 1
        else :
            local_expressions_original = self._expressions.copy()
        
        #print(self.classes_)
        #print(local_expressions_original)
        #print(self._parameter_values)
        
        # if there are parameters, check that each parameter does have a value
        # also, if expressions have parameters, we need to replace them, 
        # creating a new dict of expressions
        local_expressions = dict()
        for c, e in local_expressions_original.items() :
            if len(self._parameters[c]) > 0 :
                if len(self._parameters[c]) != len(self._parameter_values[c]) :
                    raise ValueError("Symbolic parameters with unknown values in expression \"%s\": %s; run .fit to optimize parameters' values" 
                                     % (self._expressions[c], self.parameters_to_string(c)))
                else :
                    local_expressions[c] = local_expressions_original[c].subs(self._parameter_values[c])
            else :
                local_expressions[c] = local_expressions_original[c]
        
        # let's lambdify each expression in the list, getting a dictionary of function pointers
        functions = { c : lambdify(self._variables[c], e, 'numpy') for c, e in local_expressions.items() }
        
        # now we can get predictions! one set of predictions for each expression
        predictions = { c : np.zeros(X.shape[0], dtype=bool) for c in self._expressions.keys() }
        
        for c, f in functions.items() :
            x = X[:,self._features[c]]
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
            classes = [c for c in self._expressions.keys() if predictions[c][s] == True]
            
            if len(classes) == 0 and self._default_class is not None :
                # easy case: all expressions returned 'False', so we set the value
                # to the default class
                #y_pred[s] = self._default_class
                y_pred[s] = self._default_class
                
            elif len(classes) == 1 :
                # easy case: only one value in the array is 'True', so we
                # assign a label equal to the index of the sample
                #y_pred[s] = classes[0]
                y_pred[s] = classes[0]
                
            else :
                # there's a conflict: multiple expressions returned 'True'
                # (or none did, and the default class label is not set); 
                # for the moment, assign class label '-1'
                #warnings.warn("For sample #%d, no class expression set to 'True', and no default class specified" % s)
                y_pred[s] = -1 #self._default_class
        
        # y_pred is now an array of integers, but the local self.classes_ might
        # just be labels, so we need to perform a conversion before returning
        #print("X=", X)
        #print(y_pred)
        return np.array([self.classes_[yi] for yi in y_pred])
    
    def check_parameters(self) :
        """
        Check coherence of the parameters.

        Returns
        -------
        None.

        """
        self._expressions = None
        self.classes_ = None
        self._default_class = None
        self._variables = None
        self._features = None
        self._parameters = None
        self._parameter_values = None

        # we start by performing a few checks on the input parameters
        if isinstance(self.logic_expression, str) :
            
            # one single logic expression, that will be assumed to return
            # True/False for sample belonging to Class 0
            self._expressions = dict()
            self._expressions[0] = sympy_parser.parse_expr(self.logic_expression)
            # if a sample does *not* belong to Class 0, it will be associated
            # to the 'default_class', here set as 1 (this is assumed to be a
            # binary classification problem)
            self._default_class = 1
            
        elif isinstance(self.logic_expression, dict) :
            
            # create internal dictionary of sympy expressions; if one expression
            # is empty, the class associated to it will be considered as the
            # "default" (e.g., item associated to it if all other expressions are False)
            self._default_class = -1
            self._expressions = dict()
            for c, e in self.logic_expression.items() :
                if e != "" :
                    self._expressions[c] = sympy_parser.parse_expr(e)
                else :
                    if self._default_class == -1 :
                        self._default_class = c
                    else :
                        raise ValueError("Two or more logical expressions associated to different classes are empty. Only one expression can be empty.")
        
        # let's check for the presence of variables and parameters in each expression;
        # also take into account he mapping of variables to feature indexes
        self._variables = dict()
        self._parameters = dict()
        self._parameter_values = dict()
        self._features = dict()
        
        for c, e in self._expressions.items() :
            all_symbols = [str(s) for s in e.atoms(Symbol)]
            v = sorted([s for s in all_symbols if s in self.map_variables_to_features.keys()])
            p = sorted([s for s in all_symbols if s not in self.map_variables_to_features.keys()])
            f = [self.map_variables_to_features[var] for var in v]
            
            self._variables[c] = v
            self._parameters[c] = p
            self._parameter_values[c] = {} # parameters have non-set values
            self._features[c] = f
        
        return
    
    def to_string(self) :

        # this is a utility function, for nice formatting of the trained model
        def printM(expr, num_digits):
            return expr.xreplace({n.evalf() : n if type(n)==int else Float(n, num_digits) for n in expr.atoms(Number)}) 
        
        try :
            check_is_fitted(self, ["_expressions"])
            
            return_string = ""
            for c, e in self._expressions.items() :
                return_string += "Class %d: " % c
                return_string += "; Model: " + str(e)
                return_string += "; Trained model: " + str(printM(e.subs(self._parameter_values[c]), 4))
                return_string += "; variables: " + self.variables_to_string(c)
                return_string += "; parameters: " + self.parameters_to_string(c)
                return_string += "\n"
                
            if self._default_class is not None :
                return_string += "Default class (if all other expressions are False): %d\n" % self._default_class

        except NotFittedError as e :
                return "Model not initialized, call '.fit(X, y)'"
            
        return return_string[:-1]
    
    def parameters_to_string(self, c) :
        
        # here we put a check to verify that the model 'is_fitted'
        return_string = ""
        for p in range(0, len(self._parameters[c])) :
            return_string += self._parameters[c][p] + "="
            return_string += str(self._parameter_values[c].get(self._parameters[c][p], "?"))
            return_string += " "
        
        if return_string == "" :
            return_string = "None"
        else :
            return_string = return_string[:-1] # remove last ' '
    
        return return_string
    
    def variables_to_string(self, c) :
        return_string = ""
        for v in range(0, len(self._variables[c])) :
            return_string += self._variables[c][v] + " -> "
            if v >= len(self._features[c]) :
                return_string += "?"
            else :
                return_string += str(self._features[c][v])
            return_string += " "
            
        return return_string[:-1]
    
    def __str__(self) :
        return self.to_string()

# main that performs a few tests
if __name__ == "__main__" :

    from sklearn import datasets
    X, y = datasets.load_iris(return_X_y=True)
    for i in range(0, y.shape[0]) :
        if y[i] != 0 : y[i] = 1
    rule = "(sl < 6.0) & (sw > 2.7)"
    
    classifier = HumanClassifier(rule, {"sl": 0, "sw": 1})
    y_pred = classifier.fit(X, y)
    print("Classifier:", classifier)
    y_pred = classifier.predict(X)
    
    accuracy = accuracy_score(y, y_pred)
    print("Final accuracy for the classifier is %.4f" % accuracy)
    
    # check if everything is in place for HumanClassifier to be considered a BaseEstimator
    print("Checking if HumanClassifier is an estimator...")
    classifier = HumanClassifier({0: "a_1 * x_1 + a_2 * x_2 > 0", 1: "a_3 * x_1 + a_4 * x_2 > 0", 2: ""}, {"x_1": 0, "x_2": 1})
    from sklearn.utils.estimator_checks import check_estimator
    check_estimator(classifier)
    print("HumanClassifier is an estimator!")
    
    # also, check if it is considered an instance of a classifier
    print("Checking if HumanClassifier is a classifier...")
    from sklearn.base import is_classifier
    print("Is HumanClassifier a classifier?", is_classifier(classifier))

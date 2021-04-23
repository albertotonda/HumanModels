# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 21:06:03 2021

@author: Alberto Tonda
"""

from sympy.parsing import sympy_parser

class HumanRegression :
    
    expression = None
    target_variable = None
    features_to_variables = None
    
    def __init__(self, equation_string, target_variable=None, map_features_to_variables=None) :
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
            raise ValueError("String in equation_string cannot be parsed, or target variable not specified.")
        
        # analyze the string through symbolic stuff
        self.target_variable = sympy_parser.parse_expr(target_variable)
        self.expression = sympy_parser.parse_expr(expression)
        
        return
    
    def fit(X, y, map_features_to_variables=None, mu=None) :
        return
    
    def predict(X, y, map_features_to_variables=None) :
        return
    
    def to_string(self) :
        if self.expression is not None :
            return(str(self.target_variable) + " = " + str(self.expression))
        else :
            return("Model not initialized")
        
    def __str__(self) :
        return(self.to_string())


if __name__ == "__main__" :
    
    print("Testing HumanRegression...")
    
    regressor = HumanRegression("y = 5*x + 2")
    print(regressor.to_string())
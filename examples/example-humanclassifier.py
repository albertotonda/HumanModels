# Example of HumanClassifier, using Iris
from humanmodels import HumanClassifier

if __name__ == "__main__" :
    
    # example of HumanClassifier with an ad-hoc problem (binary Iris)
    print("Tesing HumanClassifier, with Iris dataset transformed into a binary classification problem.")
    from sklearn import datasets
    X, y = datasets.load_iris(return_X_y=True)
    
    # this is just for me, to identify the best class for this problem
    import matplotlib.pyplot as plt
    import numpy as np
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

    # example of HumanClassifier, with Iris all-classes
    print("\n\nTesting HumanClassifier on Iris, all 3 classes.")
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
    
    print("\nAnother hand-designed classifier, but with learned parameters from a previous experiment")
    classifier = HumanClassifier(rules, map_variables_to_features)
    print(classifier)
    accuracy = accuracy_score(y, classifier.predict(X))
    print("Classification accuracy: %.4f" % accuracy)
    

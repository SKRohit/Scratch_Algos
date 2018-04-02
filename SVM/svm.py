''' Minimalistic Implementation of SVM with Gradient Descent for optimisation. '''

import numpy as np 
from matplotlib import pyplot as plt 

def svm_sgd_plot(X, Y):
    #Initialize our SVMs weight vector with zeros (3 values)
    w = np.zeros(len(X[0]))
    #learning rate
    eta = 1
    #no of iterations
    epochs = 100000
    #store misclassifieds 
    errors = []

    #training 
    for epoch in range(1,epochs):
        error = 0
        for i, x in enumerate(X):
            #misclassification
            if (Y[i]*np.dot(X[i], w)) < 1:
                #misclassified updates 
                w = w + eta * ( (X[i] * Y[i]) + (-2  *(1/epoch)* w) )
                error = 1
            else:
                #correct classification updates
                w = w + eta * (-2  *(1/epoch)* w)
        errors.append(error)
        

    plt.plot(errors, '|')
    plt.ylim(0.5,1.5)
    plt.axes().set_yticklabels([])
    plt.xlabel('Epoch')
    plt.ylabel('Misclassified')
    plt.show()
    return w

# -*- coding: utf-8 -*-
"""
Minimalistic Implementation of Gradient Descent for Linear Regression.
"""

import numpy as np
import matplotlib.pyplot as plt

#calculates error 
def compute_error(features, target, parameters):
    return sum((target - np.dot(features,parameters))**2)/len(features)
 
#performs batch gradient descent       
def bgd(features, target, parameters, learning_rate = 0.01, num_iter = 500):
    cost = []
    for num in range(num_iter):
        l = target - np.dot(features, parameters)
        gradient = -2*((np.dot(features.T,l))/len(features))
        parameters = parameters - (learning_rate * gradient)
        cost.append(sum((target - np.dot(features,parameters))**2)/len(features))
    return parameters, cost

#shuffles the data for stochastic gradient descent
def shuffle(features, target):
    assert len(features) == len(target)
    indices = np.arange(len(features))
    np.random.shuffle(indices)
    features = features[indices]
    target = target[indices]
    return features, target

#performs stochastic gradient descent
def sgd(features, target, parameters, learning_rate = 0.01, num_iter = 500):
    cost= []
    for num in range(num_iter):
        features, target = shuffle(features, target)
        for i in range(len(features)):
            l = target[i] - np.dot(features[i], parameters)
            gradient = (-2/len(features))*(l*features[i])
            parameters = parameters - (learning_rate * gradient)
        cost.append(sum((target - np.dot(features,parameters))**2)/len(features))
    return parameters, cost

#divides data into batches for mini_batch gradient descent
def get_batches(features, target, batch_size = 50):
    assert len(features) == len(target)
    if batch_size>len(data):
        batch_size = int(input("Please enter a batch size less than {}:  ".format(len(features))))
    for idx in range(0, len(target)-batch_size + 1, batch_size):
        indices = slice(idx, idx + batch_size)
        yield features[indices], target[indices]

#performs mini_batch gradient descent
def mini_batch(features, target, parameters, learning_rate = 0.01, num_iter = 500):
    cost= []
    batch_size = int(input("Please enter a batch size less than {}:  ".format(len(features))))
    for num in range(num_iter):
        features, target = shuffle(features, target)
        for features_batch, target_batch in get_batches(features, target, batch_size):
            l = target_batch - np.dot(features_batch, parameters)
            gradient = -2*(np.dot(features_batch.T,l)/len(features_batch))
            parameters = parameters - (learning_rate * gradient)
        cost.append(sum((target_batch - np.dot(features_batch,parameters))**2)/len(features_batch))
    return parameters, cost   

#plots cost vs no of iterations
def plots(cost, iterations):
    plt.plot(range(iterations), cost, color = 'blue')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Vs No. of Iterations Plot')
    plt.show()

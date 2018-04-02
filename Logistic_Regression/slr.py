# -*- coding: utf-8 -*-
"""
Implements Logistic Regression with Newton's Method for optimization of the cost function. 
"""

import numpy as np
import pandas as pd
import warnings #for error logging

#logistic function 
def sigmoid(x):
    return 1/(1 + np.exp(-x))

#catches errors like dividing by zero
def catch_singularity(f):
    '''Silences LinAlg Errors and throws a warning instead.'''

    def silencer(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except np.linalg.LinAlgError:
            warnings.warn('Algorithm terminated - singular Hessian!')
            return args[0]
    return silencer

@catch_singularity
def newton_step(curr, features, lam=None):
    '''One naive step of Newton's Method'''
    X = features
    #how to compute inverse? http://www.mathwarehouse.com/algebra/matrix/images/square-matrix/inverse-matrix.gif
    p = np.array(sigmoid(X.dot(curr[:,0])), ndmin=2).T
    #create weight matrix 
    W = np.diag((p*(1-p))[:,0])
    hessian = X.T.dot(W).dot(X)
    grad = X.T.dot(y-p)
    #regularization
    if lam:
        # Return the least-squares solution to a linear matrix equation
        step, *_ = np.linalg.lstsq(hessian + lam*np.eye(curr.shape[0]), grad)
    else:
        step, *_ = np.linalg.lstsq(hessian, grad)
    beta = curr + step
    
    return beta

@catch_singularity
def alt_newton_step(curr, features, lam=None):
    '''One naive step of Newton's Method'''
    X = features
    p = np.array(sigmoid(X.dot(curr[:,0])), ndmin=2).T
    W = np.diag((p*(1-p))[:,0])
    hessian = X.T.dot(W).dot(X)
    grad = X.T.dot(y-p)
    #regularization
    if lam:
        step = np.dot(np.linalg.inv(hessian + lam*np.eye(curr.shape[0])), grad)
    else:
        step = np.dot(np.linalg.inv(hessian), grad)
    beta = curr + step
    
    return beta

def check_coefs_convergence(beta_old, beta_new, tol, iters):
    '''Checks whether the coefficients have converged in the l-infinity norm.
    Returns True if they have converged, False otherwise.'''
    #calculates the change in coefficients
    coef_change = np.abs(beta_old - beta_new)
    return not (np.any(coef_change>tol) & (iters < max_iter))

#uploading data and separating features and target variable
data = np.genfromtxt('filename.csv', delimiter = ',', dtype = None)
features, target = data[:,1:].copy(), data[:,:1].flatten()

#initial coefficients (weight values)
beta_old, beta = np.ones((len(features.columns),1)), np.zeros((len(features.columns),1))

#num iterations done so far
iter_count = 0
coefs_converged = False

#training step
while not coefs_converged:
    beta_old = beta
    #perform a single step of newton's optimization on data, set updated beta values
    beta = alt_newton_step(beta, features, lam=lam)
    iter_count += 1
    #check for convergence 
    coefs_converged = check_coefs_convergence(beta_old, beta, tol, iter_count)
    print('Iterations : {}'.format(iter_count))
    print('Beta : {}'.format(beta))

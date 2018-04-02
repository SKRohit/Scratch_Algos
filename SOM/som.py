# -*- coding: utf-8 -*-
"""
Minimalistic Implementation of Self Organising Maps
"""
import numpy as np
import math
from PIL import Image


class SOM:
    def __init__(self, x_size, y_size, dim, n_iter = 1000, learning_rate = 0.1):
        ''' Initializes a Self Organizing Maps.
            x_size,y_size - dimensions of the SOM
            dim - number of the elements of the vectors in input
            n_iter - number of iterations
            learning_rate - initial learning rate
            (at the iteration t we have learning_rate(t) = learning_rate / (1 + t/T) where T is num_iteration/log(initial_radius))
        '''
        self.weights = np.random.rand(x_size, y_size, dim)*2-1
        self.sigma0 = max(self.weights.shape)/2
        self.n_iter = n_iter
        self.t_const = self.n_iter/math.log(self.sigma0)
        self.lr = learning_rate
    
    #selects the winner neuron
    def winner(self, vector):
        dist = np.sum((self.weights-vector)**2, axis = 2)
        min_idx = dist.argmin()
        return np.unravel_index(min_idx, dist.shape)
    
    #updates the neighbouring radius
    def neib_radius(self, iter_count):
        return self.sigma0*np.exp(-iter_count/self.t_const)
    
    #calcualtes a distance matrix wrt the winning neuron
    def winner_dist(self, vector):
        win = self.weights[self.winner(vector)]
        dist_map = np.zeros((self.weights.shape[0],self.weights.shape[1]))
        it = np.nditer(dist_map, flags = ['multi_index'])
        while not it.finished:
            dist_map[it.multi_index] = np.sum((win - self.weights[it.multi_index])**2)
            it.iternext()
        return dist_map
    
    
    #implements update algorithm for som
    def influence(self, vector, iter_count):
        dist_map = self.winner_dist(vector)
        nbhod_radius = self.neib_radius(iter_count)
        learn_rate = self.lr*np.exp(-iter_count/self.t_const)
        theta = learn_rate * np.exp((-dist_map**2)/(2*nbhod_radius))
        return np.expand_dims(theta,2) * (vector - self.weights)
    
    #normalises the input data    
    def normalise(self, X):
        norm_x = [X[:,i]-np.mean(X[:,i])/np.std(X[:,i]) for i in range(X.shape[1])]
        #norm_x = X-np.mean(X, axis = 0)/np.std(X, axis = 0) 
        return np.array(norm_x).T
     
    #trains the som by randomly selecting data        
    def teach_som(self, X):
        X = self.normalise(X)
        for i in range(self.n_iter):
            idx = np.random.randint(len(X))
            if i % 10 == 0:
                print ('Training Iteration {}'.format(i))
            self.weights = self.weights + self.influence(X[idx], i)
    
    #initialises number of iterations
    def init_n_iter(self):
        self.n_iter = 500 * self.weights.shape[0] * self.weights.shape[1]
    
    #shows trained som
    def show(self):
        im = Image.fromarray(self.weights.astype('uint8'))
        im.format = 'JPG'
        im.show()

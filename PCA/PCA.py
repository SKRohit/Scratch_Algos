# -*- coding: utf-8 -*-
"""
Minimalistic Implementation of Principal Component Analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PCA:
    def __init__(self, features, method = None):
        ''' Initializes Principal Component Analysis
            features - dataset without target variable
            method - legal values 'std_cov' and 'std_cor', choses whether to consider 
                     correlation or covariance matrix for eigen decomposition
         '''
        self.features = features
        self.method = method
    
    #standardises the data set 
    def normalise(self):
        return (self.features - np.mean(self.features, axis = 0))/np.std(self.features, axis = 0)

    #returns covariance or correlation matrix of the given data
    def corr_cov_mat(self):
        if (self.method == 'std_cov'):
            std_fet = self.normalise(self.features)
            return np.cov(std_fet, rowvar = False)
        else:
            return np.corrcoef(self.features.T)
    
    #performs eigen decomposition  
    def eig_decomp(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.cov_corr_mat())
        return eigenvalues, eigenvectors
    
    #sorts eigen values and eigen vectors to identify different principal components
    def sort_eigen(self):
        eig_vals, eig_vecs = self.eig_decomp()
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
        eig_pairs.sort(key = lambda x: x[0], reverse = True)
        return eig_pairs
    
    #returns new features (principal components)
    def make_new_fet(self):
        eig_pairs = self.sort_eigen()
        eigval_sort = np.array([eig_pairs[index][0] for index in range(len(eig_pairs))])
        eigvec_sort = np.array([eig_pairs[index][1] for index in range(len(eig_pairs))])
        cum_var = np.cumsum(eigval_sort)/sum(eigval_sort)
        print ('Cumulative proportion of variance explained {}'.format(cum_var))
        num_comp = int(input ('Enter the number of principal components you want to choose'))
        new_fet = np.dot(self.features, eigvec_sort[:,:num_comp]) 
        return new_fet
    
    #implements a 1D or 2D plot of obtained principal components
    def plot2D(self):
        new_fet = self.make_new_fet() 
        if new_fet.shape[1] == 1:
            plt.scatter(new_fet, len(new_fet)*[0])
            plt.title('PCA Dimensionality Reduction to 1D')
            plt.xlabel('Principal Component 1')
            plt.show()
        plt.scatter(new_fet[:,0], new_fet[:,1])
        plt.title('PCA Dimensionality Reduction to 2D')
        plt.ylabel('Principal Component 2')
        plt.xlabel('Principal Component 1')
        plt.show()
    
    #implements a 1D, 2D or 3D plot of obtained principal components    
    def plot3D(self):
        new_fet = self.make_new_fet() 
        if new_fet.shape[1] == 1:
            plt.scatter(new_fet, len(new_fet)*[0])
            plt.title('PCA Dimensionality Reduction to 1D')
            plt.xlabel('Principal Component 1')
            plt.show()
        elif new_fet.shape[1] == 2:
            plt.scatter(new_fet[:,0], new_fet[:,1])
            plt.title('PCA Dimensionality Reduction to 2D')
            plt.ylabel('Principal Component 2')
            plt.xlabel('Principal Component 1')
            plt.show()
        else:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(new_fet[:,0], new_fet[:,1], new_fet[:,2])
            ax.set_title('PCA Reduces Data to 3D')
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.set_zlabel('Principal Component 3')
            plt.show()

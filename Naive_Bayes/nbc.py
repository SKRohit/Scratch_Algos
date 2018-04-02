# -*- coding: utf-8 -*-
"""
Minimalistic Implementation of Naive Bayes Spam Classifier
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class nbclassifier:
    def __init__(self, data, method = 'tfidf', n = 1, split = 0.75):
        ''' Initializes the nbclassifier.
            data - text data to perform classification on. First column should 
                   contain labels and Second column should contain text data 
            method - legal values 'tfidf' for term frequency inverse document frequency,
                    and 'ngram' for bag of words 
            n = number of words in an n-gram
            split = split size of training data
        '''
        self.data = data
        self.train_data, self.test_data = train_test_split(data, train_size = split)
        self.train_data = [[item[0], self.ngrams(n, item[1])] for item in self.train_data]
        self.test_data = [[item[0], self.ngrams(n, item[1])] for item in self.test_data]
        self.ngram_lists = sorted(set([gram for doc in self.train_data for gram in doc[1]]))
        self.ngram_map = dict(zip(self.ngram_lists, range(len(self.ngram_lists))))
        self.freq_mat = np.zeros((len(self.train_data), len(self.ngram_map)+1))
        self.tfidf = np.zeros((len(self.train_data), len(self.ngram_lists)+1))
        self.freq_mat[:,len(self.ngram_map)] = list(zip(*self.train_data))[0]
        self.tfidf[:,len(self.ngram_map)] = list(zip(*self.train_data))[0]
        self.sub_set1 = None
        self.sub_set0 = None
        self.sub_tfidf1 = None
        self.sub_tfidf0 = None
        self.method = method
        self.alpha = 1
        self.nof = len(self.ngram_map)
        self.docs = len(self.train_data)
        self.spamCount = 0
        self.pA = 0
        self.pNotA = 0
    
    #converts the docs into a bag of n-grams
    def ngrams(self, n, doc):
        doc = doc.split(' ')
        grams = []
        for i in range(len(doc)-n+1):
            gram = ' '.join(doc[i:i+n])
            grams.append(gram.lower())
        return grams
    
    #trains the classifier on the data
    def train(self):
        tfidf = []
        nof = self.nof
        docs = self.docs
        for i, doc in enumerate(self.train_data):
            label = doc[0]
            grams = doc[1]
            if label == 1:
                self.spamCount += 1
            for gram in grams:
                j = self.ngram_map[gram]
                if label == 1:
                    self.freq_mat[i,j] += 1 
                else:
                    self.freq_mat[i,j] += 1
                
        doc_freq = np.array([np.count_nonzero(self.freq_mat[:,i]) for i in range(self.freq_mat.shape[1]-1)])
        tfidf.append([[tfidf.append((self.freq_mat[i,j])*(np.log((docs +1)/(doc_freq[j]+1))+1)) for j in range(nof)] for i in range(docs)])
        self.sub_set1 = self.freq_mat[self.freq_mat[:,nof] ==1]
        self.sub_set0 = self.freq_mat[self.freq_mat[:,nof] ==0]
        self.tfidf[:,:nof] = np.array(tfidf)
        self.sub_tfidf1 = self.tfidf[self.tfidf[:,nof] ==1]
        self.sub_tfidf0 = self.tfidf[self.tfidf[:,nof] ==0]
        self.pA = self.spamCount/docs
        self.pNotA = 1- self.pA
    
    #performs classification
    def classify(self, text, alpha=1.0):        
        self.alpha = alpha
        isSpam = self.pA * self.conditionalText(text, 1)
        notSpam = self.pNotA * self.conditionalText(text, 0)
        if (isSpam > notSpam):
            return 1
        else:
            return 0
    
    #return conditional probability of a doc given spam or ham
    def conditionalText(self, grams, label):
        result = 1.0
        if self.method == "ngram":
            for ngram in grams:
                result *= self.conditionalNgram(ngram, label)
        else:
            for ngram in grams:
                result *= self.conditionalTfIdf(ngram, label)
        return result
    
    #calculates conditional probability of an n-gram
    def conditionalNgram(self, ngram, label):
        alpha = self.alpha
        nof = self.nof
        try:
            if label ==1:
                return (sum(self.sub_set1[:,self.ngram_map[ngram]]) + alpha)/(np.sum(self.subset1[:,:nof]) + alpha*nof)
            else:
                return (sum(self.sub_set0[:,self.ngram_map[ngram]]) + alpha)/(np.sum(self.subset0[:,:nof]) + alpha*nof)
        except KeyError:
            if label ==1:
                return alpha/(np.sum(self.subset1[:,:nof]) + alpha*nof)
            else:
                return alpha/(np.sum(self.subset0[:,:nof]) + alpha*nof)

    #calculates conditional term frequency inverse document frequency for an n-gram
    def conditionalTfIdf(self, ngram, label):
        alpha = self.alpha
        nof = self.nof
        try:
            if label ==1:
                return (sum(self.sub_tfidf1[:,self.ngram_map[ngram]]) + alpha)/(np.sum(self.sub_tfidf1[:,:nof]) + alpha*nof)
            else:
                return (sum(self.sub_tfidf0[:,self.ngram_map[ngram]]) + alpha)/(np.sum(self.sub_tfidf0[:,:nof]) + alpha*nof)
        except KeyError:
            if label ==1:
                return alpha/(np.sum(self.sub_tfidf1[:,:nof]) + alpha*nof)
            else:
                return alpha/(np.sum(self.sub_tfidf0[:,:nof]) + alpha*nof)
    
    #evaluates a sample doc to be spam or ham      
    def evaluate_test_data(self):
        results = []
        for test in self.test_data:
            label = test[0]
            text = test[1]
            ruling = self.classify(text)
            if ruling == label:
                results.append(1) 
            else:
                results.append(0)
        print("Evaluated {} test cases. {:.2f}% Accuracy".format(len(results), 100.0*sum(results)/len(results)))
        return sum(results)/float(len(results))

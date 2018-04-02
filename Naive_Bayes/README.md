## Minimalistic Implementation of Naive Bayes Spam Classifier.

Implements a class nbclassifier with necessary functions to implement Naive bayes classification.

Required arguments to initialise an nbclassifier object are:

data - text data to perform classification on. First column should 
       contain labels and Second column should contain text data 
       
method - legal values 'tfidf' for term frequency inverse document frequency,
         and 'ngram' for bag of words 
         
n = number of words in an n-gram

split = split size of training data

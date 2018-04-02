## Minimalistic Implementation of Self Organising Maps.

Implements a class SOM with necessary functions to implement Self Organising Maps.

Required arguments to initialise an SOM object are:

x_size,y_size - dimensions of the SOM

dim - number of the elements of the vectors in input

n_iter - number of iterations

learning_rate - initial learning rate
(at the iteration t we have learning_rate(t) = learning_rate / (1 + t/T) where T is num_iteration/log(initial_radius))


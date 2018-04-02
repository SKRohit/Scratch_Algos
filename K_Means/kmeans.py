import numpy as np 
import matplotlib.pyplot as pyplot

''' Minimalistic Implementation of K-Means Clustering. '''

#loads dataset
def load_dataset(name):
	return np.loadtxt(name)

#calculates euclidean distance
def euclidian(a,b):
	return np.linalg.norm(a-b)

#calculates manhattan distance
def manhattan(x,y):
     return sum(abs(a-b) for a,b in zip(x,y))

def nth_root(value, n_root): 
    root_value = 1/float(n_root)
    return round (Decimal(value) ** Decimal(root_value),3)

#calculates manhattan distance 
def minkowski(x,y,p_value): 
    return nth_root(sum(pow(abs(a-b),p_value) for a,b in zip(x, y)),p_value)
 

#performs k-means
def kmeans(k, epsilon = 0, distance = 'euclidian'):
	history_centroids = []
	if distance == 'euclidian':
		dist_method = euclidian
	elif distance == 'manhattan':
		dist_method = manhattan
	else:
		dist_method = minkowski
	dataset = load_dataset('filename.txt')
	num_instances, num_features = dataset.shape
	prototypes = dataset[np.random.randint(0, num_instances-1, size = k)]
	history_centroids.append(prototypes)
	prototypes_old = np.zeros(prototypes.shape)
	belongs_to = np.zeros((num_instances, 1))
	norm = dist_method(prototypes_old, prototypes)
	iteration = 0
	while norm > epsilon:
		iteration += 1
		norm = dist_method(prototypes_old, prototypes)
		for instance_index, instance in enumerate(dataset):
			dist_vec = np.zeros((k,1))
			for prototype_index, prototype in enumerate(prototypes):
				dist_vec[prototype_index,0] = dist_method(prototype, instance)
			belongs_to[instance_index,0] = np.argmin(dist_vec)
		tmp_prototypes = np.zeros(prototypes.shape)
		for index in range(len(prototypes)):
			indices = [i in for i in len(dataset) if belongs_to[i] == index]
			tmp_prototypes[index, :] = np.mean(dataset[indices], axis = 0)
		prototypes_old = prototypes
		prototypes = tmp_prototypes
		history_centroids.append(tmp_prototypes)
	return prototypes, history_centroids, belongs_to

#plots 2 clusters 
def plot(dataset, history_centroids, belongs_to):
	colors = ['r', 'g']
	fig, ax = plt.subplots()
	for index in range(dataset.shape[0]):
        instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
        for instance_index in instances_close:
            ax.plot(dataset[instance_index][0], dataset[instance_index][1], (colors[index] + 'o'))

    history_points = []
    for index, centroids in enumerate(history_centroids):
        for inner, item in enumerate(centroids):
            if index == 0:
                history_points.append(ax.plot(item[0], item[1], 'bo')[0])
            else:
                history_points[inner].set_data(item[0], item[1])
                print("centroids {} {}".format(index, item))
                plt.show()

def execute():
    #load dataset
    dataset = load_dataset('filename.txt')
    centroids, history_centroids, belongs_to = kmeans(2)
    #plot the results
    plot(dataset, history_centroids, belongs_to)

execute()

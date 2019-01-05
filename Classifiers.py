import matplotlib.animation as animation
import matplotlib.pyplot as plt
from operator import itemgetter
import numpy as np
import random
import math
import sys

from pprint import pprint

class SOM:

	def __init__(self, input_matrix, num_nodes = 0, num_iterations = 2000, learning_rate = .45):

		# if unspecified number of nodes, use heuristical method from SOM Toolbix
		if num_nodes == 0:
			num_nodes = 5 * int(len(input_matrix) ** 0.54321)

		# annoying cornercase when dealing with one dimensional data
		if input_matrix.shape[-1] == len(input_matrix):
			self.lattice = np.random.rand(num_nodes, num_nodes, 1)
		else:
			self.lattice = np.random.rand(num_nodes, num_nodes, input_matrix.shape[-1])

		self.delta = np.zeros_like(self.lattice)
		print(self.lattice)
		print('[SOM] Lattice created with %d nodes' % num_nodes)

		self.num_iterations = num_iterations
		self.input_matrix = input_matrix

		self.orig_rate = learning_rate
		self.time_lambda = self.num_iterations / math.log(num_nodes)

		self.iteration_info = []


	# converts vector to scalar to display on heatmap
	def vector_to_scalar(self, value):
		return int(math.sqrt(np.dot(value, value))) + round(math.sqrt(np.dot(value, value)) - int(math.sqrt(np.dot(value, value))), 2)

    # default function to display data at each iteration, can be overidden by user. This one displays the weight matrix as a heatmap
	def show(self, stick):
		plt.figure(1)
		self.display()

		plt.figure(2)
		self.distance_map()

		if stick:
			plt.show()
		else:
			plt.draw()
			plt.pause(0.001)

	def display(self):
		plt.clf()
		
		tmp = np.zeros((self.lattice.shape[0], self.lattice.shape[1]))
		for i in range(self.lattice.shape[0]):
			for j in range(self.lattice.shape[1]):
				tmp[i][j] = self.vector_to_scalar(self.lattice[i][j])
				
		plt.matshow(tmp, fignum = 1, cmap = plt.cm.OrRd)
		for i in range(self.lattice.shape[0]):
			for j in range(self.lattice.shape[1]):
				pass#plt.text(j, i, str(tmp[i][j]), horizontalalignment = 'center', verticalalignment = 'center')


		plt.gcf().gca().add_artist(plt.Circle((self.iteration_info[2][1], self.iteration_info[2][0]), self.iteration_info[1], color='xkcd:azure', alpha = 0.2))
		plt.suptitle('Iteration %d' % self.iteration_info[0])
		plt.title('Radius = %f  BMU = (%d, %d) which is %f ~ %f' % (self.iteration_info[1], self.iteration_info[2][0], self.iteration_info[2][1], self.vector_to_scalar(self.lattice[self.iteration_info[2][0]][self.iteration_info[2][1]]), self.vector_to_scalar(self.iteration_info[3])))
		#plt.draw()
		#plt.pause(0.001)


	# trains self organizing map on previously specified amount of iterations
	def train(self, animate = False):
		orig_radius = (self.lattice.shape[0]) / 2 # change to double if needed
		for iteration in range(self.num_iterations):

			# get random vector from input
			input_vector = self.input_matrix[random.randint(0, len(self.input_matrix)-1)]
			
			# find BMU of given vector
			BMU = self.find_BMU(input_vector)
			#print('[SOM]\tBMU of', input_vector,'is located at', BMU[0],'which =', self.lattice[BMU[0][0],BMU[0][1]], 'with a distance of', BMU[1])

			# decaying functions
			radius = orig_radius * math.exp(float(-iteration) / self.time_lambda)
			learning_rate = self.orig_rate * math.exp(float(-iteration) / self.num_iterations)

			self.iteration_info = [iteration, radius, BMU[0], input_vector]

			if animate:
				self.show(False)
			# alter weight of all items in neighborhood
			for i in range(self.lattice.shape[0]):
				for j in range(self.lattice.shape[1]):
					dist = math.sqrt((i - BMU[0][0]) ** 2 + (j - BMU[0][1]) ** 2)
					if dist < radius:
						theta = math.exp(-(dist * dist) / (2 * radius ** 2))
						self.delta[i][j] = self.lattice[i][j]
						self.lattice[i][j] += theta * learning_rate * (np.squeeze(np.asarray(input_vector)) - self.lattice[i][j])

		
	# attempts to classify given input vector. Currently incomplete, currently just finds best matching unit
	def classify(self, input_vector):
		return self.find_BMU(input_vector)

	# find best matching unit of given input vector in lattice
	def find_BMU(self, input_vector):
		BMU = [[-1, -1], sys.maxsize]
		for i in range(self.lattice.shape[0]):
				for j in range(self.lattice.shape[1]):
					dist = math.sqrt(np.dot((input_vector - self.lattice[i][j]),(input_vector - self.lattice[i][j])))
					if dist < BMU[1]:
						BMU[0] = [i, j]
						BMU[1] = dist

		return BMU

	# returns weight matrix (lattice)
	def get_lattice(self):
		return self.lattice

	def distance_map(self):
		plt.clf()
		mean_vector = np.array([0, 0, 0, 0], dtype = 'float64')
		for xvector in self.lattice:
			for yvector in xvector:
				mean_vector += yvector
		mean_vector /= (self.lattice.shape[0] * self.lattice.shape[1])
		relative_mat = np.zeros((self.lattice.shape[0], self.lattice.shape[1]), dtype = 'float64')

		for x in range(len(self.lattice)):
			for y in range(len(self.lattice[x])):
				relative_mat[x][y] = math.sqrt(np.dot(mean_vector - self.lattice[x][y], mean_vector - self.lattice[x][y]))
				
		plt.matshow(relative_mat, fignum = 2, cmap = plt.cm.OrRd)
		'''
		for i in range(self.lattice.shape[0]):
			for j in range(self.lattice.shape[1]):
				plt.text(j, i, str(tmp[i][j]), horizontalalignment = 'center', verticalalignment = 'center')
		'''		


		plt.gcf().gca().add_artist(plt.Circle((self.iteration_info[2][1], self.iteration_info[2][0]), self.iteration_info[1], color='xkcd:azure', alpha = 0.2))
		plt.suptitle('Distance Map')
		plt.title('Radius = %f  BMU = (%d, %d) which is %f ~ %f' % (self.iteration_info[1], self.iteration_info[2][0], self.iteration_info[2][1], self.vector_to_scalar(relative_mat[self.iteration_info[2][0]][self.iteration_info[2][1]]), self.vector_to_scalar(self.iteration_info[3])))
class KMeans:

	def __init__(self, k = 2):
		self.k = k 
		self.centroid = np.zeros(2)

	def train(self, data):

		self.centroid = np.array([np.random.randint(min(data), high = max(data)) for x in range(self.k)])

		prev_centroid = np.zeros_like(self.centroid)
		while not np.array_equal(prev_centroid, self.centroid):
			self.display(data)
			weights = np.zeros([len(self.centroid), 2])
			for feature in data:
				min_index = np.argmin(np.sqrt((self.centroid - feature) * (self.centroid - feature)))
				weights[min_index][0] += feature
				weights[min_index][1] += 1

			prev_centroid[:] = self.centroid
			for x in range(len(self.centroid)):
				self.centroid[x] = weights[x][0] / weights[x][1]

	def classify(self, input_vector):
		return np.argmin(np.sqrt((self.centroid - input_vector) * (self.centroid - input_vector)))


	def display(self, data):
		plt.clf()
		plt.plot(data, np.zeros_like(data), 'x', color = 'blue', label = 'data')
		plt.plot(self.centroid, np.zeros_like(self.centroid), 'x', color = 'red', label = 'centroids')
		plt.legend(loc = 'upper left')
		plt.draw()
		plt.pause(0.001)

	def show(self):
		plt.show()

class AgglomerativeCluster:

	def __init__(self, num_clusters):
		self.num_clusters = num_clusters
		self.clusters = []

	def train(self, data):
		self.clusters = [self.Cluster(self.mag(x)) for x in data]

		ind = 0
		while len(self.clusters) > self.num_clusters:
			# self.display()  <-- extremely slow, don't use while training
			mins = [0, sys.maxsize]
			if len(self.clusters) < 10:
				centroids = []
				for cluster in self.clusters:
					if cluster.centroid != self.clusters[ind].centroid:
						centroids.append(cluster.centroid)

				if abs(np.std(centroids) - np.std(centroids + [self.clusters[ind].centroid])) > 50:
					ind = self.update_index(ind)
					continue

			for index, cluster in enumerate(self.clusters):
				if index == ind:
					continue
				if abs(cluster.centroid - self.clusters[ind].centroid) < mins[1]:
					mins = [index, abs(cluster.centroid - self.clusters[ind].centroid)]

			self.clusters[ind].update(self.clusters[mins[0]])
			self.clusters.remove(self.clusters[mins[0]])

			ind = self.update_index(ind)

		self.display()

	def update_index(self, index):
		if index > len(self.clusters) - 2:
			return 0
		else:
			return index + 1

	def mag(self, arr):
		return np.sqrt(arr.dot(arr))

	def classify(self, input_vector):
		distances = []
		for cluster in self.clusters:
			mind = sys.maxsize
			for value in cluster.value:
				if abs(self.mag(input_vector) - value) < mind:
					mind = abs(self.mag(input_vector) - value)
			distances.append(mind)

		return np.argmin(distances)


	def display(self):
		colors = ['k', 'r', 'y', 'm', 'b', 'g']
		plt.clf()
		plt.title('%d Clusters Left' % len(self.clusters))
		for x in range(len(self.clusters)):
			plt.plot(self.clusters[x].value, np.zeros_like(self.clusters[x].value), 'x', colors[x % len(colors)])
		plt.draw()
		plt.pause(0.001)

	def show(self):
		plt.show()

	class Cluster:
		def __init__(self, value):
			self.value = [value]
			self.centroid = value

		def update(self, cluster):
			for x in cluster.value:
				self.value.append(x)
			self.centroid = np.mean(self.value)

class kNearestNeighbor:

	def __init__(self, k):
		self.k = k

	def classify(self, input_vector, data):
		distances = []
		for vector in data:
			distances.append([vector[0], abs(input_vector - vector[1])])
		k_distances = sorted(distances, key = itemgetter(1))[:self.k]
		p_dict = {}
		for _class, dist in k_distances:
			if _class in p_dict:
				p_dict[_class] += 1
			else:
				p_dict[_class] = 1

		return max(p_dict.items(), key = itemgetter(1))[0]
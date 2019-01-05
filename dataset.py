import numpy as np

def get_data():
	data = []
	labels = []

	with open("bad_details.txt", 'r') as file:
		for item in file.read().split('\n')[:-1]:
				arr = [float(i) for i in item.split(',')]
				data.append(arr[:4])
				labels.append(arr[4:])
	data, labels = np.array(data), np.array(labels)
	p = np.random.permutation(len(data))
	data, labels = data[p], labels[p]

	return data[:int(len(data) * 0.8)], labels[:int(len(labels) * 0.8)], data[int(len(data)*0.8):], labels[int(len(labels)*0.8):]

def remove_outliers(arr):
	arr = np.array(arr)

	dists = np.array([])
	neighbors = []
	for coord in arr:
		neighbors.append(np.sort(np.linalg.norm(arr - coord, axis = 1))[:8])
		dists = np.concatenate((dists, neighbors[-1]))
	dev = np.std(dists)
	mean = np.mean(dists)
	
	new = []
	for index, neighbor in enumerate(neighbors):
		coord = arr[index]
		if np.mean(neighbor) > mean + 2 * dev:
			pass
		else:
			new.append(coord)
	
	return np.array(new)

def dont_remove_outliers(arr):
	return np.array(arr)
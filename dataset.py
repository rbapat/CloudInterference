import numpy as np

def get_data():

	with open("good_data.txt", 'r') as file:
		good_data = [float(item) for item in file.read().split('\n')[:-1]]

	with open("bad_data.txt", 'r') as file:
		bad_data = [float(item) for item in file.read().split('\n')[:-1]]

	return good_data, bad_data, np.random.permutation(good_data + bad_data)

def get_kNN_data(good_data, bad_data):
	X = []
	for x in good_data:
		X.append([0, x])

	for x in bad_data:
		X.append([1, x])

	return X
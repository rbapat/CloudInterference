from dataset import get_data, get_kNN_data
import numpy as np
import Classifiers

def main():
	# TODO: implement better dataset parser
	# TODO: Comment all the code :)

	good_data, bad_data, shuffled_data = get_data()

	train = np.random.permutation(good_data[:int(len(good_data)/2)] + bad_data[:int(len(bad_data)/2)])
	test = np.random.permutation(good_data[int(len(good_data)/2):] + bad_data[int(len(bad_data)/2):])

	HAC = Classifiers.AgglomerativeCluster(2)
	HAC.train(train)

	for x in test:
		prediction = HAC.classify(x)
		print(x, prediction)
	HAC.show()


if __name__ == '__main__':
	main()


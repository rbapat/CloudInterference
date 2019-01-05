import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from colorama import Fore, init
from minisom import MiniSom
import numpy as np
import dataset
import pickle
import os

DIM = 25
SIGMA = 1
LEARNING_RATE = 0.05
ITERATIONS = 10000
TEST_ITERATIONS = 100

def visualize(som, interference, no_interference):
	markers = []
	plt.figure(figsize=(DIM, DIM))
	plt.pcolor(som.distance_map().T, cmap='bone_r')
	plt.axis([-1, DIM, -1, DIM])
	markers.append(plt.plot(interference[:, 0], interference[:, 1], 'rx', markersize=12, label = 'interference data')[0])
	markers.append(plt.plot(no_interference[:, 0], no_interference[:, 1], 'go', markersize=12, label = 'no_interference data')[0])
	plt.legend(handles = markers, bbox_to_anchor=(0.5, -0.05), fancybox = True, shadow = True, ncol = 6, loc = 'center')
	return markers

def get_som_labels(train_data, train_labels):
	som = MiniSom(DIM, DIM, 4, sigma = SIGMA, learning_rate = LEARNING_RATE)
	som.train_random(train_data, ITERATIONS)

	'''
	if not os.path.isfile('som.p'):
		print('som.p not found, training SOM')
		som.train_random(train_data, 10000)
		with open('som.p', 'wb') as outfile:
			pickle.dump(som, outfile)
	else:
		print('som.p found, loading SOM')
		with open('som.p', 'rb') as infile:
			som = pickle.load(infile)
	'''
	
	interference, no_interference = [], []
	for cnt, xx in enumerate(train_data):
		w = som.winner(xx)
		if train_labels[cnt] > 0.0:
			interference.append(w)
		else:
			no_interference.append(w)

	return som, dataset.remove_outliers(interference), dataset.remove_outliers(no_interference)

def classify(winner, interference, no_interference):
	present = np.mean(np.sort(np.linalg.norm(interference - winner, axis = 1))[:5])
	not_present = np.mean(np.sort(np.linalg.norm(no_interference - winner, axis = 1))[:5])

	if present > not_present:
		return 0
	else:
		return 1
	

def main():
	accumulator = 0.0
	with open('figs/info.csv', 'w') as f:
			f.write("iteration,num_correct,num_incorrect,percent_accuracy,percent_inaccuracy,cumulative_inaccuracy\n")
			
	for index in range(TEST_ITERATIONS):
		plt.clf()
		train_data, train_labels, test_data, test_labels = dataset.get_data()
		som, interference, no_interference = get_som_labels(train_data, train_labels)

		_markers = visualize(som, interference, no_interference)
		correct = 0
		markers = [_markers[0], _markers[1], mpatches.Patch(color='lawngreen', label='Correct no interference'), mpatches.Patch(color='orangered', label='Incorrect interference'), mpatches.Patch(color='lawngreen', label='Incorrect no interference'), mpatches.Patch(color='orangered', label='Correct interference')]
		for i, x in enumerate(test_data):
			winner = som.winner(x)
			label, guess = test_labels[i], classify(winner, interference, no_interference)
			if label == 0.0 and guess == 0:
				print(Fore.GREEN + '[%d] [%d/%d] Correctly guessed no interference' % (index, i + 1, len(test_data)))
				markers[2] = plt.plot(winner[0], winner[1], 's', markersize=12, color = 'lawngreen', label='Correct no interference')[0]
				correct += 1
			elif label > 0.0 and guess == 0:
				print(Fore.RED + '[%d] [%d/%d] Failed to detect interference %f' % (index, i + 1, len(test_data), label), winner)
				markers[3] = plt.plot(winner[0], winner[1], '<', markersize=12, color = 'orangered', label='Incorrect interference')[0]
				pass
			elif label == 0.0 and guess == 1:
				print(Fore.RED + '[%d] [%d/%d] Failed to detect no_interference' % (index, i + 1, len(test_data)), winner)
				markers[4] = plt.plot(winner[0], winner[1], '>', markersize=12, color = 'lawngreen', label='Incorrect no interference')[0]
				pass
			elif label > 0.0 and guess == 1:
				print(Fore.GREEN + '[%d] [%d/%d] Correctly guessed interference %f' % (index, i + 1, len(test_data), label))
				markers[5] = plt.plot(winner[0], winner[1], 's', markersize=12, color = 'orangered', label='Correct interference')[0]
				correct += 1
		plt.legend(handles = markers, bbox_to_anchor=(0.5, -0.05), fancybox = True, shadow = True, ncol = 6, loc = 'center')

		incorrect = len(test_data) - correct
		accuracy = float(correct) / float(len(test_data)) * 100.0
		inaccuracy = float(incorrect) / float(len(test_data)) * 100.0
		accumulator += inaccuracy
		with open('figs/info.csv', 'a') as f:
			f.write('%d,%d,%d,%.2f,%.2f,%.2f\n' % (index, correct, incorrect, accuracy, inaccuracy, accumulator / (float(index) + 1)))
		print(Fore.MAGENTA + '[%d] Classifier trained with a %.2f%% accuracy and %.2f%% inaccuracy. Average inaccuracy: %.2f' % (index, accuracy, inaccuracy, accumulator / (float(index) + 1)))
		plt.savefig('figs/%d_%f.png' % (index, accuracy))


if __name__ == '__main__':
	init(convert = True)
	main()
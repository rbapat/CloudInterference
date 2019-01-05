from pprint import pprint
import numpy as np
from minisom import MiniSom
from numpy import random as rand
import pickle
from colorama import Fore, init
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import itertools
from scipy.spatial import ConvexHull

'''
I'm sorry if you are reading this
'''

#taken from https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
def point_in_hull(point, hull):
	n = len(hull)
	inside = False

	p1x,p1y = hull[0]
	for i in range(n+1):
		p2x,p2y = hull[i % n]
		if point[1] > min(p1y,p2y):
			if point[1] <= max(p1y,p2y):
				if point[0] <= max(p1x,p2x):
					if p1y != p2y:
						xinters = (point[1]-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
					if p1x == p2x or point[0] <= xinters:
						inside = not inside
		p1x,p1y = p2x,p2y
	return inside

def convex_hull(arr):
	#plt.draw()
	#plt.pause(.001)
	arr = np.unique(arr[np.lexsort((arr[:,1],arr[:,0]))], axis = 0)
	hull = [arr[0]]

	cur_ind = 0
	pi = 0
	while True:
		theta = 0
		_i = 0
		#plt.clf()
		#if(len(hull) > 1):
			#for i in range(len(hull)-1):
				#plt.plot([hull[i][0], hull[i+1][0]], [hull[i][1], hull[i+1][1]], 'r')
		#plt.plot(arr[:, 0], arr[:, 1], 'o')
		#plt.plot(arr[cur_ind][0], arr[cur_ind][1], 'go')
		for i, pt in enumerate(arr):
			if i == cur_ind or i == pi:
				continue
			delta = arr[cur_ind] - pt
			#plt.plot([arr[cur_ind][0], pt[0]], [arr[cur_ind][1], pt[1]], 'k-')
			if delta[0] == 0.0:
				_theta = np.pi / 2
			elif delta[1] == 0.0:
				_theta = 0.0
			else:
				_theta = (np.arctan(float(delta[1]) / float(delta[0])))
			if(len(hull) > 1):
				a = np.linalg.norm(pt - arr[cur_ind])
				b = np.linalg.norm(arr[cur_ind] - hull[-2])
				c = np.linalg.norm(pt - hull[-2])
				_theta = np.arccos((a*a + b*b - c*c) / (2 * a * b))
			#plt.annotate("%.3f" % _theta, xy = (pt[0], pt[1]), color='#ee8d18')
			#plt.draw()
			if _theta > theta:
				theta = _theta
				_i = i
		#plt.plot(arr[_i][0], arr[_i][1], 'ro')
		#plt.draw()
		hull.append(arr[_i])
		pi = cur_ind
		cur_ind = _i
		#plt.pause(0.05)
		if cur_ind == 0:
			break
	#plt.clf()
	#plt.plot(arr[:, 0], arr[:, 1], 'o')
	#for i in range(len(hull)-1):
	#	plt.plot([hull[i][0], hull[i+1][0]], [hull[i][1], hull[i+1][1]], 'r')
	#plt.pause(.2)
	#plt.clf()
	return hull

def main():
	dim = 25
	train_data, train_labels, test_data, test_labels = get_data()
	som = MiniSom(dim, dim, 4, sigma = 1, learning_rate = 0.05)

	if not os.path.isfile('som.p'):
		print('som.p not found, training SOM')
		som.train_random(train_data, 10000)
		with open('som.p', 'wb') as outfile:
			pickle.dump(som, outfile)
	else:
		print('som.p found, loading SOM')
		with open('som.p', 'rb') as infile:
			som = pickle.load(infile)
	
	interference, no_interference = [], []
	for cnt, xx in enumerate(train_data):
		w = som.winner(xx)
		if train_labels[cnt] > 0.0:
			interference.append(w)
		else:
			no_interference.append(w)

	interference, no_interference = remove_outliers(interference), remove_outliers(no_interference)
	interference_hull, no_interference_hull = convex_hull(interference), convex_hull(no_interference)

	'''
	plt.figure(figsize=(dim, dim))
	plt.title('interference')
	plt.pcolor(som.distance_map().T, cmap='bone_r')
	for point in interference:
		plt.plot(point[0] + 0.0, point[1] + 0.0, 'o', markerfacecolor = 'None', markeredgecolor = 'r', markersize=12, markeredgewidth=2)
	for i in range(len(interference_hull) - 1):
		plt.plot([interference[i][0] + 0.0, interference[i+1][0] + 0.0], [interference[i][1] + 0.0, interference[i+1][1] + 0.0], 'r')
	plt.axis([0, dim, 0, dim])

	plt.figure(figsize=(dim, dim))
	plt.title('no interference')
	plt.pcolor(som.distance_map().T, cmap='bone_r')
	for point in no_interference:
		plt.plot(point[0] + 0.0, point[1] + 0.0, 's', markerfacecolor = 'None', markeredgecolor = 'g', markersize=12, markeredgewidth=2)
	for i in range(len(no_interference_hull) - 1):
		plt.plot([no_interference_hull[i][0] + 0.0, no_interference_hull[i+1][0] + 0.0], [no_interference_hull[i][1] + 0.0, no_interference_hull[i+1][1] + 0.0], 'g')
	plt.axis([0, dim, 0, dim])
	'''
	plt.figure(figsize=(dim, dim))
	plt.pcolor(som.distance_map().T, cmap='bone_r')
	handles = [mpatches.Patch(color='r', label='training inference data'), mpatches.Patch(color='g', label='Ttraining no interference data'), mpatches.Patch(color='r', label='interference hull'), 
	mpatches.Patch(color='g', label='no interference hull'), mpatches.Patch(color='r', label='correct test interference data'), mpatches.Patch(color='#FFA500', label='failed test interference data'), 
	mpatches.Patch(color='g', label='correct test no interference data'), mpatches.Patch(color='#FFA500', label='failed test no interference data'), mpatches.Patch(color='#FFA500', label='unsure test interference data'),
	mpatches.Patch(color='#FFA500', label='unsure test no interference data'),  mpatches.Patch(color='#FFA500', label='failed test no interference data'), mpatches.Patch(color='#FFA500', label='failed test interference data'),]
	for point in interference:
		handles[0] = plt.plot(point[0] + 0.0, point[1] + 0.0, 'o', markerfacecolor = 'None', markeredgecolor = 'r', markersize=12, markeredgewidth=2, label = 'training interference data')[0]
	for point in no_interference:
		handles[1] = plt.plot(point[0] + 0.0, point[1] + 0.0, 's', markerfacecolor = 'None', markeredgecolor = 'g', markersize=12, markeredgewidth=2, label = 'training no interference data')[0]
	for i in range(len(interference_hull) - 1):
		handles[2] = plt.plot([interference_hull[i][0] + 0.0, interference_hull[i+1][0] + 0.0], [interference_hull[i][1] + 0.0, interference_hull[i+1][1] + 0.0], 'r', label = 'interference hull')[0]
	for i in range(len(no_interference_hull) - 1):
		handles[3] = plt.plot([no_interference_hull[i][0] + 0.0, no_interference_hull[i+1][0] + 0.0], [no_interference_hull[i][1] + 0.0, no_interference_hull[i+1][1] + 0.0], 'g', label = 'no interference hull')[0]
	plt.axis([-1, dim, -1, dim])
	

	acc = 0
	for i, x in enumerate(test_data):
		w = som.winner(x)
		yes = point_in_hull(w, interference_hull)
		no = point_in_hull(w, no_interference_hull)
		label = test_labels[i][0]

		if yes and no:
			print(Fore.MAGENTA + '[%d/%d] refine' % (i + 1, len(test_data)), label, w)
			if label > 0.0:
				handles[8] = plt.plot(w[0] + 0.0, w[1] + 0.0, '+', markerfacecolor = 'None', markeredgecolor = '#FFA500', markersize=12, markeredgewidth=2, label = 'unsure test interference data')[0]
			else:
				handles[9] = plt.plot(w[0] + 0.0, w[1] + 0.0, 'x', markerfacecolor = 'None', markeredgecolor = '#FFA500', markersize=12, markeredgewidth=2, label = 'unsure test no interference data')[0]
		elif yes:
			if label > 0.0:
				print(Fore.GREEN + '[%d/%d] interference present %f' % (i + 1, len(test_data), label))
				handles[4] = plt.plot(w[0] + 0.0, w[1] + 0.0, '<', markerfacecolor = 'None', markeredgecolor = 'r', markersize=12, markeredgewidth=2, label = 'correct test interference data')[0]
				acc += 1
			else:
				print(Fore.RED + '[%d/%d] interference present %f' % (i + 1, len(test_data), label))
				handles[5] = plt.plot(w[0] + 0.0, w[1] + 0.0, '<', markerfacecolor = 'None', markeredgecolor = '#FFA500', markersize=12, markeredgewidth=2, label = 'failed test interference data')[0]
		elif no:
			if label == 0.0:
				print(Fore.GREEN + '[%d/%d] no interference present %f' % (i + 1, len(test_data), label))
				handles[6] = plt.plot(w[0] + 0.0, w[1] + 0.0, '>', markerfacecolor = 'None', markeredgecolor = 'g', markersize=12, markeredgewidth=2, label = 'correct test no interference data')[0]
				acc += 1
			else:
				print(Fore.RED + '[%d/%d] no interference present %f' % (i + 1, len(test_data), label))
				handles[7] = plt.plot(w[0] + 0.0, w[1] + 0.0, '>', markerfacecolor = 'None', markeredgecolor = '#FFA500', markersize=12, markeredgewidth=2, label = 'failed test no interference data')[0]
		else:
			if label == 0.0:
				print(Fore.RED + '[%d/%d] couldnt find anything %f' % (i + 1, len(test_data), label))
				handles[10] = plt.plot(w[0] + 0.0, w[1] + 0.0, '>', markerfacecolor = 'None', markeredgecolor = '#FFA500', markersize=12, markeredgewidth=2, label = 'failed test no interference data')[0]
			else:
				print(Fore.RED + '[%d/%d] couldnt find anything %f' % (i + 1, len(test_data), label))
				handles[11] = plt.plot(w[0] + 0.0, w[1] + 0.0, '>', markerfacecolor = 'None', markeredgecolor = '#FFA500', markersize=12, markeredgewidth=2, label = 'failed test interference data')[0]
	accuracy = acc * 100 / len(test_labels)
	print('accuracy:', accuracy)

	plt.title('SOM Plotted with training and testing data')
	plt.legend(handles = handles, bbox_to_anchor=(0.5, -0.05), fancybox = True, shadow = True, ncol = 6, loc = 'center')
	plt.show()


if __name__ == '__main__':
	init(convert = True)
	main()
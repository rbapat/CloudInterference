# CloudInterference
Performance Interference detection using Machine Learning. Several algorithms are currently implemented and further developments are coming soon.

### Design

For now, the model utilizes [minisom](https://github.com/JustGlowing/minisom) as the Self-Organizing Map. I will improve my SOM implementation in [Classifiers.py](Classifiers.py) then near future and use that as the models's SOM. 4-Dimensional data extracted from servers (cpu_usage, network_receive, network_respond, response times) is run through the SOM and their BMU is plotted onto the SOM's distance map. 

In the [Convex Hull implementation](convex_hull_som.py), a convex hull is drawn around the clusters and used for classifcation of new data. This is very inaccurate due to the nature of the convex hull algorithm.
In the [Psuedo k-nearest-neighbor implementation](k-nearest-neighbor_som.py), the data is classified using an algorithm very simililar to a k-nearest-neighbor. This model yields roughly a 95% accuracy with the current data.

To test precision, data was run through 100 distinct SOMs. A plot of each SOM was stored in the [figs](/figs) directory, along with a description of each in [info.csv](figs/info.csv).

### Algorithms implemented in Classifiers.py

* Self-Organizing Map
* KMeans
* AgglomerativeCluster
* k-Nearest Neighbor

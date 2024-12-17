from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from sklearn.model_selection import ParameterGrid
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# FIND SUITABLE K
def suitable_k_kmean(X, max_k=10):
    avgs = list()
    min_k = 2
    for k in range(min_k,max_k):
        kmeans = KMeans(n_clusters=k).fit(X)
        s = metrics.silhouette_score(X, kmeans.labels_)
        print('silhouette Coefficients for k=', kmeans ,' is ',s)
        avgs.append(s)

    suitable_K = avgs.index(max(avgs)) + min_k
    print('Optimal K = {} silhouette Coefficients: {}'.format(suitable_K, '%.4g' % max(avgs)))
    return suitable_K

def suitable_k_agglo(X, max_k=10):
    avgs = list()
    min_k = 2
    for k in range(min_k,max_k):
        agglo= AgglomerativeClustering(n_clusters=k).fit(X)
        s = metrics.silhouette_score(X, agglo.labels_)
        print('silhouette Coefficients for k=', agglo ,' is ',s)
        avgs.append(s)

    suitable_K = avgs.index(max(avgs)) + min_k
    print('Optimal K = {} silhouette Coefficients: {}'.format(suitable_K, '%.4g' % max(avgs)))
    return suitable_K

def distances_DBCSAN(X, min_pts=5):
    # Set min_samples
    min_pts = min_pts  # Adjust based on domain knowledge
    neighbors = NearestNeighbors(n_neighbors=min_pts)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)

    # Sort distances for the k-distance graph
    distances = np.sort(distances[:, -1])
    plt.plot(distances)
    plt.xlabel('Data Points')
    plt.ylabel('Distance to {} n Neighbor'.format(min_pts))
    plt.title('K-Distance Graph')
    plt.grid(True)



def parameters_grid_DBSCAN(X):
    # Parameter grid for tuning
    param_grid = {
        'eps': np.arange(5, 25, 1), # 0.5 3.5 0.15
        'min_samples': range(2, 30)
    }

    best_params, best_score = None, -1

    # Loop through parameter grid
    for params in ParameterGrid(param_grid):
        dbscan = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])

        # Fit DBSCAN and assign cluster labels
        labels = dbscan.fit_predict(X)

        # Exclude noise points (label == -1)
        valid_labels = labels[labels != -1]

        # Check if there are at least two unique clusters (excluding noise)
        if len(set(valid_labels)) > 1:
            # Evaluate clustering on non-noise points
            score = metrics.silhouette_score(X[labels != -1], valid_labels)

            if score > best_score:
                best_score = score
                best_params = params
    print(f'best eps: {best_params['eps']}, min_pts: {best_params['min_samples']}')

    return best_params['eps'], best_params['min_samples']

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import DBSCAN
from utils.evaluate_clustering import evaluate_clustering
from utils.plot_clustering import (plot_cls_pca_2d, plot_cls_pca_3d, plot_cluster_label_percentage,
                             plot_pca_2d)
from utils.clustering_extended import distances_DBCSAN, parameters_grid_DBSCAN


# Part1: Load dataset
red_wine = "https://drive.google.com/uc?id=1jXQIFh6y3xo52byug_UcqBdtZgjOMn_D"
df = pd.read_csv(red_wine)

# show details dataset
print(f'Example dataset:\n{df.head()}')
print(f'Shape of dataset: {df.shape}')
print(f'quality count: {Counter(df['quality'])}\n') # Imbalanced data

# ---------------------------------------------------------------------------------------------
# Part 2: Initial parameters and exploration
# Step1: Sample exactly n rows for each category
n = 199
balanced_df = pd.concat([
    df[df['quality'] == 5].sample(n=n, random_state=42),
    df[df['quality'] == 6].sample(n=n, random_state=42),
    df[df['quality'] == 7].sample(n=n, random_state=42)
])

balanced_df = balanced_df.reset_index(drop=True)
print(f'quality count after balance: {Counter(balanced_df['quality'])}\n')

# Step2: assign parameters
X = np.array(balanced_df.drop(['quality'], axis=1).values)
y = balanced_df['quality']
title = 'Wine Data'

# plot
plot_pca_2d(balanced_df, title=title, n_components=2)
plt.show()

# ---------------------------------------------------------------------------------------------
# Part3: Apply with algorithms
# Step 1-1: Check distance for n min_pts
distances_DBCSAN(X, 4)  # distance ~ 0-20
plt.show()

# Step 1-2: get best parameters by range distance from above
eps, min_samples = parameters_grid_DBSCAN(X)
print(f'eps suitable: {eps}')
print(f'min_sample: {min_samples}')

# Step 1-3: Apply DBSCAN Clustering 0.5,4
dbscan = DBSCAN(eps=eps ,min_samples=min_samples)

# Step 1-3: Get the cluster labels
balanced_df['dbscan_cluster'] = dbscan.fit_predict(X)

# Step 1-4: Calculate proportion and percentage of each label in each cluster
# proportion
cls_label_dist = balanced_df.groupby(['dbscan_cluster', 'quality']).size().unstack(fill_value=0)
print(cls_label_dist)

# percentage
cls_label_percentage = cls_label_dist.div(cls_label_dist.sum(axis=1), axis=0) * 100
print(cls_label_percentage)

# ---------------------------------------------------------------------------------------------
# Part 4: Evaluation model
# DBSCAN can produce noise points labeled as -1
if len(balanced_df[balanced_df['dbscan_cluster'] != -1]) :
    dbscan_results = evaluate_clustering(X[balanced_df['dbscan_cluster'] != -1],
                                         balanced_df['dbscan_cluster'][balanced_df['dbscan_cluster'] != -1],
                                         y[balanced_df['dbscan_cluster'] != -1])
    print(f'\nDBSCAN:'
          f'\nSilhouette Score: {dbscan_results[0]:.4g}'
          f'\nAdjusted Rand Index: {dbscan_results[1]:.4g}'
          f'\nDavies-Bouldin Index: {dbscan_results[2]:.4g}')
else:
    print('DBSCAN found no clusters.')

# ---------------------------------------------------------------------------------------------
# Part5: plot
# Reduce to 2D using PCA
cluster = balanced_df['dbscan_cluster']
plot_cls_pca_2d(X, df_cluster=cluster, title=f'DBSCAN Clustering of {title} (2D PCA)', n_components=2)

# Reduce to 3D using PCA
plot_cls_pca_3d(X, df_cluster=cluster, title=f'DBSCAN Clustering of {title} (3D PCA)',n_components=3)

# Plot the percentage distribution of labels in each cluster as a stacked bar plot
plot_cluster_label_percentage(cls_label_percentage, 'Proportion of Each Label in DBSCAN Clusters')


plt.tight_layout()
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import DBSCAN
from evaluate_clustering import evaluate_clustering
from plot_clustering import (plot_cls_pca_2d, plot_cls_pca_3d, plot_cluster_label_percentage,
                             plot_pca_2d)
from clustering_extended import distances_DBCSAN, parameters_grid_DBSCAN


# Part1: Load dataset
red_wine = "https://drive.google.com/uc?id=1jXQIFh6y3xo52byug_UcqBdtZgjOMn_D"
df = pd.read_csv(red_wine)

# show details dataset
print(f'Example dataset:\n{df.head()}')
print(f'Shape of dataset: {df.shape}')
print(f'quality count: {Counter(df['quality'])}\n') # Imbalanced data

# ---------------------------------------------------------------------------------------------
# Part 2: Initial parameters and exploration
# Step1: Sample exactly n rows for each category
n = 199
balanced_df = pd.concat([
    df[df['quality'] == 5].sample(n=n, random_state=42),
    df[df['quality'] == 6].sample(n=n, random_state=42),
    df[df['quality'] == 7].sample(n=n, random_state=42)
])

balanced_df = balanced_df.reset_index(drop=True)
print(f'quality count after balance: {Counter(balanced_df['quality'])}\n')

# Step2: assign parameters
X = np.array(balanced_df.drop(['quality'], axis=1).values)
y = balanced_df['quality']
title = 'Wine Data'

# plot
plot_pca_2d(balanced_df, title=title, n_components=2)
plt.show()

# ---------------------------------------------------------------------------------------------
# Part3: Apply with algorithms
# Step 1-1: Check distance for n min_pts
distances_DBCSAN(X, 4)  # distance ~ 0-20
plt.show()

# Step 1-2: get best parameters by range distance from above
eps, min_samples = parameters_grid_DBSCAN(X)
print(f'eps suitable: {eps}')
print(f'min_sample: {min_samples}')

# Step 1-3: Apply DBSCAN Clustering 0.5,4
dbscan = DBSCAN(eps=eps ,min_samples=min_samples)

# Step 1-3: Get the cluster labels
balanced_df['dbscan_cluster'] = dbscan.fit_predict(X)

# Step 1-4: Calculate proportion and percentage of each label in each cluster
# proportion
cls_label_dist = balanced_df.groupby(['dbscan_cluster', 'quality']).size().unstack(fill_value=0)
print(cls_label_dist)

# percentage
cls_label_percentage = cls_label_dist.div(cls_label_dist.sum(axis=1), axis=0) * 100
print(cls_label_percentage)

# ---------------------------------------------------------------------------------------------
# Part 4: Evaluation model
# DBSCAN can produce noise points labeled as -1
if len(balanced_df[balanced_df['dbscan_cluster'] != -1]) :
    dbscan_results = evaluate_clustering(X[balanced_df['dbscan_cluster'] != -1],
                                         balanced_df['dbscan_cluster'][balanced_df['dbscan_cluster'] != -1],
                                         y[balanced_df['dbscan_cluster'] != -1])
    print(f'\nDBSCAN:'
          f'\nSilhouette Score: {dbscan_results[0]:.4g}'
          f'\nAdjusted Rand Index: {dbscan_results[1]:.4g}'
          f'\nDavies-Bouldin Index: {dbscan_results[2]:.4g}')
else:
    print('DBSCAN found no clusters.')

# ---------------------------------------------------------------------------------------------
# Part5: plot
# Reduce to 2D using PCA
cluster = balanced_df['dbscan_cluster']
plot_cls_pca_2d(X, df_cluster=cluster, title=f'DBSCAN Clustering of {title} (2D PCA)', n_components=2)

# Reduce to 3D using PCA
plot_cls_pca_3d(X, df_cluster=cluster, title=f'DBSCAN Clustering of {title} (3D PCA)',n_components=3)

# Plot the percentage distribution of labels in each cluster as a stacked bar plot
plot_cluster_label_percentage(cls_label_percentage, 'Proportion of Each Label in DBSCAN Clusters')


plt.tight_layout()
plt.show()

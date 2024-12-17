import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import AgglomerativeClustering
from evaluate_clustering import evaluate_clustering
from plot_clustering import (plot_cls_pca_2d, plot_cls_pca_3d, plot_cluster_label_percentage,
                             plot_pca_2d)
from clustering_extended import suitable_k_agglo


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
# Step 1-1: Find suitable k for K-means
suitable_k = suitable_k_agglo(X, max_k=10)  # check suitable k = 2
k = 3                                       # use k = 3 cause, known data set have 3 categories

# Step 1-2: Apply Agglomerativ
agglo = AgglomerativeClustering(n_clusters=k)

# Step 1-3: Get the cluster labels
balanced_df['agglo_cluster'] = agglo.fit_predict(X)

# Step 1-4: Calculate proportion and percentage of each label in each cluster
# proportion
cls_label_dist = balanced_df.groupby(['agglo_cluster', 'quality']).size().unstack(fill_value=0)
print(cls_label_dist)

# percentage
cls_label_percentage = cls_label_dist.div(cls_label_dist.sum(axis=1), axis=0) * 100
print(cls_label_percentage)

# ---------------------------------------------------------------------------------------------
# Part 4: Evaluation model
agglo_results = evaluate_clustering(X, balanced_df['agglo_cluster'], y)
print(f'\nAgglomerative Clustering: '
      f'\nSilhouette Score: {agglo_results[0]:.4g}'
      f'\nAdjusted Rand Index: {agglo_results[1]:.4g}'
      f'\nDavies-Bouldin Index: {agglo_results[2]:.4g}\n')

# ---------------------------------------------------------------------------------------------
# Part5: plot
# Reduce to 2D using PCA
cluster = balanced_df['agglo_cluster']
plot_cls_pca_2d(X, df_cluster=cluster, title=f'Agglomerative Clustering of {title} (2D PCA)', n_components=2)

# Reduce to 3D using PCA
plot_cls_pca_3d(X, df_cluster=cluster, title=f'Agglomerative Clustering of {title} (3D PCA)',n_components=3)

# Plot the percentage distribution of labels in each cluster as a stacked bar plot
plot_cluster_label_percentage(cls_label_percentage, 'Proportion of Each Label in Agglomerative Clusters')


plt.tight_layout()
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import AgglomerativeClustering
from evaluate_clustering import evaluate_clustering
from plot_clustering import (plot_cls_pca_2d, plot_cls_pca_3d, plot_cluster_label_percentage,
                             plot_pca_2d)
from clustering_extended import suitable_k_agglo


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
# Step 1-1: Find suitable k for K-means
suitable_k = suitable_k_agglo(X, max_k=10)  # check suitable k = 2
k = 3                                       # use k = 3 cause, known data set have 3 categories

# Step 1-2: Apply Agglomerativ
agglo = AgglomerativeClustering(n_clusters=k)

# Step 1-3: Get the cluster labels
balanced_df['agglo_cluster'] = agglo.fit_predict(X)

# Step 1-4: Calculate proportion and percentage of each label in each cluster
# proportion
cls_label_dist = balanced_df.groupby(['agglo_cluster', 'quality']).size().unstack(fill_value=0)
print(cls_label_dist)

# percentage
cls_label_percentage = cls_label_dist.div(cls_label_dist.sum(axis=1), axis=0) * 100
print(cls_label_percentage)

# ---------------------------------------------------------------------------------------------
# Part 4: Evaluation model
agglo_results = evaluate_clustering(X, balanced_df['agglo_cluster'], y)
print(f'\nAgglomerative Clustering: '
      f'\nSilhouette Score: {agglo_results[0]:.4g}'
      f'\nAdjusted Rand Index: {agglo_results[1]:.4g}'
      f'\nDavies-Bouldin Index: {agglo_results[2]:.4g}\n')

# ---------------------------------------------------------------------------------------------
# Part5: plot
# Reduce to 2D using PCA
cluster = balanced_df['agglo_cluster']
plot_cls_pca_2d(X, df_cluster=cluster, title=f'Agglomerative Clustering of {title} (2D PCA)', n_components=2)

# Reduce to 3D using PCA
plot_cls_pca_3d(X, df_cluster=cluster, title=f'Agglomerative Clustering of {title} (3D PCA)',n_components=3)

# Plot the percentage distribution of labels in each cluster as a stacked bar plot
plot_cluster_label_percentage(cls_label_percentage, 'Proportion of Each Label in Agglomerative Clusters')


plt.tight_layout()
plt.show()

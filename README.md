# Clustering Project
![Last Commit](https://img.shields.io/github/last-commit/JPP-J/clustering_project?style=flat-square)
![Jupyter Notebook](https://img.shields.io/badge/jupyter%20notebook-98.9%25-blue?style=flat-square)
![Languages](https://img.shields.io/github/languages/count/JPP-J/clustering_project?style=flat-square)

This repo is home to the code that accompanies Jidapa's *Clustering Project* , which provides; 
- example python code for clustering data on [winequality_red](https://drive.google.com/file/d/1jXQIFh6y3xo52byug_UcqBdtZgjOMn_D/view?usp=drive_link) and [example results](example_results_clustering.ipynb) details with:
  - K-mean algorithms
  - Aggomerative (Hierarchical Clustering bottom-up) 
  - DBSCAN (Density based  Clustering) 
- [rapid miner report](https://drive.google.com/file/d/1n-Islo_OX2Ijr09WMZmhxFzlv-Of2SO2/view?usp=drive_link) for Clustering:
  - K-means
  - Top-down clustering/Divisive (Hierarchical Clustering)
 
  ---

This project clusters the [Wine Quality dataset (red wine)](https://drive.google.com/file/d/1jXQIFh6y3xo52byug_UcqBdtZgjOMn_D/view?usp=drive_link) using various algorithms:

- **K-Means Clustering**
- **Agglomerative (Hierarchical) Clustering**
- **DBSCAN (Density-Based Clustering)**

## Dataset overview

- 1599 samples with 12 features including quality score.
- Balanced subset used for clustering: Quality 5, 6, 7 each with 199 samples.

## Libraries Used
- `pandas`
- `NumPy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## Results summary

| Algorithm           | Silhouette Score | Adjusted Rand Index | Davies-Bouldin Index |
|---------------------|------------------|---------------------|----------------------|
| K-Means (k=2)       | 0.8459           | 0.01845             | 0.6801               |
| Agglomerative (k=2) | 0.6313           | 0.01741             | 0.6931               |
| DBSCAN (eps=19)     | 0.8459           | 0.00002             | 0.1283               |


## Insights

- K-Means with **k=2** gave the best silhouette score.
- DBSCAN detected dense clusters but labeled most points as one cluster.
- Agglomerative clustering showed moderate performance.


## Usage

Run the clustering scripts:

- `cls_kmean.py`
- `cls_DBSCAN.py`
- `cls_agglomerative.py`

Each script outputs cluster quality metrics and visualizations can see in [demo results](example_results_clustering.ipynb)

## References

- [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
- [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)

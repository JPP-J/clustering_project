# 🍷 Clustering Project – Wine Quality Data Analysis
![Last Commit](https://img.shields.io/github/last-commit/JPP-J/clustering_project?style=flat-square)
![Jupyter Notebook](https://img.shields.io/badge/jupyter%20notebook-98.9%25-blue?style=flat-square)
![Languages](https://img.shields.io/github/languages/count/JPP-J/clustering_project?style=flat-square)

This repo is home to the code that accompanies Jidapa's *Clustering Project* , which provides; 

## 📌 Overview

This project applies unsupervised learning to cluster the **[Wine Quality (Red)](https://drive.google.com/file/d/1jXQIFh6y3xo52byug_UcqBdtZgjOMn_D/view?usp=drive_link)** dataset using multiple clustering algorithms. The goal is to group wines with similar characteristics and assess the effectiveness of each clustering approach.



## 🧩 Problem Statement

Understanding the underlying structure of wine quality data can be challenging due to its multidimensional nature. This project aims to identify natural groupings of wines using clustering techniques to assist in exploratory data analysis and potentially uncover hidden patterns that affect wine quality.


## 🔍 Approach

A series of **unsupervised clustering algorithms** were applied to the dataset, including **K-Means**, **Agglomerative Hierarchical Clustering**, and **DBSCAN**. Results were evaluated using clustering quality metrics to determine which method best captured the natural groupings in the data.



## 🎢 Processes

1. **ETL (Extract, Transform, Load)** – Load and preprocess the Wine Quality (Red) dataset  
2. **EDA** – Explore data through PCA 2D 
3. **Clustering Techniques** – Apply and tune:
   - K-Means combine with suitable k value (e.g., 2-10)
   - Agglomerative (Hierarchical) Clustering combine with suitable k value (e.g., 2-10)
   - DBSCAN combine with grid search parameters  
4. **Evaluation** – Use clustering validation metrics:
   - Silhouette Score
   - Adjusted Rand Index (ARI)
   - Davies-Bouldin Index (DBI)  
5. **Visualization** – Use scatter plots with PCA2D and PCA3D, luster bar each class and , Centroid Plot of Clusters to interpret cluster behavior



## 🎯 Results & Impact

| Algorithm           | Silhouette Score | Adjusted Rand Index | Davies-Bouldin Index |
|---------------------|------------------|---------------------|----------------------|
| K-Means (k=2)       | 0.8459           | 0.01845             | 0.6801               |
| Agglomerative (k=2) | 0.6313           | 0.01741             | 0.6931               |
| DBSCAN (eps=19)     | 0.8459           | 0.00002             | 0.1283               |

- **K-Means** achieved the best silhouette score, showing compact and well-separated clusters, can see [demo output](example_results_clustering.ipynb).
- **DBSCAN** revealed dense cluster structures but assigned many points to a single group. 
- **Agglomerative Clustering** performed moderately, offering insights into hierarchical structure.

### Insights

- K-Means with **k=2** gave the best silhouette score.
- DBSCAN detected dense clusters but labeled most points as one cluster.
- Agglomerative clustering showed moderate performance.



## ⚙️ Development Challenges

Clustering high-dimensional, real-world data poses several challenges:

- **Choosing the Number of Clusters (k):** K selection impacts algorithm performance, particularly in K-Means and Agglomerative.
- **Distance Metric Sensitivity:** Hierarchical and DBSCAN clustering are heavily influenced by the choice of distance and linkage methods.
- **Parameter Tuning:** DBSCAN’s `eps` and `min_samples` parameters required careful tuning to avoid labeling most data as noise.
- **Imbalanced Data:** Some quality levels were underrepresented, so a balanced subset (e.g., 199 samples of quality 5, 6, and 7) was selected and rebalance selected features.

## 📊 Libraries Used
- `pandas`, `NumPy`, `matplotlib`, `seaborn`, `scikit-learn`

## 📁 Project Structure & Usage

- [`cls_kmean.py`](cls_kmean.py) – K-Means clustering implementation  
- [`cls_agglomerative.py`](cls_agglomerative.py) – Agglomerative clustering script  
- [`cls_DBSCAN.py`](cls_DBSCAN.py) – DBSCAN algorithm script  
- [`example_results_clustering.ipynb`](example_results_clustering.ipynb) – Visualizations and performance results  



## 📚 References & Resources

- [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
- [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)




> Other tool for clustering - rapid miner: [rapid miner report](https://drive.google.com/file/d/1n-Islo_OX2Ijr09WMZmhxFzlv-Of2SO2/view?usp=drive_link)

*This project provides a hands-on example of applying unsupervised learning techniques for exploratory data analysis, offering a foundation for further pattern discovery and segmentation tasks.*


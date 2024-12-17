import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

#  ======================================= NORMAL PLOT  =============================================
def plot_exploration_scatter_2d(x, y, xlabel, ylabel, title):
    plt.figure(figsize=(10, 6))
    plt.ylim(top=200)
    plt.scatter(x=x, y=y, c='b', alpha=0.3)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def plot_cluster_scatter_2d(df, centroid_df, x_col='', y_col='', title='', x_label='', y_label=''):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue='cluster', palette='plasma')
    plt.scatter(centroid_df.iloc[:,0], centroid_df.iloc[:,1], marker='*', s=100, c='black', label='centroids')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(title='cluster')

def plot_centroid_cluster(centroid_df):
    # Plot the centroids for each cluster
    plt.figure(figsize=(10, 6))
    for i in range(centroid_df.shape[0]):
        plt.plot(centroid_df.columns, centroid_df.iloc[i], label=f'Cluster {i}')

    # Add titles and labels
    plt.title('Centroid Plot of Clusters')
    plt.xlabel('Features')
    plt.ylabel('Centroid Values')
    plt.legend()
    plt.grid(visible=True, linestyle='--', alpha=0.5)

def plot_cluster_label_percentage(cluster_label_percentage, title):
    # Plot the percentage distribution of labels in each cluster as a stacked bar plot
    ax = cluster_label_percentage.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab20', zorder=3)

    # Add annotations on each bar segment
    for p in ax.patches:
        # Get the width, height and x-position of the rectangle (bar segment)
        width = p.get_width()
        height = p.get_height()
        x_position = p.get_x()
        y_position = p.get_y()

        # Annotate the bar with the percentage value (height of the segment)
        ax.text(x_position + width / 2, y_position + height / 2, f'{height:.1f}%',
                ha='center', va='center', fontsize=9, color='white')

    # Title and labels
    plt.title(f'{title}')
    plt.xlabel('Cluster')
    plt.ylabel('Percentage (%)')
    plt.legend(title='Labels', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y')
    plt.tight_layout()

# ======================================= PCA =======================================================
def plot_tnse_2d(X, title, n_components=2):
    # Reduce the dimensions to 2D for visualization using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X) # X.toarray()

    # Plot the clusters using t-SNE
    plt.figure(figsize=(10, 6))
    plt.scatter(X_tsne[:, 0],X_tsne[:, 1],alpha=0.8, zorder=3)
    # plt.colorbar()
    plt.title(f'{title}')
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    # plt.legend(loc='upper right')
    plt.grid(True)

def plot_pca_2d(X, title='', n_components=2):
    # df_cluster: df['cluster']
    # Reduce the dimensions to 2D for visualization using PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Plot the clusters
    # plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1],alpha=0.8, zorder=3)
    # plt.legend()
    plt.title(f'{title}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)


# ======================================= PCA + CLUSTER =============================================
# Format_1
# Reduce the dimensions to 2D for visualization using PCA
def plot_cls_pca_2d(X, df_cluster, title, n_components=2):
    # df_cluster: df['cluster']
    # Reduce the dimensions to 2D for visualization using PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Get unique clusters
    unique_clusters = df_cluster.unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_clusters)))  # Generate distinct colors from viridis

    # Plot the clusters
    plt.figure(figsize=(10, 6))
    for i, cluster in enumerate(unique_clusters):
        cluster_data = X_pca[df_cluster == cluster]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1],
                    color=colors[i], label=f'Cluster {cluster}', s=50, marker='o', zorder=3)
    # # Original Code
    # plt.figure(figsize=(10, 6))
    # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'], cmap='viridis', marker='o', s=50, label='Clusters')
    # plt.colorbar()  # For showing the cluster colors

    plt.legend()
    plt.title(f'{title}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)



# ===============================================================================
# Reduce to 3D using PCA
def plot_cls_pca_3d(X, df_cluster, title, n_components=3):
    # Get unique clusters
    unique_clusters = df_cluster.unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_clusters)))  # Generate distinct colors from viridis

    # Reduce to 3D using PCA
    pca_3d = PCA(n_components=n_components)
    X_pca_3d = pca_3d.fit_transform(X)

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for i, cluster in enumerate(unique_clusters):
        cluster_data = X_pca_3d[df_cluster == cluster]
        scatter = ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2],
                             color=colors[i], label=f'Cluster {cluster}', s=50, zorder=3)

    # Adding labels and a colorbar
    ax.set_title(f'{title}')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    fig.legend(loc='center left')
    # fig.colorbar(scatter)


def plot_cls_tnse_2d(X, df_cluster, title, n_components=2):
    # Reduce the dimensions to 2D for visualization using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # Get unique clusters
    unique_clusters = df_cluster.unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_clusters)))  # Generate distinct colors from viridis

    # Plot the clusters using t-SNE
    # plt.figure(figsize=(10, 6))
    for i, cluster in enumerate(unique_clusters):
        cluster_data = X_tsne[df_cluster == cluster]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1],
                    color=colors[i], label=f'Cluster {cluster}', marker='o', s=50, zorder=3)

    # plt.colorbar()
    plt.title(f'{title}')
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(loc='upper right')
    plt.grid(True)




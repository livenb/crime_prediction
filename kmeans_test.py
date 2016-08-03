import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist, pdist


def elbow_silhouette_kmeans(X):
    dists = []
    s_scores = []
    for k in xrange(2, 20):
        print k
        model = KMeans(n_clusters=k)
        model.fit(X)
        centroids = model.cluster_centers_
        labels = model.labels_
        dist = cdist(X, centroids)
        dists.append(dist.min(axis=1).sum())
        # s_scores.append(silhouette_score(X, labels, metric='euclidean'))
    plt.figure()
    plt.plot(xrange(2, 20), dists, 'b*-')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('sum of squares')
    plt.title('Elbow for KMeans clustering')
    # plt.figure()
    # plt.plot(xrange(2, 20), s_scores)
    # plt.grid(True)
    # plt.title('Choose K based on Silhouette Score')
    # plt.xlabel('Number of Cluster')
    # plt.ylabel('Silhouette Score')
    # plt.show()

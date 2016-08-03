import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist, pdist


def load_data(filename):
    df = pd.read_csv(filename)
    return df


def clean_data(sfdata):
    dropLst = ['WARRANTS', 'MISSING PERSON', 'OTHER OFFENSES',
               'NON-CRIMINAL', 'TRESPASSING', 'SUICIDE',
               'LIQUOR LAWS', 'RUNAWAY', 'EMBEZZLEMENT',
               'FAMILY OFFENSES', 'BAD CHECKS',
               'GAMBLING', 'BRIBERY', 'PORNOGRAPHY/OBSCENE MAT', 'TREA']
    print sfdata.info()
    sfdata = sfdata.drop(sfdata[sfdata['Category'].isin(dropLst)].index)
    print sfdata.info()
    sfdata = time_enginerring(sfdata)
    return sfdata


def time_enginerring(sfdata):
    sfdata['Hour'] = sfdata['Time'].apply(lambda x: int(x.split(':')[0]))
    sfdata['Month'] = sfdata['Date'].apply(lambda x: int(x.split('/')[0]))
    sfdata['Day'] = sfdata['Date'].apply(lambda x: int(x.split('/')[1]))
    sfdata['Year'] = sfdata['Date'].apply(lambda x: int(x.split('/')[2]))
    return sfdata


def silhouette_kmeans(X):
    s_scores = []
    for k in xrange(50):
        model = KMeans(n_clusters=k)
        model.fit(X)
        labels = model.labels_
        s_scores.append(silhouette_score(X, labels, metric='euclidean'))
    plt.plot(range(50), s_scores)
    plt.title('Choose K based on Silhouette Score')
    plt.xlabel('Number of Cluster')
    plt.ylabel('Silhouette Score')
    plt.show()


def elbow_kmeans(X):
    dists = []
    for k in xrange(2, 50):
        model = KMeans(n_clusters=k)
        model.fit(X)
        centroids = model.cluser_centers_
        dist = cdist(X, centroids)
        dists.append(dist.min(axis=1).sum())
    plt.plot(xrange(2, 50), dists, 'b*-')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('sum of squares')
    plt.title('Elbow for KMeans clustering')
    plt.show()


if __name__ == '__main__':
    sffile = 'data/SFPD.csv'
    lafile1 = 'data/LA_SHERIFF_1.csv'
    lafile2 = 'data/LA_SHERIFF_2.csv'
    sfdata = load_data(sffile)
    print sfdata.info()
    sfdata = clean_data(sfdata)
    X = sfdata[['X', 'Y']].values
    elbow_kmeans(X)
   

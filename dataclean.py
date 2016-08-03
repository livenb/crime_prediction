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
    # print sfdata.info()
    sfdata = sfdata.drop(sfdata[sfdata['Category'].isin(dropLst)].index)
    # print sfdata.info()
    sfdata = time_enginerring(sfdata)
    return sfdata


def time_enginerring(sfdata):
    sfdata['Hour'] = sfdata['Time'].apply(lambda x: int(x.split(':')[0]))
    sfdata['Month'] = sfdata['Date'].apply(lambda x: int(x.split('/')[0]))
    sfdata['Day'] = sfdata['Date'].apply(lambda x: int(x.split('/')[1]))
    sfdata['Year'] = sfdata['Date'].apply(lambda x: int(x.split('/')[2]))
    return sfdata

    
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
        s_scores.append(silhouette_score(X, labels, metric='euclidean'))
    plt.plot(xrange(2, 20), dists, 'b*-')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('sum of squares')
    plt.title('Elbow for KMeans clustering')
    plt.figure()
    plt.plot(range(50), s_scores)
    plt.title('Choose K based on Silhouette Score')
    plt.xlabel('Number of Cluster')
    plt.ylabel('Silhouette Score')
    plt.show()


if __name__ == '__main__':
    sffile = 'data/SFPD.csv'
    lafile1 = 'data/LA_SHERIFF_1.csv'
    lafile2 = 'data/LA_SHERIFF_2.csv'
    sfdata = load_data(sffile)
    # print sfdata.info()
    sfdata = clean_data(sfdata)
    X = sfdata[['X', 'Y']].values
    elbow_silhouette_kmeans(X)
    # model = KMeans()
    # model.fit(X)
    # print model.cluster_centers_
   

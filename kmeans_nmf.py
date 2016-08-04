import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist, pdist
from sklearn.decomposition import NMF


def elbow_silhouette_kmeans(X, year=None):
    dists = []
    s_scores = []
    for k in xrange(2, 20):
        print year, k
        model = KMeans(n_clusters=k)
        model.fit(X)
        centroids = model.cluster_centers_
        labels = model.labels_
        dist = cdist(X, centroids)
        dists.append(dist.min(axis=1).sum())
        s_scores.append(silhouette_score(X, labels, metric='euclidean'))
    fig = plt.figure()
    ax1 = fig.addsubplot(121)
    ax1.plot(xrange(2, 20), dists, 'b*-')
    ax1.grid(True)
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('sum of squares')
    ax1.title('Elbow for KMeans clustering {}'.format(year))
    ax2 = fig.addsubplot(122)
    ax2.plot(xrange(2, 20), s_scores)
    ax2.grid(True)
    ax2.set_title('Choose K based on Silhouette Score')
    ax2.set_xlabel('Number of Cluster')
    ax2.set_ylabel('Silhouette Score')
    # plt.show()


def kmeans_by_year(X, yrs, years):
    for yr in years:
        X_yr = X[yrs == yr]
        elbow_silhouette_kmeans(X_yr, yr)
    plt.show()


def build_nmf(X, yrs, years):
    Ws = []
    Hs = []
    for yr in years:
        X_yr = X[yrs == yr]
        nmfModel = NMF(n_components=10)
        W = nmfModel.fit_transform(X_yr)
        H = nmfModel.components_
        Ws.append(W)
        Hs.append(H)
    return Ws, Hs


def plot_cluster():
    plt.figure()
    plt.show()


if __name__ == '__main__':
    sfdata = pd.read_csv('data/sfpd_clean.csv')
    dropLst = ['Descript', 'PdDistrict', 'Address', 'Year']
    X = sfdata.drop(dropLst, axis=1).values
    yrs = sfdata['Year'].values
    years = sorted(sfdata['Year'].unique())
    kmeans_by_year(X, yrs, years)

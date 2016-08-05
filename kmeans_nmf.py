# import matplotlib
# matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist, pdist
from sklearn.decomposition import NMF
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from nltk.stem.wordnet import WordNetLemmatizer


def elbow_silhouette_kmeans(X, year=None):
    dists = []
    # s_scores = []
    for k in xrange(2, 50):
        print year, k
        model = KMeans(n_clusters=k)
        model.fit(X)
        centroids = model.cluster_centers_
        labels = model.labels_
        dist = cdist(X, centroids)
        dists.append(dist.min(axis=1).sum())
        # s_scores.append(silhouette_score(X, labels, metric='euclidean'))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(xrange(2, 50), dists, 'b*-')
    ax1.grid(True)
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('sum of squares')
    ax1.set_title('Elbow for KMeans clustering {}'.format(year))
    # ax2 = fig.addsubplot(122)
    # ax2.plot(xrange(2, 20), s_scores)
    # ax2.grid(True)
    # ax2.set_title('Choose K based on Silhouette Score')
    # ax2.set_xlabel('Number of Cluster')
    # ax2.set_ylabel('Silhouette Score')
    # plt.show()
    plt.savefig('img/{}_kmeans.png'.format(year))


def kmeans_by_year(X, yrs, years):
    for yr in years:
        X_yr = X[yrs == yr]
        elbow_silhouette_kmeans(X_yr, yr)
    # plt.show()
    print 'kmeans done!'


def get_lemmatized_word(line):
    lem = WordNetLemmatizer()
    line = line.lower().split()
    line = [lem.lemmatize(x.strip(string.punctuation).encode()) for x in line]
    return ' '.join(line)


def get_tfidf(content):
    content = [get_lemmatized_word(x)  for x in content]
    vec = TfidfVectorizer(stop_words='english')
    tfidf = vec.fit_transform(content)
    return tfidf
    
    
def build_nmf(X, yrs, years):
    Ws = []
    Hs = []
    for yr in years:
        print yr
        X_yr = X[yrs == yr]
        nmfModel = NMF(n_components=20)
        W = nmfModel.fit_transform(X_yr)
        H = nmfModel.components_
        Ws.append(W)
        Hs.append(H)
        print H
        sns.heatmap(H)
        plt.title('Heatmap of Lattent Feature - {}'.format(yr))
        plt.savefig('Heatmap of Lattent Feature - {}.png'.format(yr))
    return Ws, Hs


def plot_cluster(W):
    plt.figure()

    plt.show()


if __name__ == '__main__':
    sfdata = pd.read_csv('data/sfpd_clean.csv')
    dropLst = ['Descript', 'PdDistrict', 'Address', 'Year', 'Category']
    X = sfdata.drop(dropLst, axis=1).values
    yrs = sfdata['Year'].values
    years = sorted(sfdata['Year'].unique())
    # kmeans_by_year(X, yrs, years)
    Ws, Hs = build_nmf(X, yrs, years)
    tfidf = get_tfidf(sfdata['Descript'].values)
    X_t = np.concatenate((X, tfidf), axis=1)
    # kmeans_by_year(X_t, yrs, years)
    # Ws, Hs = build_nmf(X_t, yrs, years)


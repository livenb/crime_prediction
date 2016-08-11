import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import NMF


path = 'data/'


def load_file(filename):
    return pd.read_csv(path+filename+'.csv')


def kmean_pipline(df):
    X = df.drop(['Year', 'Month', 'zipcode'], axis=1).values
    scaler = MinMaxScaler()
    X_sca = scaler.fit_transform(X)
    model = KMeans(5, n_init=3)
    model.fit(X_sca)
    labels = model.labels_
    centroids = model.cluster_centers_ 
    df['class'] = labels
    cntClass = df.groupby(['Year', 'Month', 'class'])['zipcode'].count().unstack().reset_index()


def nmf_test(df):
    X = df.drop(['Year', 'zipcode'], axis=1).values
    scaler = MinMaxScaler()
    X_sca = scaler.fit_tranform(X)
    scores = []
    for k in xrange(2, 11):
        model = NMF(n_components=k)
        W = model.fit_transform(X_sca)
        labels = W.argmax(axis=1)
        score = silhouette_score(X_sca, labels)
        scores.append(score)
    plt.plot(xrange(2, 11), scores, 'b*-')
    plt.show()




if __name__ == '__main__':
    sffile = 'sfdf'
    lafile = 'ladf'
    louisfile = 'louisdf'
    defile = 'dedf'
    phfile = 'phdf'

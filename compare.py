import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


path = 'data/'


def load_file(filename):
    return pd.read_csv(path+filename+'.csv')


def kmean_pipline(df):
    X = df.drop(['Year', 'Month', 'zipcode'], axis=1).values
    scaler = MinMaxScaler()
    X_sca = scaler.fit_tranform(X)
    model = KMeans(5, n_init=3)
    model.fit(X)
    labels = model.labels_
    centroids = model.cluster_centers_ 
    df['class'] = labels
    cntClass = df.grouby(['Year', 'Month', 'class'])['zipcode'].count().unstack().reset_index()




if __name__ == '__main__':
    sffile = 'sfdf'
    lafile = 'ladf'
    louisfile = 'louisdf'
    defile = 'dedf'
    phfile = 'phdf'

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import nmf


def get_daily_data(ladf):
    ladaily = ladf.groupby(['Year', 'Month',
                            'Day', 'ZIP', 'CrimeCat'])['Unnamed: 0'].count()
    ladaily = ladaily.unstack()
    ladaily = ladaily.reset_index()
    ladaily = ladaily.fillna(0)
    return ladaily


def get_monthly_data(ladf):
    lamon = ladf.groupby(['Year', 'Month',
                          'ZIP', 'CrimeCat'])['Unnamed: 0'].count()
    lamon = lamon.unstack()
    lamon = lamon.reset_index()
    lamon = lamon.fillna(0)
    return lamon


if __name__ == '__main__':
    ladf = pd.read_csv('data/la_clean.csv')
    # ladaily = get_daily_data(ladf)
    lamon = get_monthly_data(ladf)

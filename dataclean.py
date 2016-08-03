import pandas as pd


def load_data(filename):
    df = pd.read_csv(filename)
    return df


def clean_sf_data(sfdata):
    dropLst = ['WARRANTS', 'MISSING PERSON', 'OTHER OFFENSES',
               'NON-CRIMINAL', 'TRESPASSING', 'SUICIDE',
               'LIQUOR LAWS', 'RUNAWAY', 'EMBEZZLEMENT',
               'FAMILY OFFENSES', 'BAD CHECKS',
               'GAMBLING', 'BRIBERY', 'PORNOGRAPHY/OBSCENE MAT', 'TREA']
    sfdata = sfdata.drop(sfdata[sfdata['Category'].isin(dropLst)].index)
    sfdata.drop(sfdata[sfdata['Y'] > 40].index, inplace=True)
    sfdata = time_enginerring(sfdata)
    return sfdata


def time_enginerring(sfdata):
    sfdata['Hour'] = sfdata['Time'].apply(lambda x: int(x.split(':')[0]))
    sfdata['Month'] = sfdata['Date'].apply(lambda x: int(x.split('/')[0]))
    sfdata['Day'] = sfdata['Date'].apply(lambda x: int(x.split('/')[1]))
    sfdata['Year'] = sfdata['Date'].apply(lambda x: int(x.split('/')[2]))
    return sfdata

    


if __name__ == '__main__':
    sffile = 'data/SFPD.csv'
    lafile1 = 'data/LA_SHERIFF_1.csv'
    lafile2 = 'data/LA_SHERIFF_2.csv'
    sfdata = load_data(sffile)
    # print sfdata.info()
    sfdata = clean_sf_data(sfdata)
    X = sfdata[['X', 'Y']].values
    elbow_silhouette_kmeans(X)
   

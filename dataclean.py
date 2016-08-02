import pandas as pd
import numpy as np

class crimedata(object):
    def __init__(self, df):
       self.df = df


def load_data(filename):
    df = pd.read_csv(filename)
    return df
    

def clean_data(df):
    pass


if __name__ == '__main__':
    sffile = 'data/SFPD.csv'
    lafile1 = 'data/LA_SHERIFF_1.csv'
    lafile2 = 'data/LA_SHERIFF_2.csv'
    sfdata = load_data(sffile)


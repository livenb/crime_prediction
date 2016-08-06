import pandas as pd


sfdata = pd.read_csv('data/sfpd_clean.csv')
sfdf = sfdata.groupby(['Year', 'PdDistrict','CrimeCat'])['Unnamed: 0'].count()
sfdf.unstack().reset_index()

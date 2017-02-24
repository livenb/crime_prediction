import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import fiona
from itertools import chain
from descartes import PolygonPatch
from shapely.geometry import Polygon
from matplotlib.collections import PatchCollection
from sklearn.decomposition import NMF
from pysal.esda.mapclassify import Natural_Breaks
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches


colormaps = ['Blues', 'Oranges', 'Greens', 'Reds', 'Purples']
classes = ['pattern 1: Theft & Burglary', 'pattern 2: Drug related',
           'pattern 3: Minor crime & Assaut', 'pattern 4: Prostitution',
           'pattern 5: Vehicle Theft']
rev_map = {'Telegraph Hill': 'North Beach',
           'Central Sunset': 'Sunset District',
           'Duboce Triangle': 'Hayes Valley',
           'Forest Hill': 'West of Twin Peaks',
           'Forest Hills Extension': 'West of Twin Peaks',
           'Forest Knolls': 'West of Twin Peaks',
           'Golden Gate Heights': 'Sunset District',
           'Inner Parkside': u'Sunset District',
           'Lakeside': 'Lake Shore',
           'Lincoln Park': 'Sea Cliff',
           'Little Hollywood': 'Visitacion Valley',
           'Lone Mountain': 'Inner Richmond',
           'Lower Pacific Heights': 'Pacific Heights',
           'Merced Heights': 'Oceanview',
           'Merced Manor': 'Lake Shore',
           'Mission Bay': 'South of Market',
           'Outer Parkside': 'Sunset District',
           'Outer Sunset': 'Sunset District',
           'Parkside': 'Sunset District',
           'Stonestown': 'Lake Shore',
           'West Portal': 'West of Twin Peaks',
           'Yerba Buena': 'South of Market',
           'Sunnyside': 'Westwood Park',
           'Pine Lake Park': 'Lake Shore',
           'Hunters Point': 'Bayview',
           'Candlestick Point': 'Bayview',
           'Bayview Heights': 'Bayview',
           }


def build_map_obj(shapefile):
    shp = fiona.open(shapefile + '.shp')
    bds = shp.bounds

    shp.close()
    extra = 0.01
    ll = (bds[0], bds[1])
    ur = (bds[2], bds[3])
    coords = list(chain(ll, ur))

    w, h = coords[2] - coords[0], coords[3] - coords[1]

    m = Basemap(
                projection='tmerc',
                lon_0=(coords[0] + coords[2])/2,
                lat_0=(coords[1] + coords[3])/2,
                ellps='helmert',
                llcrnrlon=coords[0] - w * 0.01,
                llcrnrlat=coords[1] - extra + 0.01 * h,
                urcrnrlon=coords[2] + w * 0.01,
                urcrnrlat=coords[3] + extra + 0.01 * h,
                resolution='i',
                suppress_ticks=True
                )

    m.readshapefile(shapefile,
                    'SF',
                    color='black',
                    zorder=2
                    )
    return m, coords


def plot_map(m, coords, df_map, info, savefig=False):
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, axisbg='w', frame_on=True)
    # draw wards with grey outlines
    norm = Normalize()
    for i in xrange(5):
        color = colormaps[i]
        cmap = plt.get_cmap(color)
        cond = (df_map['class'] == (i+1))
        inx = df_map[cond].index
        if cond.sum() > 0:
            pc = PatchCollection(df_map[cond]['patches'],
                                 match_original=True, alpha=0.75)
            pc.set_facecolor(cmap(norm(df_map.loc[inx, 'cls_%d'%(i+1)].values)))
            ax.add_collection(pc)
    if (df_map['class'] == 0).sum() > 0:
        pc = PatchCollection(df_map[df_map['class'] == 0]['patches'],
                             match_original=True, alpha=0.1
                             )
        pc.set_facecolor('grey')
        ax.add_collection(pc)
    x, y = m(coords[0], coords[3]+0.006)

    details = ax.annotate(info, xy=(x, y), size=20, color='k')

    # Draw a map scale
    m.drawmapscale(
        coords[0]+0.02, coords[1]-0.004,
        coords[0], coords[1],
        2,
        barstyle='fancy', labelstyle='simple',
        fillcolor1='w', fillcolor2='#555555',
        fontcolor='#555555', units='mi',
        zorder=5)

    legend_patches = []
    for i in range(5):
        legend_patches.append(mpatches.Patch(color='C%d' % i,
                                             label=classes[i]))
    ax.legend(handles=legend_patches, loc='upper right')

    fig.set_size_inches(12, 12)
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=200, alpha=True)


def clean_crime(df):
    df['Category'] = df['Category'].apply(lambda x: 'SEX OFFENSES'
                                          if x == 'SEX OFFENSES, FORCIBLE'
                                          or x == 'SEX OFFENSES, NON FORCIBLE'
                                          else x
                                          )
    droplist = ['BAD CHECKS', 'BRIBERY', 'EXTORTION', 'GAMBLING', 'SUICIDE',
                'NON-CRIMINAL', 'TREA', 'PORNOGRAPHY/OBSCENE MAT']
    return df[~df['Category'].isin(droplist)]


def weight_mat(X):
    idf = np.log(X.shape[0]/(X > 0).sum(axis=0))
    X_add = (np.multiply(X, idf) + 0.5 * X)
    return X_add


def load_data(filename):
    df = pd.read_csv(filename, parse_dates=['Date'])
    df = clean_crime(df)
    crimes = df['Category'].unique()
    gp = df[df['Year'] != 2017].groupby(['Year', 'Neighborhood', 'Category'])
    yr_cnt = gp['IncidntNum'].count().unstack().reset_index()
    yr_cnt.fillna(0, inplace=True)
    return yr_cnt, crimes


def build_nmf(X, k=5):
    mod = NMF(n_components=k)
    W = mod.fit_transform(X)
    H = mod.components_
    return W, H


def get_patch_df(m):
    df_plot = pd.DataFrame({
                'poly': [Polygon(xy) for xy in m.SF],
                'Neighborhood': [ward['nbrhood'] for ward in m.SF_info]})
    df_plot['patches'] = df_plot['poly'].map(lambda x: PolygonPatch(x,
                                                                    ec='#555555',
                                                                    lw=.2,
                                                                    alpha=1.0,
                                                                    zorder=4
                                                                    )
                                             )
    return df_plot


if __name__ == '__main__':
    shapefile = 'map/geo_export_50b99cdd-c217-4020-a75f-d2d864fc4e9b'
    filename = 'sfpd_neib.csv'

    yr_cnt, crimes = load_data(filename)
    X = yr_cnt[crimes].values
    X = weight_mat(X)
    W, H = build_nmf(X, k=5)

    m, coords = build_map_obj(shapefile)
    W_df = pd.DataFrame(W, columns=['cls_%d' % (i+1) for i in range(5)])
    for col in W_df.columns:
        W_df.loc[:, col] = Natural_Breaks(W_df.loc[:, col], k=5).yb
    yr_cnt = pd.concat([yr_cnt, W_df], axis=1)
    yr_cnt['class'] = W.argmax(axis=1) + 1
    cl = ['class'] + ['cls_%d' % (i+1) for i in range(5)]

    df_plot = get_patch_df(m)

    cnt = 1
    for yr in range(2003, 2017):
        cond = (yr_cnt['Year'] == yr)
        df_new = yr_cnt[cond]
        df_map = pd.merge(df_plot, df_new, how='left', on='Neighborhood')
        for name in rev_map:
            inx1 = df_map[df_map['Neighborhood'] == name].index
            val = df_new[df_new['Neighborhood'] == rev_map[name]][cl].values
            df_map.set_value(inx1, cl, val)
        df_map.fillna(0, inplace=True)
        info = 'San Franscisco Commuties Categoried by Crime -- {}'.format(yr)
        name = 'img/year_{:03d}.png'.format(cnt)
        plot_map(m, coords, df_map, info, name)
        cnt += 1

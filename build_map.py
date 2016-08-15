import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from mpl_toolkits.basemap import Basemap
import fiona
from descartes import PolygonPatch
from shapely.geometry import shape, mapping
from shapely.geometry import Point, Polygon, MultiPoint, MultiPolygon
from shapely.prepared import prep
from pyproj import Proj, transform
from fiona.crs import from_epsg
from itertools import chain
from pysal.esda.mapclassify import Natural_Breaks


crimes = {1: 'Theft/Larcery', 2: 'Robebery', 3: 'Nacotic/Alcochol',
          4: 'Assault', 5: 'Grand Auto Theft', 6: 'Vandalism',
          7: 'Burglary', 8: 'Homicide', 9: 'Sex Crime', 10: 'DUI'}


def get_year_df(df):
    layr = df.groupby(['Year', 'ZIP', 'CrimeCat'])['Unnamed: 0'].count()
    layr = layr.unstack().reset_index()
    layr = layr.fillna(0)
    return layr


def get_month_df(df):
    lamo = df.groupby(['Year', 'Month', 'ZIP', 'CrimeCat'])['Unnamed: 0'].count()
    lamo = lamo.unstack().reset_index()
    lamo = lamo.fillna(0)
    return lamo 


def get_quater(x):
    if x < 4:
        return 1
    elif 4 <= x < 7:
        return 2
    elif 7 <= x < 10:
        return 3
    else:
        return 4


def get_quater_df(df):
    laqt = get_month_df(df)
    laqt['Quater'] = laqt['Month'].apply(get_quater)
    laqt = laqt.drop('Month', axis=1).groupby(['Year', 'Quater', 'ZIP']).sum()
    laqt = laqt.reset_index()
    return laqt


def build_nmf_all(X, k=5):
    scaler = MinMaxScaler()
    X_sca = scaler.fit_transform(X)
    nmfModel = NMF(n_components=k)
    W = nmfModel.fit_transform(X_sca)
    H = nmfModel.components_
    print 'NMF done!'
    plot_heatmap(H.T, k=k)
    labelsNMF = W.argmax(axis=1)
    return W, H, labelsNMF, nmfModel

def plot_heatmap(data, title=None, k=5):
    fig, ax = plt.subplots(figsize = (12, 9))
    heatmap = ax.pcolor(data, cmap=plt.cm.Blues)

    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(k)+0.5, minor=False, )
    ax.set_yticks(np.arange(10)+0.5, minor=False)

    # want a more natural, table-like display
#     ax.invert_yaxis()
#     ax.xaxis.tick_top()
    classLabel = ['cls-{}:'.format(i) for i in range(1, k+1)]
    ax.set_xticklabels(classLabel, minor=False)
    ax.set_yticklabels(crimes.values(), minor=False)
    ax.set_title('Heatmap of Communites Crime Topics')
    if title != None:
        plt.savefig(title)
    # plt.show()


def get_basemap(shapefile):
    shp = fiona.open(shapefile+'.shp')
    bds = shp.bounds
    shp.close()
    extra = 0.01
    ll = (bds[0], bds[1])
    ur = (bds[2], bds[3])
    coords = list(chain(ll, ur))
    w, h = coords[2] - coords[0], coords[3] - coords[1]
    m = Basemap(projection='tmerc',
                lon_0= -118.2437,
                lat_0= 34.0522,
                ellps = 'WGS84',
                llcrnrlon=coords[0],
            #     llcrnrlat=coords[1] - extra + 0.01 * h,
                llcrnrlat=33.6,
                urcrnrlon=coords[2],
                urcrnrlat=coords[3],
                lat_ts=0,
                resolution='i',
                suppress_ticks=True)
    m.readshapefile(shapefile, 'LA', color='black', zorder=2)
    return m, coords

# Convenience functions for working with colour ramps and bars
def colorbar_index(ncolors, cmap, labels=None, **kwargs):
    """
    This is a convenience function to stop you making off-by-one errors
    Takes a standard colour ramp, and discretizes it,
    then draws a colour bar with correctly aligned labels
    """
    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors+0.5)
    colorbar = plt.colorbar(mappable, **kwargs)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
    colorbar.set_ticklabels(range(ncolors))
    if labels:
        colorbar.set_ticklabels(labels)
    return colorbar

def cmap_discretize(cmap, N):
    """
    Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet.
        N: number of colors.

    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)

    """
    if type(cmap) == str:
        cmap = get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0., 0., 0., 0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N + 1)
    cdict = {}
    for ki, key in enumerate(('red', 'green', 'blue')):
        cdict[key] = [(indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki]) for i in xrange(N + 1)]
    return LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)

def prep_df(df, W, labels):
    W_df =  pd.DataFrame(W)
    for col in W_df.columns:
        W_df.loc[:, col] = Natural_Breaks(W_df.loc[:, col], k=5).yb
    df['class'] = labels + 1
    df = pd.concat((df.drop(range(1,11), axis=1), W_df), axis=1)
    df['ZIP'] = df['ZIP'].apply(int).apply(str)
    return df

def get_df_map(df, m):
    df_map = pd.DataFrame({
            'poly': [Polygon(xy) for xy in m.LA],
            'zipcode': [ward['zipcode'] for ward in m.LA_info]})
    df_map = pd.merge(df_map, df, how='left', left_on='zipcode', right_on='ZIP',
                      left_index=True, right_index=False)
    df_map = df_map.fillna(0)
    df_map['patches'] = df_map['poly'].map(lambda x:
                                           PolygonPatch(x, ec='#555555',
                                                lw=.2, alpha=1.0, zorder=4))
#     print df_map.head()
    return df_map

def build_map_nmf(df_map, m, coords, info, title):
    # plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, axisbg='w', frame_on=True)
    # draw wards with grey outlines
    norm = Normalize()
    cmaps = []
    colors = ['Blues', 'Greens', 'Purples', 'Reds', 'Oranges']
    for i in xrange(5):
        color = colors[i]
        cmap = plt.get_cmap(color)
        pc = PatchCollection(df_map[df_map['class'] == i+1]['patches'], match_original=True, alpha=0.8)
        pc.set_facecolor(cmap(norm(df_map.loc[(df_map['class'] == i+1), i].values)))
        ax.add_collection(pc)
    pc = PatchCollection(df_map[df_map['class'] == 0]['patches'], match_original=True, alpha=0.2)
    pc.set_facecolor('grey')
    ax.add_collection(pc)
    x, y = m(coords[0] + 0.05, coords[1] + 1.0)
    details = plt.annotate(info, xy=(x, y), size=22, color='#555555')
    # Draw a map scale
    m.drawmapscale(
        coords[0] + 0.2, coords[1] + 0.95,
        coords[0], coords[1],
        20., fontsize=8,
        barstyle='fancy', labelstyle='simple',
        fillcolor1='w', fillcolor2='#555555',
        fontcolor='#555555', units='mi',
        zorder=5)
    plt.tight_layout()
    fig.set_size_inches(12, 16)
    plt.savefig(title, dpi=240, alpha=True)


def build_map_yr(data):
    layr = get_year_df(data)
    X_yr = layr.drop(['Year', 'ZIP'], axis=1).values
    W, H, labelsNMF, nmfModel = build_nmf_all(X_yr)
    plot_heatmap(H.T, 'img/year_H.png')
    m, coords = get_basemap(LA_shapefile)
    new_df = prep_df(layr, W, labelsNMF)
    for yr in xrange(2004, 2016):
        print yr
        df_map = get_df_map(new_df[new_df['Year'] == yr], m)
        title = 'img/la-map-{}.png'.format(yr)
        info = 'Los Angeles Commuties\nCategoried by Crime\n Year: {0}'.format(yr)
        build_map_nmf(df_map, m, coords, info, title)


def build_map_month_yrbased(data):
    layr = get_year_df(data)
    X_yr = layr.drop(['Year', 'ZIP'], axis=1).values
    Wyr, H, labelsNMFyr, nmfModel = build_nmf_all(X_yr)
    lamo = get_month_df(data)
    X_mo = lamo.drop(['Year', 'Month', 'ZIP'], axis=1).values
    print 'Transfer Monthly Class'
    W = nmfModel.transform(X_mo)
    labelsNMF = W.argmax(axis=1)
    print 'build base map'
    m, coords = get_basemap(LA_shapefile)
    print 'build new dataframe'
    new_df = prep_df(lamo, W, labelsNMF)
    for yr in xrange(2004, 2017):
        for mo in xrange(1, 13):
            print yr, mo
            df_map = get_df_map(new_df[(new_df['Year'] == yr) |
                                (new_df['Month'] == mo)], m)
            if mo < 10:
                title = 'img/la-map-{0}0{1}.png'.format(yr, mo)
            else:
                title = 'img/la-map-{0}{1}.png'.format(yr, mo)
            info = 'Los Angeles Commuties\nCategoried by Crime\n{0}--{1}'.format(yr, mo)
            build_map_nmf(df_map, m, coords, info, title)


def build_map_month(data):
    lamo = get_month_df(data)
    X_mo = lamo.drop(['Year', 'Month', 'ZIP'], axis=1).values
    W, H, labelsNMF, nmfModel = build_nmf_all(X_mo)
    m, coords = get_basemap(LA_shapefile)
    new_df = prep_df(lamo, W, labelsNMF)
    for yr in xrange(2004, 2017):
        for mo in xrange(1, 13):
            print yr, mo
            df_map = get_df_map(new_df[(new_df['Year'] == yr) |
                                (new_df['Month'] == mo)], m)
            if mo < 10:
                title = 'img/la-map-{0}0{1}.png'.format(yr, mo)
            else:
                title = 'img/la-map-{0}{1}.png'.format(yr, mo)
            info = 'Los Angeles Commuties\nCategoried by Crime\n{0}--{1}'.format(yr, mo)
            build_map_nmf(df_map, m, coords, info, title)
    plot_heatmap(H.T, 'img/month.png')
    return new_df

def build_map_quater(data):
    laqt = get_quater_df(data)
    X_qt = laqt.drop(['Year', 'Quater', 'ZIP'], axis=1).values
    W, H, labelsNMF, nmfModel = build_nmf_all(X_qt)
    m, coords = get_basemap(LA_shapefile)
    new_df = prep_df(laqt, W, labelsNMF)
    cnt = 1
    for yr in xrange(2004, 2017):
        for qt in xrange(1, 5):
            print yr, qt
            df_map = get_df_map(new_df[(new_df['Year'] == yr) |
                                (new_df['Quater'] == qt)], m)
            if cnt < 10:
                title = 'img/la00{}.png'.format(cnt)
            else:
                title = 'img/la0{}.png'.format(cnt)
            info = 'Los Angeles Commuties\nCategoried by Crime\nYear{0}-Qt.{1}'.format(yr, qt)
            build_map_nmf(df_map, m, coords, info, title)
            cnt += 1
    plot_heatmap(H.T, 'img/quater_H.png')
    return new_df


if __name__ == '__main__':
    LA_shapefile = 'map/LA-ZIPCodes/geo_export_1cf2ba2c-a35e-47e7-a586-b3ff394055e9'
    ladata = pd.read_csv('data/la_clean.csv')
    newdf = build_map_yr(ladata)
    # newdf = build_map_month(ladata)
    # build_map_month_yrbased(ladata)
    # newdf = build_map_quater(ladata)
    # plt.show()

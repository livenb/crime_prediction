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

    X_mo = lamo.drop(['Year', 'ZIP'], axis=1).values
    return X_mo

def build_nmf_all(X, k=5):
    nmfModel = NMF(n_components=k)
    W = nmfModel.fit_transform(X)
    H = nmfModel.components_
    print 'NMF done!'
    plot_heatmap(H.T, k)
    return W, H

def plot_heatmap(data, k):
    fig, ax = plt.subplots(figsize = (8,8))
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
    ax.set_title('Heatmap of Lattent Feature')
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
                llcrnrlon=coords[0] - extra * w,
            #     llcrnrlat=coords[1] - extra + 0.01 * h,
                llcrnrlat=33.6 - extra + 0.01 * h,
                urcrnrlon=coords[2] + extra * w,
                urcrnrlat=coords[3] + extra + 0.01 * h,
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
        W_df.loc[:, col] = Natural_Breaks(W_df.loc[:, col], k=10).yb
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

def build_map_nmf(df_map, m, coords, yr, title):
    # plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, axisbg='w', frame_on=True)
    # draw wards with grey outlines
    norm = Normalize()
    cmaps = []
    colors = ['Blues', 'Greens', 'Oranges', 'Reds', 'Purples']
    for i in xrange(5):
        color = colors[i]
        cmap = plt.get_cmap(color)
        pc = PatchCollection(df_map[df_map['class'] == i+1]['patches'], match_original=True, alpha=0.8)
        pc.set_facecolor(cmap(norm(df_map.loc[(df_map['class'] == i+1), i].values)))
        ax.add_collection(pc)
    pc = PatchCollection(df_map[df_map['class'] == 0]['patches'], match_original=True, alpha=0.2)
    pc.set_facecolor('Grey')
    ax.add_collection(pc)
    x, y = m(coords[0], coords[1] + 1.0)
    info = 'Los Angeles Commuties\n Categoried by Crime\n   {}'.format(yr)
    details = plt.annotate(info, xy=(x, y), size=18, color='#555555')
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
    scaler = MinMaxScaler()
    X_sca = scaler.fit_transform(X_yr)
    W, H = build_nmf_all(X_sca)
    labelsNMF = W.argmax(axis=1)
    # years = layr['Year']
    # yrs = sorted(years.unique())
    m, coords = get_basemap(LA_shapefile)
    new_df = prep_df(layr, W, labelsNMF)
    for yr in xrange(2004, 2016):
        print yr
        df_map = get_df_map(new_df[new_df['Year'] == yr], m)
        title = 'img/la-map-{}.png'.format(yr)
        build_map_nmf(df_map, m, coords, yr, title)


def build_map_month(data):
    lamo = get_month_df(data)
if __name__ == '__main__':
    LA_shapefile = 'map/LA-ZIPCodes/geo_export_1cf2ba2c-a35e-47e7-a586-b3ff394055e9'
    ladata = pd.read_csv('data/la_clean.csv')
    # df_map = get_df_map(new_df[new_df['Year'] == 2004])
    # build_yr_map_nmf(df_map, m, coords, 2004)
    build_map_yr(ladata)
    plt.show()
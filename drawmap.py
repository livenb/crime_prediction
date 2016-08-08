import pandas as pd
import numpy as np
from mpl_toolkits.basemap import Basemap
import fiona
from pyproj import Proj, transform
from fiona.crs import from_epsg
from itertools import chain
from descartes import PolygonPatch
from shapely.geometry import shape, mapping
from shapely.geometry import Point, Polygon, MultiPoint, MultiPolygon
from shapely.prepared import prep
from pysal.esda.mapclassify import Natural_Breaks
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
# %matplotlib inline

LAcenter = (-118.2437, 34.0522)
LAshapefile = 'map/LA-ZIPCodes/geo_export_1cf2ba2c-a35e-47e7-a586-b3ff394055e9'
SFcenter = (-122.4194, 37.7749)
SFshapefile = 'map/SF-ZipCodes/SFZipCodes'
#     llcrnrlat=33.5 - extra + 0.01 * h,


def build_basemap(shapefile, cityname, center, extra=0.01):
    shp = fiona.open(shapefile + '.shp')
    bds = shp.bounds
    shp.close()
    ll = (bds[0], bds[1])
    ur = (bds[2], bds[3])
    coords = list(chain(ll, ur))
    w, h = coords[2] - coords[0], coords[3] - coords[1]
    m = Basemap(projection='tmerc',
                lon_0=center[0],
                lat_0=center[1],
                ellps='WGS84',
                llcrnrlon=coords[0] - extra * w,
                llcrnrlat=coords[1] - extra + 0.01 * h,
                urcrnrlon=coords[2] + extra * w,
                urcrnrlat=coords[3] + extra + 0.01 * h,
                lat_ts=0,
                resolution='i',
                suppress_ticks=True)
    m.readshapefile(shapefile,
                    cityname,
                    color='black',
                    zorder=2)
    return m, coords


def makePoints(dat):
    # Create Point objects in map coordinates from dataframe lon and lat values
    map_points = [Point(m(mapped_x, mapped_y))
                  for mapped_x, mapped_y
                  in zip(dat['longitude'], dat['latitude'])]
    map_points = pd.Series()
    plt_points = MultiPoint(list(map_points.values))
    hoods_polygon = prep(MultiPolygon(list(df_map['poly'].values)))
    pts = filter(hoods_polygon.contains, plt_points)
    return pts


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
        cdict[key] = [(indices[i], colors_rgba[i - 1, ki],
                      colors_rgba[i, ki])
                      for i in xrange(N + 1)]
    return LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)


def plot_scatter_map(df_map, coords, m):
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, axisbg='w', frame_on=True)
    # draw ward patches from polygons
    df_map['patches'] = df_map['poly'].map(lambda x: PolygonPatch(
        x,
        fc='#555555',
        ec='w', lw=.25, alpha=.9,
        zorder=4))
    # we don't need to pass points to m() because we calculated using map_points and shapefile polygons
    # dev = m.scatter(
    #     [geom.x for geom in ldn_points],
    #     [geom.y for geom in ldn_points],
    #     5, marker='o', lw=.25,
    #     facecolor='#33ccff', edgecolor='w',
    #     alpha=0.9, antialiased=True,
    #     label='Blue Plaque Locations', zorder=3)
    # plot boroughs by adding the PatchCollection to the axes instance
    ax.add_collection(PatchCollection(df_map['patches'].values, match_original=True))
    # copyright and source data info
    smallprint = ax.text(
        1.03, 0.2,
        'Total points:',
        ha='right', va='bottom',
        size=40,
        color='#555555',
        transform=ax.transAxes)

    # Draw a map scale
    m.drawmapscale(
        coords[0] + 0.18, coords[1] + 1.11,
        coords[0], coords[1],
        20.,
        barstyle='fancy', labelstyle='simple',
        fillcolor1='w', fillcolor2='#555555',
        fontcolor='#555555', units='mi',
        zorder=5)
    plt.title("Crime in LA")
    plt.tight_layout()
    # this will set the image width to 722px at 100dpi
    fig.set_size_inches(15, 20)
    # plt.savefig('data/london_plaques.png', dpi=100, alpha=True)
    plt.show()


def plot_choropleth_map(df_map, coords, m):
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, axisbg='w', frame_on=False)

    # use a blue colour ramp - we'll be converting it to a map using cmap()
    cmap = plt.get_cmap('Blues')
    # draw wards with grey outlines
    df_map['patches'] = df_map['poly'].map(lambda x: PolygonPatch(x, ec='#555555', lw=.2, alpha=1., zorder=4))
    pc = PatchCollection(df_map['patches'], match_original=True)
    # impose our colour map onto the patch collection
    norm = Normalize()
    pc.set_facecolor(cmap(norm(df_map['jenks_bins'].values)))
    ax.add_collection(pc)

    # Add a colour bar
    cb = colorbar_index(ncolors=len(df_map['jenks_bins'].unique()),
                        cmap=cmap, shrink=0.5, labels=None)
    cb.ax.tick_params(labelsize=30)

    # # Show highest densities, in descending order
    # highest = '\n'.join(
    #     value[1] for _, value in df_map[(df_map['jenks_bins'] == 4)][:10].sort().iterrows())
    # highest = 'Most Dense Wards:\n\n' + highest
    # # Subtraction is necessary for precise y coordinate alignment
    # details = cb.ax.text(
    #     -1., 0 - 0.007,
    #     highest,
    #     ha='right', va='bottom',
    #     size=5,
    #     color='#555555')

    # Bin method, copyright and source data info
    # smallprint = ax.text(
    #     1.03, 0,
    #     'Classification method: natural breaks\nContains Ordnance Survey data\n$\copyright$ Crown copyright and database right 2013\nPlaque data from http://openplaques.org',
    #     ha='right', va='bottom',
    #     size=4,
    #     color='#555555',
    #     transform=ax.transAxes)

    # Draw a map scale
    m.drawmapscale(
        coords[0] + 0.2, coords[1] + 1.015,
        coords[0], coords[1],
        20.,
        barstyle='fancy', labelstyle='simple',
        fillcolor1='w', fillcolor2='#555555',
        fontcolor='#555555', units='mi',
        zorder=5)
    # this will set the image width to 722px at 100dpi
    plt.tight_layout()
    fig.set_size_inches(15, 20)
    # plt.savefig('data/london_plaques.png', dpi=100, alpha=True)
    plt.show()


if __name__ == '__main__':
    m, coords = build_basemap(LAshapefile, 'LA', LAcenter)
    # Set up a map dataframe
    df_map = pd.DataFrame({
        'poly': [Polygon(xy) for xy in m.LA],
        'zipcode': [ward['zipcode'] for ward in m.LA_info]})
    df_map['area_m'] = df_map['poly'].map(lambda x: x.area)
    df_map['area_km'] = df_map['area_m'] / 100000

    plot_scatter_map(df_map, coords, m)

    df_map['count'] = np.random.randint(1000, 10000, df_map.shape[0])
    # Calculate Jenks natural breaks for density
    breaks = Natural_Breaks(df_map['count'].values, k=5)
    df_map['jenks_bins'] = breaks.yb
    plot_choropleth_map(df_map, coords, m)
    plt.show()

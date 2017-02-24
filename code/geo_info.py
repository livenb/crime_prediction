import pandas as pd
import geocoder
import pickle

key = ''
neib_map = {u'Ashbury Heights': 'Buena Vista Park/Ashbury Heights',
            u'Balboa Park': 'Mission Terrace',
            u'Bayshore Heights': 'Visitacion Valley',
            u'Buena Vista': 'Buena Vista Park/Ashbury Heights',
            u'Central Waterfront': 'Central Waterfront/Dogpatch',
            u'Chinatown': 'Nob Hill',
            u'Dogpatch': 'Central Waterfront/Dogpatch',
            u'Civic Center': 'Van Ness/Civic Center',
            u'Cole Valley': 'Cole Valley/Parnassus Heights',
            u'Crocker-Amazon': 'Crocker Amazon',
            u'Dolores Heights': 'Eureka Valley / Dolores Heights',
            u'Eureka Valley': 'Eureka Valley / Dolores Heights',
            u'Fillmore District': 'Van Ness/Civic Center',
            u'Financial District': 'Financial District/Barbary Coast',
            u'Embarcadero': 'Financial District/Barbary Coast',
            u"Fisherman's Wharf": 'North Waterfront',
            u'Haight-Ashbury': 'Haight Ashbury',
            u'Laurel Heights': 'Jordan Park / Laurel Heights',
            u'Lakeshore': 'Lake Shore',
            u'Lower Haight': 'Hayes Valley',
            u'Marina District': 'Marina',
            u'Mid-Market': 'South of Market',
            u'Mission District': 'Inner Mission',
            u'NoPa': 'North Panhandle',
            u'Polk Gulch': 'Van Ness/Civic Center',
            u'Somisspo': 'Inner Mission',
            u'South Park': 'South Beach',
            u'Union Square': 'Downtown',
            u'Sunnydale': 'Visitacion Valley',
            u'Southern Hills': 'Visitacion Valley',
            u'The Castro': 'Eureka Valley / Dolores Heights',
            u'Vista del Mar': 'Outer Richmond',
            u'Mount Davidson': 'Miraloma Park',
            u'Panhandle': 'North Panhandle',
            u'Lower Nob Hill': 'Downtown',
            u'Westlake': 'Lake Shore',
            u'Ingleside Terraces': 'Ingleside Terrace'
            }

def get_geo_info(df, loadfile=False):
    loc_dict = {}
    if loadfile:
        loc_dict = pickle.load(open('loc_dict.pkl'))
    for loc in set(zip(df['Y'], df['X'])):
        if loc not in loc_dict:
            g = geocoder.google(loc, method='reverse', key=key)
            if g.status == 'OK':
                loc_dict[loc] = g.neighborhood
            else:
                pickle.dump(open('loc_dict.pkl'))
    nei_list = []
    for loc in zip(df['Y'], df['X']):
        nei_list.append(loc_dict.get(loc, 'None'))
    return nei_list


def add_neighborhood(df):
    nei_list = get_geo_info(df)
    df['Neighborhood'] = nei_list
    df['Neighborhood'].apply(lambda x: neib_map.get(x, x), inplace=True)
    return df

import pyprind
from geopy.exc impot GeocoderServiceError
import time
import pickle
from geopy.geocoders import Nominatim
import re

def get_zipcode(loc):
    geolocator = Nominatim()
    loc = re.sub(r'\(|\)', '', loc)
    location = geolocator.reverse(loc)
    zipcode = location.raw['address']['postcode']
    return zipcode

def get_zip_dict(locations):
    global locDict
    n = len(locations)
    bar = pyprind.ProgBar(n)
    with open('data/sfpost.txt', 'w+') as f:
        for i in xrange(n):
            loc = locations[i]
#             print loc
            if loc not in locDict:
                try:
                    zipcode = get_zipcode(loc)
#                     print zipcode
                except GeocoderServiceError as e:
                    print e.args[0]
                    if e.args[0] == 'HTTP Error 420: unused':
                        time.sleep(1800)
                        zipcode = get_zipcode(loc)
                    elif e.args[0] == '<urlopen error [Errno 10060] A connection attempt failed because the connected party did not properly respond after a period of time, or established connection failed because connected host has failed to respond>':
                        time.sleep(5)
                        zipcode = get_zipcode(loc)
                    elif e.args[0] == '<urlopen error [Errno 65] No route to host>':
                        time.sleep(1800)
                        zipcode = get_zipcode(loc)
                f.write('{}, {} \n'.format(loc, zipcode))
                locDict[loc] = zipcode
            bar.update()
    return locDict


if __name__:'__main__':
    filename = 'data/sfpd_clean.csv'
    sfdata = pd.read_csv(filename)
    locations = sfdata['Location'].unique()
    locDict1 = get_zip_dict(locations[:100], locDict)

import pandas as pd


crimes = {1: 'Theft/Larcery', 2: 'Robebery', 3: 'Nacotic/Alcochol',
          4: 'Assault', 5: 'Grand Auto Theft', 6: 'Vandalism',
          7: 'Burglary', 8: 'Homicide', 9: 'Sex Crime', 10: 'DUI'}


def load_data(filename):
    df = pd.read_csv(filename)
    return df


def clean_sf_data(filename):
    sfDict = {1: ['LARCENY/THEFT'],
              2: ['ROBBERY'],
              3: ['DRUNKENNESS', 'DRUG/NARCOTIC', 'LIQUOR LAWS'],
              4: ['ASSAULT'],
              5: ['VEHICLE THEFT'],
              6: ['VANDALISM'],
              7: ['BURGLARY'],
              9: ['PORNOGRAPHY/OBSCENE MAT', 'SEX OFFENSES, FORCIBLE',
                  'SEX OFFENSES, NON FORCIBLE'],
              10: ['DRIVING UNDER THE INFLUENCE']}
    sfdata = load_data(filename)
    sfdata['CrimeCat'] = 0
    for key in sfDict:
        crimes = sfDict[key]
        sfdata['CrimeCat'] += sfdata['Category'].apply(lambda x:
                                                       key if x in crimes
                                                       else 0)
    sfdata = sfdata.drop(sfdata[sfdata['CrimeCat'] == 0].index)
    sfdata.drop(sfdata[sfdata['Y'] > 40].index, inplace=True)
    sfdata['Hour'] = sfdata['Time'].apply(lambda x: int(x.split(':')[0]))
    sfdata['Month'] = sfdata['Date'].apply(lambda x: int(x.split('/')[0]))
    sfdata['Day'] = sfdata['Date'].apply(lambda x: int(x.split('/')[1]))
    sfdata['Year'] = sfdata['Date'].apply(lambda x: int(x.split('/')[2]))
    dropLst = ['Time', 'Date', 'Category', 'IncidntNum',
               'Location', 'Resolution', 'PdId']
    sfdata = sfdata.drop(dropLst, axis=1)
    dowDict = {'Thursday':4, 'Friday':5, 'Wednesday':3,
                'Monday':1, 'Sunday':0,'Saturday':6, 'Tuesday':2}
    sfdata['DayOfWeek'] =sfdata['DayOfWeek'].apply(lambda x: dowDict[x]) 
    sfdata.to_csv('data/sfpd_clean.csv')
    return sfdata


def clean_ppd_data(filename):
    ppdDict = {1: ['Thefts', 'Theft from Vehicle'], 
               2: ['Robbery No Firearm', 'Robbery Firearm'],
               3: ['Narcotic / Drug Law Violations',
                   'Liquor Law Violations', 'Public Drunkenness'],
               4: ['Aggravated Assault Firearm', 'Aggravated Assault No Firearm'],
               5: ['Motor Vehicle Theft'],
               6: ['Vandalism/Criminal Mischief'],
               7: ['Burglary Non-Residential', 'Burglary Residential'],
               8: ['Homicide - Criminal', 'Homicide - Gross Negligence',
                   'Homicide - Justifiable'],
               9: ['Rape', 'Prostitution and Commercialized Vice'],
               10: ['DRIVING UNDER THE INFLUENCE']}

    
def clean_ls_data(filename):
    lvDict = {1: ['THEFT/LARCENY', 'VEHICLE BREAK-IN/THEFT'],
              2: ['ROBBERY'],
              3: ['DRUGS/ALCOHOL VIOLATIONS'],
              4: ['ASSAULT'],
              5: ['MOTOR VEHICLE THEFT'],
              6: ['VANDALISM'],
              7: ['BURGLARY'],
              8: ['HOMICIDE'],
              9: ['SEX CRIMES'],
              10: ['DUI']}


def clean_la_data(filename1, filename2):
    ladict = {1: ['LARCENY THEFT'],
              2: ['ROBBERY', ],
              3: ['DRUNK / ALCOHOL / DRUGS', 'NARCOTICS', 'LIQUOR LAWS'],
              4: ['AGGRAVATED ASSAULT', 'NON-AGGRAVATED ASSAULTS'],
              5: ['GRAND THEFT AUTO'],
              6: ['VANDALISM'],
              7: ['BURGLARY'],
              8: ['CRIMINAL HOMICIDE'],
              9: ['SEX OFFENSES FELONIES', 'SEX OFFENSES MISDEMEANORS',
                  'FORCIBLE RAPE'],
              10: ['DRUNK DRIVING VEHICLE / BOAT']}

def clean_detroit_data(filename):
    dpdDict = {1: ['LARCENY'],
               2: ['ROBBERY'],
               3: ['DANGEROUS DRUGS', 'DRUNKENNESS', 'LIQUOR'],
               4: ['ASSAULT', 'AGGRAVATED ASSAULT'],
               5: ['STOLEN VEHICLE'],
               6: ['DAMAGE TO PROPERTY'],
               7: ['OTHER BURGLARY', 'BURGLARY'],
               8: ['HOMICIDE', 'JUSTIFIABLE HOMICIDE', 'NEGLIGENT HOMICIDE'],
               9: ['OBSCENITY'],
               10: ['OUIL']}


def clean_nyc_data(filename):
    nycDict = {1: ['GRAND LARCENY'],
               2: ['ROBBERY'],
               4: ['FELONY ASSAULT'],
               5: ['GRAND LARCENY OF MOTOR VEHICLE'],
               7: ['BURGLARY'],
               8: ['MURDER & NON-NEGL. MANSLAUGHTE'],
               9: ['RAPE']}


if __name__ == '__main__':
    sffile = 'data/SFPD.csv'
    lafile1 = 'data/LA_SHERIFF_1.csv'
    lafile2 = 'data/LA_SHERIFF_2.csv'
    sfdata = clean_sf_data(sffile)

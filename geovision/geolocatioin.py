#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 22:43:48 2020

@author: cinema
"""

import numpy as np
import pandas as pd
import json
import urllib.request
import requests
import haversine as hs

data = pd.read_csv('/Users/cinema/Desktop/Projects/llv/loclifetimevalue/python/worldcities.csv')

lng = 0
lat = 0
query = 0

data = data.drop(['capital', 'city_ascii', 'iso2', 'iso3', 'id', 'admin_name'], axis=1)
city = 'Jakarta'
data_lng = data.query("city == @city")['lng']
data_lat = data.query("city == @city")['lat']
data_lng = data_lng.reset_index(drop=True)
data_lat = data_lat.reset_index(drop=True)


industry = 'education'    
#data_lng = data['lng']
#data_lat = data['lat']
cluster_size = 25
url_3 = 'https://discover.search.hereapi.com/v1/discover?at=52.5228,13.4124&q=telecommunications&limit=25&apikey=7uXzaQsBY2_eFvWyfptFUrMsjzBcVP4nlYc-Udzl2TA'

#url = 'https://discover.search.hereapi.com/v1/discover?at=' + str(data_lat[0]) +','+ str(data_lng[0]) + '&q=' + str(industry) + '&limit=15&apikey=7uXzaQsBY2_eFvWyfptFUrMsjzBcVP4nlYc-Udzl2TA'
url = 'https://discover.search.hereapi.com/v1/discover?at=' + str(data_lat[0]) +','+ str(data_lng[0]) + '&q=' + str(industry) + '&limit=' + str(cluster_size) + '&apikey=7uXzaQsBY2_eFvWyfptFUrMsjzBcVP4nlYc-Udzl2TA'
#https://discover.search.hereapi.com/v1/discover?at=52.5228,13.4124&q=petrol+station&limit=5
url_2 = 'https://geocode.search.hereapi.com/v1/geocode?q=5+Rue+Daunou%2C+75000+Paris%2C+France/Authorization:Bearer[7uXzaQsBY2_eFvWyfptFUrMsjzBcVP4nlYc-Udzl2TA]'
geodata = requests.get(url).json()

gr = [d for d in geodata['items']]
title = pd.DataFrame(gr)
tile = title['position']
#tile = tile.tolist()
#tile_lat = [tile['lat'] for x in tile]
#lat = [d.get('lat') for d in tile]
#tile_lar = tile_lat['lat']
#lengthgr = len(gr)

def getHaversineDistance(x_long, x_lat, y_long, y_lat):
    return hs.haversine((x_long, x_lat),(y_long, y_lat))


def process_pandemic_data(new_df):

    # Columns renaming
    #df.columns = [col.lower() for col in df.columns]
    df = pd.DataFrame([])
    # Create a zone per zone/subzone
    #df['country'] = df['country'].apply(str) + ' ' + df['sub zone'].apply(lambda x: str(x).replace('nan', ''))
    new_df = new_df.drop(['id','resultType','categories','access'], axis=1)
    # Extracting latitute and longitude
    df['lat'] = [d.get('lat') for d in new_df['position']]
    df['lng'] = [d.get('lng') for d in new_df['position']]
    df['district'] = [d.get('district') for d in new_df['address']]
    df['city'] = [d.get('city') for d in new_df['address']]
    df['postal'] = [d.get('postalCode') for d in new_df['address']]
    df['title'] = new_df['title']
    lat_long = np.stack([df['lat'], df['lng']], axis=1)
    #for x in lat_long:
    #    for y in lat_long:
    #        getHaversineDistance(x[0], x[0], y[0], y[0])
            
    # Saving countries positions (latitude and longitude per subzones)
    positions = df[['title','city','district','postal','lat', 'lng']]#.drop_duplicates(['zone']).set_index(['zone'])

    # Pivoting per category
    #df = pd.pivot_table(df, values='count', index=['date', 'zone'], columns=['category'])
    #df.columns = ['confirmed', 'deaths', 'recovered']

    # Merging locations after pivoting
    #df = df.join(country_position)

    # Filling nan values with 0
    #df = df.fillna(0)

    # Compute bubble sizes
    df['size'] = new_df['distance'].apply(lambda x: (np.sqrt(x/100) + 1) if x > 500 else (np.log(x) / 2 + 1)).replace(np.NINF, 0)
    
    # Compute bubble color
    df['color'] = new_df['distance'].fillna(0).replace(np.inf , 0)
    
    return df


ge = process_pandemic_data(title)


from math import cos, sin, atan2, sqrt, radians, degrees
ge = pd.concat([ge['lat'],ge['lng']], axis=1)
def center_geolocation(geolocations):
    """
    Provide a relatively accurate center lat, lon returned as a list pair, given
    a list of list pairs.
    ex: in: geolocations = ((lat1,lon1), (lat2,lon2),)
        out: (center_lat, center_lon)
    """
    x = 0
    y = 0
    z = 0
    lat = []
    long = []
    

    for lat, lon in geolocations:
        lat = radians(lat)
        lon = radians(lon)
        x += cos(lat) * cos(lon)
        y += cos(lat) * sin(lon)
        z += sin(lat)

    x = degrees(float(x / len(geolocations)))
    y = degrees(float(y / len(geolocations)))
    z = degrees(float(z / len(geolocations)))
    
    return (atan2(z, sqrt(x * x + y * y)), atan2(y, x))

L = [
     (-74.2813611,40.8752222),
     (-73.4134167,40.7287778),
     (-74.3145014,40.9475244),
     (-74.2445833,40.6174444),
     (-74.4148889,40.7993333),
     (-73.7789256,40.6397511)
    ]
import decimal as Decimal
def find_center(L):
    lat = []
    long = []
    lat = [d for d in L['lat']]
    lng = [d for d in L['lng']]
    return sum(lat)/len(lat) , sum(lng)/len(lng)

    #df = pd.concat([pd.Series(sum(lat)/len(lat)),pd.Series(sum(long)/len(long))], axis=1)
    #print(pd.DataFrame(df,columns=['lat','lng']))
    
lat, lng = find_center(ge)
#geo = center_geolocation(title['position'])

from math import radians, cos, sin, asin, sqrt

#def haversine(clat, clng, df):
def haversine(L):
    #calculate great circle distance between 2 points on earth specified in decimal degrees. 
    clat, clng = find_center(ge)
    #clat = map(radians, [la])
    #clng = map(radians, [ln])
    #lat = map(radians, [d for d in L['lat']])
    #lng = map(radians,[d for d in L['lng']])
   
    lat = [d for d in L['lat']]
    lng = [d for d in L['lng']]
   
    dist_lat = radians(lat - [clat])
    dist_lng = radians(clng - [lng])
    a = sin(dist_lat/2)**2 + cos(clat) * cos(lat) * sin (dist_lng/2)**2
    c = 2 * asin(sqrt(a))
    eradius = 3959.87433
    print(eradius * c)



clat, clng = find_center(ge)
#clat = map(radians, [a])
#clng = map(radians, [ln])
#lat = map(radians, [d for d in L['lat']])
#lng = map(radians,[d for d in L['lng']])
   
lat = [d for d in ge['lat']]
lng = [d for d in ge['lng']]
   
dist_lat = np.radians([clat - x for x in lat])
dist_lng = np.radians([clng - x for x in lng])
sin_dlat = [sin(x/2)**2 for x in dist_lat]
sin_dlng = [sin(x/2)**2 for x in dist_lng]
#a = [sin(dist_lat/2)**2 + cos(clat) * cos(lat) * sin (dist_lng/2)**2
a = [x + cos(clat) * cos(z) * y for x , y, z in zip(sin_dlat, sin_dlng,lat)]
c = [2 * asin(sqrt(x)) for x in a]
eradius = 3959.87433
#print([eradius * x for x in c])    


#distance = haversine(ge)
   
'''
Calculate distance using the Haversine Formula
'''

def hanversine(coord1: object, coord2: object):
    import math

    # Coordinates in decimal degrees (e.g. 2.89078, 12.79797)
    lon1, lat1 = coord1
    lon2, lat2 = coord2
    #my addition
    clat, clng = find_center(ge)
    lat = [d for d in ge['lat']]
    lng = [d for d in ge['lng']]

    
    R = 6371000  # radius of Earth in meters
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)
    
    #my addition
    phi_i = [math.radians(x) for x in lat]
    phi_j = clat
    
    
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    #my addition
    delta_ph = [math.radians(x - clat) for x in lat]
    delta_lamb = [math.radians(y - clng) for y in lng]
    
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2.0) ** 2
    
    #my addition
    #df = pd.concat([delta_ph,lat,delta_lamb], axis=1)
    b = [math.sin(x/2.0)**2 + math.cos(y) * math.cos(phi_j) * math.sin(z/2.0)**2 for x,y,z in zip(delta_ph, lat, delta_lamb)]
    
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    #my addition
    cee = [2 * math.atan2(math.sqrt(x), math.sqrt(1-x)) for x in b]
    
    
    meters = [ R * x for x in cee]  # output distance in meters
    km = [x / 1000.0 for x in meters]  # output distance in kilometers
    
    mrs = R * c
    krm = mrs / 1000.0
    
    mrs = round(mrs, 3)
    krm = round(krm, 3)


    #print(f"Distance: {meters} m")
    #print(f"Distance: {km} km")    
    #print(f"Distance: {mrs} mrs")
    #print(f"Distance: {krm} krm")
    df = pd.DataFrame(np.column_stack([meters, km]), columns=['meters', 'km'])
    return df

#print("Value======== ", hanversine([-0.116773, 51.510357], [-77.009003, 38.889931]))

cord = hanversine([-0.116773, 51.510357], [-77.009003, 38.889931])
print(cord)

#cordae = hanversine([-6.21448, 106.84519], [-6.21391,106.84636])








'''


def process_pandemic_data(df):

    # Columns renaming
    df.columns = [col.lower() for col in df.columns]

    # Create a zone per zone/subzone
    df['zone'] = df['zone'].apply(str) + ' ' + df['sub zone'].apply(lambda x: str(x).replace('nan', ''))
    
    # Extracting latitute and longitude
    df['lat'] = df['location'].apply(lambda x: x.split(',')[0])
    df['lon'] = df['location'].apply(lambda x: x.split(',')[1])

    # Saving countries positions (latitude and longitude per subzones)
    country_position = df[['zone', 'lat', 'lon']].drop_duplicates(['zone']).set_index(['zone'])

    # Pivoting per category
    df = pd.pivot_table(df, values='count', index=['date', 'zone'], columns=['category'])
    df.columns = ['confirmed', 'deaths', 'recovered']

    # Merging locations after pivoting
    df = df.join(country_position)

    # Filling nan values with 0
    df = df.fillna(0)

    # Compute bubble sizes
    df['size'] = df['confirmed'].apply(lambda x: (np.sqrt(x/100) + 1) if x > 500 else (np.log(x) / 2 + 1)).replace(np.NINF, 0)
    
    # Compute bubble color
    df['color'] = (df['recovered']/df['confirmed']).fillna(0).replace(np.inf , 0)
    
    return df


'''


'''
# Selecting the day to display
day = '2020-05-01'
tmp = df.xs(day)

# Create the figure and feed it all the prepared columns
fig = go.Figure(
    go.Scattermapbox(
        lat=tmp['lat'],
        lon=tmp['lon'],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=tmp['size'],
            color=tmp['color']
        )
    )
)

# Specify layout information
fig.update_layout(
    mapbox=dict(
        accesstoken=mapbox_access_token, #
        center=go.layout.mapbox.Center(lat=45, lon=-73),
        zoom=1
    )
)

# Display the figure
fig.show()

'''





#cities = pd.read_csv('python/worldcities.csv')#.to_dict()
city_names = data['city']
#cities = cities.drop(['city_ascii','country','iso3','iso2','admin_name', 'id'], axis=1)
#"city","city_ascii","lat","lng","country","iso2","iso3","admin_name","capital","population","id"
list_city_names = len(city_names)
city_val = dict(zip(city_names, city_names))
city_keys = list(city_val.keys())
nested_city = city_val[city_keys[0]]
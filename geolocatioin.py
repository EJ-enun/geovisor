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

data = pd.read_csv('/Users/cinema/Desktop/Projects/llv/loclifetimevalue/python/worldcities.csv')

lng = 0
lat = 0
query = 0

data = data.drop(['capital', 'city_ascii', 'iso2', 'iso3', 'id', 'admin_name'], axis=1)
city = 'Mumbai'
data_lng = data.query("city == @city")['lng']
data_lat = data.query("city == @city")['lat']
data_lng = data_lng.reset_index(drop=True)
data_lat = data_lat.reset_index(drop=True)


industry = 'tourism'    
#data_lng = data['lng']
#data_lat = data['lat']
cluster_size = 5
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

#cities = pd.read_csv('python/worldcities.csv')#.to_dict()
city_names = data['city']
#cities = cities.drop(['city_ascii','country','iso3','iso2','admin_name', 'id'], axis=1)
#"city","city_ascii","lat","lng","country","iso2","iso3","admin_name","capital","population","id"
list_city_names = len(city_names)
city_val = dict(zip(city_names, city_names))
city_keys = list(city_val.keys())
nested_city = city_val[city_keys[0]]
        

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
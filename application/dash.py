###############################################################################
#                                MAIN                                         #
###############################################################################

# Setup
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from shapely.geometry import Point, Polygon
import geopandas as gpd
import geojsonio
import dash_leaflet as dl
import numpy as np
import pandas as pd
import plotly.express as px
from settings import config, about
import plotly.graph_objs as go
import urllib.request
import requests
#from python.data import Data
#from python.industry_struct import data_struct
#from python.model import Model
#from python.result import Result



# Read data
#data = data_struct()
#data.get_data()



# App Instance
app = dash.Dash(name=config.name, assets_folder=config.root+"/application/static", external_stylesheets=[dbc.themes.LUX, config.fontawesome])
app.title = config.name

def getCity():
        cities = pd.read_csv('python/worldcities.csv')#.to_dict()
        city_names = cities['city']
        cities = cities.drop(['city_ascii','country','iso3','iso2','admin_name', 'id'], axis=1)
        #"city","city_ascii","lat","lng","country","iso2","iso3","admin_name","capital","population","id"
        list_city_names = len(city_names)
        city_val = dict(zip(city_names, city_names))
        city_keys = list(city_val.keys())
        return city_val
    
def getCityDict():
        cities = pd.read_csv('python/worldcities.csv')#.to_dict()
        city_names = cities['city']
        cities = cities.drop(['city_ascii','country','iso3','iso2','admin_name', 'id'], axis=1)
        #"city","city_ascii","lat","lng","country","iso2","iso3","admin_name","capital","population","id"
        list_city_names = len(city_names)
        city_val = dict(zip(city_names, city_names))
        city_keys = list(city_val.keys())
        nested_city = city_val[city_keys[0]]
        return city_val

def getIndustry():
        #industries = {'agriculture':'Agriculture','basic metal production':'Basic Metal Production','chemicals':'Chemical','commerce':'Commerce','construction':'Construction','education':'Education','manufacturing':'Equipment Manufacturing','financial services':'Financial Services','food':'Food, Drinks, Fobacco','forestry':'Forestry','health':'Health Services', 'tourism':'Tourism','mining':'Mining','mechanical':'Mechanical and Electrical Engineering','media':'Media and Culture','oil':'Oil and Gas','postal':'Postal and Telecommunications Services','public service':'Public Service','shipping':'Shipping, Ports, Fisheries','textiles':'Textiles','transport':'Transport','utilities':'Utilities'}  
        industries = {'Agriculture':'agriculture','Basic Metal Production':'basic metal production','Chemicals':'chemical','Commerce':'commerce','Construction':'Construction','Education':'Education','Equipment Manufacturing':'Manufacturing','Financial services':'Financial Services','Food, Drinks, Tobacco':'Food, Drinks, Tobacco','Forestry':'Forestry','Health Services':'Health Services', 'Tourism':'Tourism','Mining':'Mining','Mechanical and Electrical Engineering':'Mechanical and Electrical Engineering','Media and Culture':'Media','Oil and Gas':'Oil and Gas','Postal and Telecommunications Services':'Postal and Telecommunications Services','public service':'Public Service','shipping':'Shipping, Ports, Fisheries','textiles':'Textiles','transport':'Transport','utilities':'Utilities'} 
        ind_val = list(industries.keys())
        return ind_val

def getIndustryDict():
        industries = {'Agriculture':'agriculture','Basic Metal Production':'basic metal production','Chemicals':'chemical','Commerce':'commerce','Construction':'Construction','Education':'Education','Equipment Manufacturing':'Manufacturing','Financial services':'Financial Services','Food, Drinks, Tobacco':'Food, Drinks, Tobacco','Forestry':'Forestry','Health Services':'Health Services', 'Tourism':'Tourism','Mining':'Mining','Mechanical and Electrical Engineering':'Mechanical and Electrical Engineering','Media and Culture':'Media','Oil and Gas':'Oil and Gas','Postal and Telecommunications Services':'Postal and Telecommunications Services','public service':'Public Service','shipping':'Shipping, Ports, Fisheries','textiles':'Textiles','transport':'Transport','utilities':'Utilities'} 
        ind_val = list(industries.keys())
        nested_ind = industries[ind_val[0]]
        return nested_ind




def api_call_cities(city, industry, cluster_size):
    cities = pd.read_csv('python/worldcities.csv')#.to_dict()
    cities = cities.drop(['capital', 'city_ascii', 'iso2', 'iso3', 'id', 'admin_name'], axis=1)
    data_lng = cities.query("city == @city")['lng']
    data_lat = cities.query("city == @city")['lat']
    data_pop = cities.query("city == @city")['population']
    data_lng = data_lng.reset_index(drop=True)
    data_lat = data_lat.reset_index(drop=True)
 
     
    url_3 = 'https://discover.search.hereapi.com/v1/discover?at=52.5228,13.4124&q=telecommunications&limit=25&apikey=7uXzaQsBY2_eFvWyfptFUrMsjzBcVP4nlYc-Udzl2TA'

    url = 'https://discover.search.hereapi.com/v1/discover?at=' + str(data_lat[0]) +','+ str(data_lng[0]) + '&q=' + str(industry) + '&limit=' + str(cluster_size) + '&apikey=7uXzaQsBY2_eFvWyfptFUrMsjzBcVP4nlYc-Udzl2TA'

    #https://discover.search.hereapi.com/v1/discover?at=52.5228,13.4124&q=petrol+station&limit=5
    url_2 = 'https://geocode.search.hereapi.com/v1/geocode?q=5+Rue+Daunou%2C+75000+Paris%2C+France/Authorization:Bearer[7uXzaQsBY2_eFvWyfptFUrMsjzBcVP4nlYc-Udzl2TA]'
    geodata = requests.get(url).json()

    gr = [d for d in geodata['items']]
    title = pd.DataFrame(gr)
    tile = title['position']
    return title, data_pop


def process_data(new_df):

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
#ttps://discover.search.hereapi.com/v1/discover?at=52.5228,13.4124&q=petrol+station&limit=5
# Navbar
navbar = dbc.Nav(className="nav nav-pills", children=[
    ## logo/home
    dbc.NavItem(html.Img(src=app.get_asset_url("logo.jpeg"), height="40px")),
    ## about
    dbc.NavItem(html.Div([
        dbc.NavLink("About", href="/", id="about-popover", active=False),
        dbc.Popover(id="about", is_open=False, target="about-popover", children=[
            dbc.PopoverHeader("How it works"), dbc.PopoverBody(about.txt)
        ])
    ])),
    ## links
    dbc.DropdownMenu(label="Links", nav=True, children=[
        dbc.DropdownMenuItem([html.I(className="fa fa-linkedin"), "  Contacts"], href=config.contacts, target="_blank"), 
        dbc.DropdownMenuItem([html.I(className="fa fa-github"), "  Code"], href=config.code, target="_blank")
    ])
])



# Input
city = dbc.FormGroup([
    html.H4("Select City"),
    dcc.Dropdown(id="city",options=[{"label":x,"value":x} for x in getCityDict()], value='Delhi')
]) 

industry = dbc.FormGroup([
    html.H4("Select Industry"),
    dcc.Dropdown(id="industry", options=[{'label':x,'value':x} for x in getIndustry()], value='Agriculture')
])


cluster_size = dbc.FormGroup([
    html.H4("Cluster Size"),
    dcc.Dropdown(id="cluster_size", options=[{"label":x,"value":x} for x in {'5': 5, '10' : 10, '15':15, '20' : 20 ,'25' : 25, '30' : 30, '35':35, '40':40}], value='15')
])

search_term = dcc.Input(id="input1", type="text", placeholder="")

input_map = dl.Map(dl.TileLayer(), style={'width': '1000px', 'height': '500px'})


def getMapBox():
    cities = pd.read_csv('python/worldcities.csv')#.to_dict()
    city_names = cities['city']
    cities = cities.drop(['city_ascii','country','iso3','iso2','admin_name', 'id'], axis=1)        
    fig = px.scatter_mapbox(cities, lat="lat", lon="lng", hover_name="city", hover_data=["capital", "population"],color_discrete_sequence=["fuchsia"], zoom=3, height=300)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig.show() 






# App Layout
app.layout = dbc.Container(fluid=True, children=[
    ## Top
    html.H1(config.name, id="nav-pills"),
    navbar,
    html.Br(),html.Br(),html.Br(),
    ## Body
    dbc.Row([
        ### input + panel
        dbc.Card(    
            dbc.Col(md=12, children=[
            html.Br(),html.Br(),html.Br(),
            city, 
            #html.Br(),html.Br(),html.Br(),
            industry,
            #html.Div(id="output-panel")
            cluster_size,
            html.Br(),
            html.Button('Visualize', id='visualize', n_clicks=0),
            html.Br(),html.Br(),
        ]),style={'height':'100vh'}),
        ### plots
        dbc.Col(children=[
            dbc.Card(dbc.Col(children=[dcc.Graph(id='map'), html.H1(id='output-panel')]), style={'height':'100vh'}),
            ])
        ])
    ])
    
#])

#[Output('map', 'figure'),Output('map', 'config'),]
@app.callback([Output('map', 'figure'),Output('map', 'config')],[Input('visualize', 'n_clicks')],[State('city','value'),State('industry','value'),State('cluster_size','value')])
def update_map_callback(n_clicks, city, industry, cluster_size):
    data, pop = api_call_cities(city, industry, cluster_size)
    tmp = process_data(data)
    #tmp = [processed_data
    map_figure = {
        'data': [
            go.Scattermapbox(
                lat=tmp['lat'],
                lon=tmp['lng'], 
                mode='markers',
                marker=go.scattermapbox.Marker(size=tmp['size'],color=tmp['color'],showscale=True,colorbar={'title':'Distance From Cluster Center', 'titleside':'top', 'thickness':4, 'ticksuffix':' %'},)
            )
        ],
        'layout': go.Layout(
            autosize=True,
            hovermode='closest',
            mapbox=dict(
                accesstoken='pk.eyJ1IjoidG9kZGthcmluIiwiYSI6Ik1aSndibmcifQ.hwkbjcZevafx2ApulodXaw',
                center=dict(
                    lat=45,
                    lon=-73
                ),
                zoom=1
            )
        )}

    map_config = dict(scrollZoom = True)
    
    return map_figure, map_config
    #return map_figure, map_config #"Value:{}, :{}".format(city, industry)  #
# Python functions for about navitem-popover
@app.callback(output=Output("about","is_open"), inputs=[Input("about-popover","n_clicks")], state=[State("about","is_open")])
def about_popover(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(output=Output("about-popover","active"), inputs=[Input("about-popover","n_clicks")], state=[State("about-popover","active")])
def about_active(n, active):
    if n:
        return not active
    return active

#python function to render output panel
@app.callback(Output("output-panel","children"),[Input("visualize","n_clicks")],[State("city","value"),State("industry","value"), State("cluster_size","value")])
def render_output_panel(n_clicks, city, industry, cluster_size):
    data, pop = api_call_cities(city, industry, cluster_size)
    panel = html.Div([
        html.H4('Key Metrics'),    
        dbc.Card(body=True, className="text-white bg-primary", children=[
            
            html.H6("City:", style={"color":"white"}),
            html.H3("{}".format(city), style={"color":"white"}),
            
            html.H6("Industry:", className="text-danger"),
            html.H3("{}".format(industry), className="text-danger"),
            
            html.H6("Population:", style={"color":"white"}),
            html.H3("{}".format(pop), style={"color":"white"}),
            
            html.H6("Population as a percentage of each industry:", className="text-danger"),
            html.H3("{}".format(pd.to_numeric(pop) / int(cluster_size)), className="text-danger"),
            
        ])
    ])
    return panel
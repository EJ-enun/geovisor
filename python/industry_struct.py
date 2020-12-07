import numpy as np
import pandas as pd


class data_struct:
	#Convert city, industries and sectors to dict dtypes.  
	def __init__(self):

	def getCity():
		cities = pd.read_csv('python/worldcities.csv')#.to_dict()
		city_names = cities['city']
		list_city_names = len(city_names)
		city_val = dict(zip(range(0,list_city_names), city_names))
		city_val = list(city_val.keys())
		nested_city = cities[city_val[0]]
		return city_val

	def getIndustry():
		#industries = {'Agriculture':'agriculture','Basic Metal Production':'basic metal production','Chemicals':'chemical','Commerce':'commerce','Construction':'Construction','Education':'Education','Equipment Manufacturing':'Manufacturing','Financial services':'Financial Services','Food, Drinks, Tobacco':'Food, Drinks, Tobacco','Forestry':'Forestry','Health Services':'Health Services', 'Tourism':'Tourism','Mining':'Mining','Mechanical and Electrical Engineering':'Mechanical and Electrical Engineering','Media and Culture':'Media','Oil and Gas':'Oil and Gas','Postal and Telecommunications Services':'Postal and Telecommunications Services','public service':'Public Service','shipping':'Shipping, Ports, Fisheries','textiles':'Textiles','transport':'Transport','utilities':'Utilities'}	
		industries = {'agriculture':'Agriculture','basic metal production':'Basic Metal Production'}
		ind_val = list(industries.keys())
		nested_ind = industries[ind_val]
		return industries

	def getSector():


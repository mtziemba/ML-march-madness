#!/usr/bin/env python
import math
import csv
from StringIO import StringIO
import numpy as np
import pandas as pd
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest

teams_pd = pd.read_csv('data/Teams.csv')
gamelist = []

__author__ = "Mary Ziemba and Alex Deckey, based on code by Mark Nemececk for COMPSCI 270, Spring 2017, Duke University"
__copyright__ = "Mary Ziemba and Alex Deckey"
__credits__ = ["Mary Ziemba", "Alex Deckey", "David Duquette",
                    "Camila Vargas Restrepo", "Melanie Krassel"]
__license__ = "Creative Commons Attribution-NonCommercial 4.0 International License"
__version__ = "1.0.0"
__email__ = "mtz3@duke.edu"

def getTeamName(team_id):
    return teams_pd[teams_pd['Team_Id'] == team_id].values[0][1]

def get_data(years,custom=False):
	'''
	Create train and test dataset for classifiers.
	''' 
	matchups_headers, matchups = get_matchup_data()
	curr_matchups = index_matchups(matchups,years)

	if custom:
		data, target = load_feature_vectors_and_outcomes(years)
		return data, target, curr_matchups
	data, target = load_feature_vectors_and_outcomes(years)
 	return split_dataset(data, target, train_size=0.75)

def index_matchups(matchups,years):
	'''
	Create a lookup value to determine which games were correctly/incorrectly classified.
	'''
	games = list()
	for matchup in matchups:
		if matchup[1] in years and matchup[6] < matchup[8]:
			games.append((matchup[0],matchup[5], getTeamName(matchup[6]),matchup[7], getTeamName(matchup[8]), matchup[10]))
	return games

def get_peripheral_game_data():
	reload_file('data/Tournament_Game_Data_Lat_Long.csv','results')
	tourney_results_data = np.load('data/results.npy')
	seed_0 = tourney_results_data['Seed0']
	seed_1 = tourney_results_data['Seed1']
	t_round = tourney_results_data['Round']
	lat = tourney_results_data['Latitude']
	longit = tourney_results_data['Longitude']
	data = np.column_stack((seed_0,seed_1,t_round,lat,longit))
	headers = ('Seed0','Seed1','Round','Latitude','Longitude')
	return headers, data

def reload_file(f, fname):
	'''
	Creates a .npy file of the data.
	* f is a csv
	* fname is what the filename will be, no need to include the leading "data/".
	'''
 	my_file = open(f,'rU')
 	if ' ' in fname:
 		fname = '_'.join(fname.strip().split())
	data = np.genfromtxt(my_file,delimiter=',',names=True,dtype=None)
	np.save('data/' + fname + '.npy',data)
	return data

def get_matchup_data():
	'''
	Takes the numpy representation of `Tournament_Game_Data_Lat_Long.csv` and 
	puts them in a form that is usable by the get_labels and 
	get_difference_vectors functions
	'''
	# Uncomment this line if we made changes to the kenpom file and need to reload it
	reload_file('data/Tournament_Game_Data_Lat_Long.csv','results')
	tourney_results_data = np.load('data/results.npy')
	headers = tourney_results_data.dtype.names
	return (headers, tourney_results_data)

def get_kenpom_data():
	# # Uncomment this line if we made changes to the kenpom file and need to reload it
	# reload_file('data/Kenpom/kenpom.csv','kenpom')
	kenpom_data = np.load('data/kenpom.npy')
	headers = kenpom_data.dtype.names
	return (headers, kenpom_data)

def get_labels(years):
	'''
	Create the "labels" for the games, which is whether team0
	 or team1 wins. Takes an argument `years` to get only the 
	games from a range of years.
	'''
	matchups_labels, matchups = get_matchup_data()
	temp_labels = list()
	for game in matchups:
		if game['School0'] < game['School1'] and game['Year'] in years:	# The file has both:
			result = 1 if game['Diff'] < 1 else 0 		#   2015,1458,1181,-5 and
			temp_labels.append(result)		 		#   2015,1181,1458,5
	labels = np.array(temp_labels)			 		# and we only want one of them.
	return labels 							 	# So we say if team0 < team1 (arbitrarily)

def get_player_data():
	reload_file('data/class_years.csv','class_years')
	data = np.load('data/class_years.npy')
	headers = data.dtype.names
	return (headers, data)

def get_reg_season_data():
	# # Uncomment this line if we made changes to the kenpom file and need to reload it
	# reload_file('data/Kenpom/kenpom.csv','kenpom')
	regular_season_data = np.load('data/regular_season.npy')
	headers = regular_season_data.dtype.names
	return (headers, regular_season_data)
	
def get_school_location_data():
	reload_file('data/school_locations.csv','school location lookup')
	data = np.load('data/school_location_lookup.npy')
	headers = data.dtype.names
	return (headers, data)

def calc_distance(team_lat,team_long,game_lat,game_long):
	xx = (team_lat - game_lat) ** 2
	yy = (team_long - game_long) ** 2
	return math.sqrt(xx + yy)

def get_weighted_seed(seed, rd):
	# return seed
	if rd == 'First Round':
		return 0.9 * seed
	elif rd == 'Second Round':
		return 0.7 * seed
	elif rd == 'Regional Semifinal' or rd == 'Third Round':
		return 0.5 * seed
	elif rd == 'Regional Final':
		return 0.3 * seed
	elif rd == 'National Semifinal':
		return 0.15 * seed
	elif rd == 'National Final':
		return 0.1 * seed
	elif rd == 'Opening Round' or rd == 'First Four':
		return 0
	raise ValueError('Invalid round')

def get_conference_data():
	reload_file('data/Conference.csv','conference')
	data = np.load('data/conference.npy')
	headers = data.dtype.names
	return (headers, data)

def make_team_vector(team,year,kp,team_info_d,cci,seed,rd,game,loc_lookup):
	l = list()
	l.extend([kp.get((team, year)).get('Win_Pctg'),
			  kp.get((team, year)).get('Kenpom_AdjEM'),
			  kp.get((team, year)).get('Kenpom_SOS'),
			  team_info_d.get((team,year))[1],    # McDonalds all-Americans
			  team_info_d.get((team,year))[0]])   # Upperclassmen %age
	conf_champ = 1 if team in cci[int(year)] else 0
	l.append(conf_champ)
	l.append(get_weighted_seed(int(seed), str(rd)))
	(school_lat,school_long) = loc_lookup.get(team)
	l.append(calc_distance(school_lat,school_long,float(game['Latitude']),float(game['Longitude'])))
	return l

# USE THIS
def getSeasonData(team_id, year):
    year_data_pd = reg_season_compact_pd[reg_season_compact_pd['Season'] == year]
    gamesWon = year_data_pd[year_data_pd.Wteam == team_id] 
    totalPointsScored = gamesWon['Wscore'].sum()
    gamesLost = year_data_pd[year_data_pd.Lteam == team_id] 
    totalGames = gamesWon.append(gamesLost)
    numGames = len(totalGames.index)
    totalPointsScored += gamesLost['Lscore'].sum()

def get_difference_vectors(years):
	'''
	Create the difference vectors that correspond to the games, 
	referenced in get_labels. Takes an argument `years` to get
	 only the games from a range of years.
	'''	
	# Loading numpy data structures to sew the different columns together
	matchups_headers, matchups = get_matchup_data()
	kenpom_headers, kenpom_data = get_kenpom_data()
	periphs_headers, peripherals = get_peripheral_game_data()
	school_loc_headers, school_locations = get_school_location_data()
	conf_headers, conference_data = get_conference_data()
	player_headers, players_data = get_player_data()
	# reg_seas_headers, regular_season_data = get_reg_season_data()
	
	# Prepare Kenpom data for use, choose variables
	kenpom = dict()
	for row in kenpom_data:
		team_id = int(row['team_id'])
		year = int(row['Year'])
		info = {'Win_Pctg': float(row['WL']),
				'Kenpom_AdjEM': float(row['AdjEM']),
				'Kenpom_SOS': float(row['SOS_AdjEM'])}
		kenpom[(team_id, year)] = info
	
	# Pick conference champ variables
	conference_champ_info = dict()
	for row in conference_data:
		team_id = int(row[14])
		year = int(row['Year'])
		if year not in conference_champ_info.keys():
			conference_champ_info[year] = list()
		conference_champ_info[year].append(team_id)
	
	# Get team info (right now, pct juniors and seniors)
	team_info = dict()
	for row in players_data:
		team_id = int(row[1])
		year = int(row[2])
		# pct_upper = float(row[5] + row[6]) / sum([float(row[i]) for i in range(3,7)])
		pct_upper = float(row[6]) / sum([float(row[i]) for i in range(3,7)])
		num_mcDonalds = int(row[9])
		team_info[(team_id, year)] = (pct_upper, num_mcDonalds)

	# School location data
	school_loc_lookup = dict()
	for school in school_locations:
		school_id = int(school[6])
		school_lat = float(school[4])
		school_long = float(school[5])
		school_loc_lookup[school_id] = (school_lat,school_long)

	# Wins in last 10
	wins_in_last_ten = dict()

 	diff_vectors = list()

 	# Write difference vectors to csv
 	f = open('data/diff_vectors_latest.csv','w+')
 	f.truncate()
 	diff_file = csv.writer(f, delimiter=',',quotechar='"')

	count = 0
	for game in matchups:
		if game['School0'] < game['School1'] and int(game['Year']) in years:
			team0 = int(game['School0'])
			team1 = int(game['School1'])
# 			print("Game " + str(count) + " " + getTeamName(team0) + " vs " + getTeamName(team1))
# 			gamelist.append((count,getTeamName(team0),team0,getTeamName(team1),team1))
			game_year = game['Year']
			v0 = make_team_vector(team0,game_year,kenpom,team_info,conference_champ_info,int(game['Seed0']),str(game['Round']),game,school_loc_lookup)
			v1 = make_team_vector(team1,game_year,kenpom,team_info,conference_champ_info,int(game['Seed1']),str(game['Round']),game,school_loc_lookup)
			if team1 == 1458:
				print "Wisco: " + str(v1), game["Date"]
			diff = [x0 - x1 for x0, x1 in zip(v0, v1)]
			diff_file.writerow(diff)
			diff_vectors.append(diff)
			count += 1
	dataset = np.array([np.array(xi) for xi in diff_vectors],dtype=None)
	return dataset


def load_feature_vectors_and_outcomes(years):	
	# Our difference vectors
	data = get_difference_vectors(years)
	
	# Our labels for those games
	target = get_labels(years)
	return (data, target)

def split_dataset(data, target, train_size=0.8):
    '''
    Splits the provided data and targets into training and test sets
	split_dataset
    '''
    data_train, data_test, target_train, target_test = train_test_split(data, target, train_size=train_size, random_state=0)
    return data_train, data_test, target_train, target_test


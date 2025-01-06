# -*- coding: utf-8 -*-
"""

@author:

    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au

    Python script for Big Data Bowl 2025 entry calculating In Motion Index.
    This script uses the associated helperFuncs.py script for calculations and
    visualisations.

    Run on python version 3.10.14

"""

# =========================================================================
# Import packages
# =========================================================================

# Note that variable package versions may result in slightly different outcomes,
# or maybe not work at all!
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib import lines
from matplotlib.offsetbox import (AnnotationBbox, DrawingArea, OffsetImage, TextArea)
import nfl_data_py as nfl
import math
from PIL import Image, ImageChops, ImageDraw
import requests
import os
from glob import glob
from tqdm import tqdm
import pickle
import random
from itertools import groupby

# Import helper functions
from helperFuncs import createField, drawFrame, downloadTeamImages, downloadPlayerImages, cropPlayerImg, calcIMI, radar_factory

# =========================================================================
# Set-up
# =========================================================================

# Set a boolean value to re-analyse data or simply load from dictionary
# Default is False --- swap to True if wishing to re-run analysis
calcInMotionIndex = False

# Set a boolean value to reproduce visuals
# Default is False --- swap to True if wishing to re-create visuals
createVisuals = False

# Set a boolean value to summarise data
# Default is False --- swap to True if wishing to review summary data
summariseData = False

# Set weights for IMI components
weightsIMI = {'targeted': 0.20, 'reception': 0.20, # 'catchRate': 0.0,
              'yards': 0.20, 'yac': 0.15,
              'peakSeparation': 0.05, 'separationAtCatch': 0.10,
              'releaseSpeed': 0.10
              }

# Set matplotlib parameters
from matplotlib import rcParams
rcParams['font.sans-serif'] = 'Arial'
rcParams['font.weight'] = 'bold'
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 16
rcParams['axes.linewidth'] = 1.5
rcParams['axes.labelweight'] = 'bold'
rcParams['legend.fontsize'] = 10
rcParams['xtick.major.width'] = 1.5
rcParams['ytick.major.width'] = 1.5
rcParams['legend.framealpha'] = 0.0
rcParams['savefig.dpi'] = 300
rcParams['savefig.format'] = 'pdf'

# Add custom fonts for use with matplotlib
fontDir = [os.getcwd()+os.sep+os.path.join('..','fonts')]
for font in font_manager.findSystemFonts(fontDir):
    font_manager.fontManager.addfont(font)

# Import team details from nfl_data_py
teamData = nfl.import_team_desc()

# Import 2022 season roster details
rosterData = nfl.import_seasonal_rosters([2022])

# =========================================================================
# Download team images for visualisations
# =========================================================================

"""

Note that this doesn't need to be re-run more than once. It doesn't take long,
but it's not really necessary to do again. Change the flag if re-processing
images is desired.

"""

# Download team images flag
getTeamImages = False

# Check whether to download team images again
if getTeamImages:

    # Run function to download team images
    print('Downloading team logo images...')
    downloadTeamImages(teamData, os.path.join('..','img','team'))

# =========================================================================
# Read in data files
# =========================================================================

# Read in the games data
games = pd.read_csv(os.path.join('..','data','games.csv'))

# Read in play description data
plays = pd.read_csv(os.path.join('..','data','plays.csv'))

# Read in player specific play data
player_play = pd.read_csv(os.path.join('..','data','player_play.csv'))

# Load player data
players = pd.read_csv(os.path.join('..','data','players.csv'))

# Read in the tracking data across weeks
# Get tracking files
trackingFiles = glob(os.path.join('..','data','tracking*.csv'))
tracking = {}
# Loop through and load tracking files
print('Importing tracking data files...')
for fName in tqdm(trackingFiles):
    # Identify week number
    weekNo = int(fName.split('week_')[-1].split('.csv')[0])
    # Load tracking data
    tracking[f'week{weekNo}'] = pd.read_csv(fName)
    # Join player positioning information onto a week's worth of tracking data
    tracking[f'week{weekNo}'] = tracking[f'week{weekNo}'].merge(players.loc[:, ['nflId', 'position']], how='left')

# =========================================================================
# Identify route runners in dataset and static vs. motion routes
# =========================================================================

# Check for calculating
if calcInMotionIndex:

    # Set the minimum number of routes run to be included in the dataset
    # Note this is relatively arbitrary and could be changed to check different players
    minRoutes = 100

    # Set the list of route runners to collect
    routeRunners = []

    # Loop through unique player IDs
    for nflId in player_play['nflId'].unique():
        # Extract players data and sum the number of routes run
        # If it's greater than or equal to 50, keep that ID
        if np.nansum(player_play.loc[player_play['nflId'] == nflId]['wasRunningRoute']) > minRoutes:
            routeRunners.append(nflId)

    # Identify motion vs. stationary route plays for each route runner
    routeDataDict = {'nflId': [], 'playerName': [], 'teamName': [],
                     'motionRoutePlays': [], 'nMotionRoutes': [],
                     'stationRoutePlays': [], 'nStationRoutes': []}

    # Loop through players
    print('Extracting route data for route runners...')
    for nflId in tqdm(routeRunners):

        # Extract plays for current player
        playerPlays = player_play.loc[player_play['nflId'] == nflId]

        # Extract name and team for current player
        teamName = rosterData.loc[rosterData['gsis_it_id'] == str(nflId),]['team'].values[0]
        playerName = rosterData.loc[rosterData['gsis_it_id'] == str(nflId),]['player_name'].values[0]

        # Set a list to store motion vs. non-motion plays where a route was run by player
        motionRoutePlays = []
        stationRoutePlays = []

        # Loop through players plays and store the appropriate plays
        for gameId, playId, inMotion, routeRun in zip(
                playerPlays['gameId'], playerPlays['playId'],
                playerPlays['inMotionAtBallSnap'], playerPlays['wasRunningRoute']):

            # Check for passing play as we only want to include these
            # Do this by looking for pass event indicators in tracking data
            weekNo = games.loc[(games['gameId'] == gameId)]['week'].values[0]
            playEvents = tracking[f'week{weekNo}'].loc[
                (tracking[f'week{weekNo}']['gameId'] == gameId) &
                (tracking[f'week{weekNo}']['playId'] == playId)]['event'].unique().tolist()
            if any([str(eventName).startswith('pass_') for eventName in playEvents]):

                # Check for motion and route
                # Annoyingly these are in different formats (boolean vs. int)
                if inMotion == True and routeRun == 1.0:
                    # Store in motion route plays
                    motionRoutePlays.append((gameId, playId))
                elif not inMotion == True and routeRun == 1.0:
                    # Store in station route plays
                    stationRoutePlays.append((gameId, playId))

        # # Sanity checks
        # # Get a count of routes run
        # print(f'Number of routes run total: {int(np.nansum(playerPlays["wasRunningRoute"]))}')
        # # Get a count of the motion vs. stationary plays
        # print(f'Number of routes run from motion: {len(motionRoutePlays)}')
        # print(f'Number of routes run from stationary: {len(stationRoutePlays)}')

        # Add to dictionary
        routeDataDict['nflId'].append(nflId)
        routeDataDict['playerName'].append(playerName)
        routeDataDict['teamName'].append(teamName)
        routeDataDict['motionRoutePlays'].append(motionRoutePlays)
        routeDataDict['nMotionRoutes'].append(len(motionRoutePlays))
        routeDataDict['stationRoutePlays'].append(stationRoutePlays)
        routeDataDict['nStationRoutes'].append(len(stationRoutePlays))

    # Convert dictionary to dataframe
    routeData = pd.DataFrame.from_dict(routeDataDict)

    # Calculate proportion of motion and station route plays for each player
    routeData['motionRouteProp'] = routeData['nMotionRoutes'] / (routeData['nMotionRoutes'] + routeData['nStationRoutes'])
    routeData['stationRouteProp'] = routeData['nStationRoutes'] / (routeData['nMotionRoutes'] + routeData['nStationRoutes'])

    # Save route data to dictionary
    with open(os.path.join('..', 'outputs', 'results', 'summaryRouteData.pkl'), 'wb') as pklFile:
        pickle.dump(routeData, pklFile)

else:

    # Load the already extracted route data
    with open(os.path.join('..', 'outputs', 'results', 'summaryRouteData.pkl'), 'rb') as pklFile:
        routeData = pd.DataFrame.from_dict(pickle.load(pklFile))

    # Get list of route runners for later analysis
    routeRunners = routeData['nflId'].to_list()

# =========================================================================
# Process player image head shots for visualisations
# =========================================================================

"""

Note that this doesn't need to be re-run more than once. It takes a bit of time,
particularly if you're cropping all images, so isn't really worth running again
unless the details of route runners are changed. Change the flags below if wishing
to re-process images.

"""

# Download player images flag
getPlayerImages = False

# Crop player images flag
cropPlayerImages = False

# Check whether to download player images
if getPlayerImages:

    # Run the image download function
    # Only download images for route runners selected
    downloadPlayerImages(rosterData.loc[rosterData['gsis_it_id'].isin([str(nflId) for nflId in routeRunners])].reset_index(drop = True),
                         os.path.join('..','img','player'))

#Check whether to crop images
if cropPlayerImages:

    # Set the tuple of player image files and colouring circular border
    imgList = [(os.path.join('..','img','player',str(nflId)+'.png'),
                teamData.loc[teamData['team_abbr'] == routeData.loc[routeData['nflId'] == nflId]['teamName'].values[0],]['team_color'].values[0],
                teamData.loc[teamData['team_abbr'] == routeData.loc[routeData['nflId'] == nflId]['teamName'].values[0],]['team_color2'].values[0]) \
               for nflId in routeData['nflId']]

    #Import and run the helper function
    cropPlayerImg(imgList, os.path.join('..','img','player'))

# =========================================================================
# Create a visual for the top 10 highest proportion of motion routes
# =========================================================================

# Check for creating visuals
if createVisuals:

    # Extract the top 10 motion route players
    top10_motionRouteIds = routeData.sort_values(by = 'motionRouteProp', ascending = False).iloc[0:10]['nflId'].to_list()

    # Display top 10 in-motion route runners
    for nflId in top10_motionRouteIds:
        # Get player and team name
        displayName = routeData.loc[routeData['nflId'] == nflId]['playerName'].values[0]
        teamName = routeData.loc[routeData['nflId'] == nflId]['teamName'].values[0]
        # Print out summary
        print(f'#{top10_motionRouteIds.index(nflId)+1}: {displayName} [id: {nflId}] ({teamName}) - '
              f'{"{0:.2f}".format(routeData.loc[routeData["nflId"] == nflId]["motionRouteProp"].values[0]*100)}% motion routes'
              f' ({routeData.loc[routeData["nflId"] == nflId]["nMotionRoutes"].values[0]}/'
              f'{routeData.loc[routeData["nflId"] == nflId]["nMotionRoutes"].values[0] + routeData.loc[routeData["nflId"] == nflId]["nStationRoutes"].values[0]}'
              f' routes in motion at snap)')

    # Create figure
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6,8))

    # Set subplot spacing
    plt.subplots_adjust(left = 0.025, right = 0.975, bottom = 0.02, top = 0.92)

    # Add figure title
    fig.text(0.01, 0.97,
             'Top 10 Players for % of Routes In Motion at the Snap',
             font = 'Kuunari', fontsize = 24,
             ha = 'left', va = 'center')

    #Add descriptive text
    fig.text(0.01, 0.94,
             'Players required a minimum of 100 routes run in the Big Data Bowl 2025 dataset',
             font = 'Arial', fontsize = 8, fontweight = 'normal',
             ha = 'left', va = 'center', fontstyle = 'italic')

    # Set the x and y limits
    ax.set_xlim([-13,100])
    ax.set_ylim([0.5,10.5])

    # Invert y-axis so #1 is at the top
    ax.yaxis.set_inverted(True)

    # Turn off axis
    ax.axis('off')

    # Add player data
    for nflId in top10_motionRouteIds:

        # Add image
        # -------------------------------------------------------------------------

        # Load player image
        playerImg = plt.imread(os.path.join('..', 'img', 'player', f'{nflId}_cropped_col0.png'))
        # Create the offset image
        offsetImg = OffsetImage(playerImg, zoom=0.04)
        offsetImg.image.axes = ax
        # Create the annotation box
        aBox = AnnotationBbox(offsetImg, [-7.5, top10_motionRouteIds.index(nflId) + 1],
                              bboxprops={'lw': 0, 'fc': 'none', 'clip_on': False}
                              )
        # Add the image
        ax.add_artist(aBox)

        # Add bars
        # -------------------------------------------------------------------------

        # First add empty bar for 100% indicator
        # Create the rectangle
        fullBox = patches.Rectangle((0, top10_motionRouteIds.index(nflId) + 1 - 0.1),
                                         100, 0.15,
                                         clip_on=False,
                                         edgecolor=teamData.loc[
                                             teamData['team_abbr'] == routeData.loc[routeData['nflId'] == nflId]['teamName'].values[0],
                                         ]['team_color'].values[0], facecolor='none')
        # Add to axis
        ax.add_patch(fullBox)

        # Next fill the box to desired points
        propRec = patches.Rectangle((0, top10_motionRouteIds.index(nflId) + 1.0 - 0.1),
                                    routeData.loc[routeData['nflId'] == nflId]['motionRouteProp'].values[0]*100, 0.15,
                                    clip_on=False,
                                    edgecolor='none',
                                    facecolor=teamData.loc[
                                              teamData['team_abbr'] == routeData.loc[routeData['nflId'] == nflId]['teamName'].values[0],
                                    ]['team_color'].values[0])
        # Add to axis
        ax.add_patch(propRec)

        # Add player text
        # -------------------------------------------------------------------------

        # Add player name
        playerName = routeData.loc[routeData['nflId'] == nflId]['playerName'].values[0]
        ax.text(0, top10_motionRouteIds.index(nflId) + 1.0 - 0.1, playerName,
                color=teamData.loc[
                    teamData['team_abbr'] == routeData.loc[routeData['nflId'] == nflId]['teamName'].values[0],
                ]['team_color'].values[0],
                font='Quakerhack', fontsize=16,
                ha='left', va='bottom', clip_on=False)

        # Add player details
        playerNo = '#'+str(int(rosterData.loc[rosterData['gsis_it_id'] == str(nflId),['jersey_number']].values[0][0]))
        playerPos = rosterData.loc[rosterData['gsis_it_id'] == str(nflId), ['position']].values[0][0]
        playerTeam = teamData.loc[teamData['team_abbr'] == routeData.loc[routeData['nflId'] == nflId]['teamName'].values[0]]['team_name'].values[0]
        ax.text(0, top10_motionRouteIds.index(nflId) + 1.0 + 0.1,
                f'{playerNo}, {playerPos}, {playerTeam}',
                color=teamData.loc[
                    teamData['team_abbr'] == routeData.loc[routeData['nflId'] == nflId]['teamName'].values[0],
                ]['team_color'].values[0],
                font='Arial', fontsize=8, fontweight='normal',
                ha='left', va='top', clip_on=False)

        # Add route proportion
        routePropStr = f'{"{0:.1f}".format(routeData.loc[routeData["nflId"] == nflId]["motionRouteProp"].values[0]*100)}% of routes in motion at the snap'
        ax.text(100, top10_motionRouteIds.index(nflId) + 1.0 + 0.1,
                routePropStr,
                color=teamData.loc[
                    teamData['team_abbr'] == routeData.loc[routeData['nflId'] == nflId]['teamName'].values[0],
                ]['team_color'].values[0],
                font='Arial', fontsize=8, fontweight='normal',
                ha='right', va='top', clip_on=False)

    # Save figure
    fig.savefig(os.path.join('..','outputs','figure','topPlayers_inMotionRoutes.png'),
                format = 'png', dpi = 600, transparent = True)

    # Close figure
    plt.close('all')

# =========================================================================
# Quantify proportion of passing plays with motion at snap for teams
# =========================================================================

# Check for calculating
if calcInMotionIndex:

    # Set-up dictionary to store data
    teamPlaysDict = {'teamAbbr': [], 'nStationaryPlays': [], 'nMotionPlays': []}

    # Loop through teams
    for teamAbbr in tqdm(teamData['team_abbr']):

        # Get team passing plays
        teamPlays = plays.loc[
            (plays['possessionTeam'] == teamAbbr) &
            (plays['passResult'].isin(['C','I','S','IN','R']))
        ]

        # Set variables to count up
        nStationaryPlays = 0
        nMotionPlays = 0

        # Loop through game and play Ids and identify plays as stationary or in motion
        for gameId, playId in zip(teamPlays['gameId'], teamPlays['playId']):
            # Extract in motion player info for current play
            inMotionAtSnap = player_play.loc[
                (player_play['gameId'] == gameId) &
                (player_play['playId'] == playId) &
                (player_play['teamAbbr'] == teamAbbr)
            ]['inMotionAtBallSnap'].to_list()
            # Extract whether route was run at snap
            runningRoute = player_play.loc[
                (player_play['gameId'] == gameId) &
                (player_play['playId'] == playId) &
                (player_play['teamAbbr'] == teamAbbr)
                ]['wasRunningRoute'].to_list()
            # Look up combinations and sum
            nPlayersInMotionRouteAtSnap = np.sum([inMotionAtSnap[ii] == True and runningRoute[ii] == 1.0 for ii in range(len(inMotionAtSnap))])
            # Add appropriately to play indicators
            if nPlayersInMotionRouteAtSnap > 0:
                nMotionPlays += 1
            else:
                nStationaryPlays += 1

        # Append to dictionary
        teamPlaysDict['teamAbbr'].append(teamAbbr)
        teamPlaysDict['nStationaryPlays'].append(nStationaryPlays)
        teamPlaysDict['nMotionPlays'].append(nMotionPlays)

    # Save results to file
    with open(os.path.join('..', 'outputs', 'results', 'summaryTeamPlays.pkl'), 'wb') as pklFile:
        pickle.dump(teamPlaysDict, pklFile)

    # Convert to dataframe
    teamMotionDf = pd.DataFrame.from_dict(teamPlaysDict)

    # Calculate proportion of in-motion plays and sort by these values
    teamMotionDf['propInMotion'] = teamMotionDf['nMotionPlays'] / (teamMotionDf['nMotionPlays']+teamMotionDf['nStationaryPlays'])
    teamMotionDf.sort_values(by = 'propInMotion', ascending = False, inplace = True)

else:

    # Load the already extracted team data
    with open(os.path.join('..', 'outputs', 'results', 'summaryTeamPlays.pkl'), 'rb') as pklFile:
        teamPlaysDict = pd.DataFrame.from_dict(pickle.load(pklFile))

    # Convert to dataframe for later analysis
    teamMotionDf = pd.DataFrame.from_dict(teamPlaysDict)

    # Calculate proportion of in-motion plays and sort by these values
    teamMotionDf['propInMotion'] = teamMotionDf['nMotionPlays'] / (teamMotionDf['nMotionPlays'] + teamMotionDf['nStationaryPlays'])
    teamMotionDf.sort_values(by='propInMotion', ascending=False, inplace=True)

# Check for calculating if results need to be printed
if calcInMotionIndex:

    # Display data for in-motion routes by teams
    # Note that this includes some zeros for teams that are no longer in competition!
    for ii in range(len(teamMotionDf)):
        # Print out summary
        print(f'#{ii + 1}: {teamMotionDf.iloc[ii]["teamAbbr"]} - '
              f'{"{0:.2f}".format(teamMotionDf.iloc[ii]["propInMotion"] * 100)}% of passing plays with player in motion at snap')

# =========================================================================
# Create visual of top 10 teams for motion at snap
# =========================================================================

# Check for creating visuals
if createVisuals:

    # Create figure
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6,8))

    # Set subplot spacing
    plt.subplots_adjust(left = 0.025, right = 0.975, bottom = 0.02, top = 0.92)

    # Add figure title
    fig.text(0.01, 0.97,
             'Top 10 Teams for % of Pass Plays with Motion at the Snap',
             font = 'Kuunari', fontsize = 24,
             ha = 'left', va = 'center')

    #Add descriptive text
    fig.text(0.01, 0.94,
             'Includes all plays with a passing outcome from the Big Data Bowl 2025 dataset',
             font = 'Arial', fontsize = 8, fontweight = 'normal',
             ha = 'left', va = 'center', fontstyle = 'italic')

    # Set the x and y limits
    ax.set_xlim([-13,100])
    ax.set_ylim([0.5,10.5])

    # Invert y-axis so #1 is at the top
    ax.yaxis.set_inverted(True)

    # Turn off axis
    ax.axis('off')

    # Add player data
    for ii in range(10):

        # Get team data
        teamAbbr = teamMotionDf.iloc[ii]['teamAbbr']
        inMotionProp = teamMotionDf.iloc[ii]['propInMotion']
        teamCol = teamData.loc[teamData['team_abbr'] == teamAbbr]['team_color'].values[0]
        teamName = teamData.loc[teamData['team_abbr'] == teamAbbr]['team_name'].values[0]

        # Fix annoying issue that '9' character isn't included in font for SF
        if teamName == 'San Francisco 49ers':
            teamName = 'San Francisco'

        # Add image
        # -------------------------------------------------------------------------

        # Load team image
        teamImg = plt.imread(os.path.join('..', 'img', 'team', f'{teamAbbr}.png'))
        # Create the offset image
        offsetImg = OffsetImage(teamImg, zoom=0.08)
        offsetImg.image.axes = ax
        # Create the annotation box
        aBox = AnnotationBbox(offsetImg, [-8.5, ii + 1],
                              bboxprops={'lw': 0, 'fc': 'none', 'clip_on': False}
                              )
        # Add the image
        ax.add_artist(aBox)

        # Add bars
        # -------------------------------------------------------------------------

        # First add empty bar for 100% indicator
        # Create the rectangle
        fullBox = patches.Rectangle((0, ii + 1 - 0.1),
                                    100, 0.15,
                                    clip_on=False,
                                    edgecolor=teamCol,
                                    facecolor='none')
        # Add to axis
        ax.add_patch(fullBox)

        # Next fill the box to desired points
        propRec = patches.Rectangle((0, ii + 1.0 - 0.1),
                                    inMotionProp*100, 0.15,
                                    clip_on=False,
                                    edgecolor='none',
                                    facecolor=teamCol)
        # Add to axis
        ax.add_patch(propRec)

        # Add team text
        # -------------------------------------------------------------------------

        # Add team name
        ax.text(0, ii + 1.0 - 0.1, teamName,
                color=teamCol,
                font='Quakerhack', fontsize=16,
                ha='left', va='bottom', clip_on=False)

        # Add route proportion
        routePropStr = f'{"{0:.1f}".format(inMotionProp*100)}% of pass plays with a route runner in motion at the snap'
        ax.text(100, ii + 1.0 + 0.1,
                routePropStr,
                color=teamCol,
                font='Arial', fontsize=8, fontweight='normal',
                ha='right', va='top', clip_on=False)

    # Save figure
    fig.savefig(os.path.join('..','outputs','figure','topTeams_inMotionRoutes.png'),
                format = 'png', dpi = 600, transparent = True)

    # Close figure
    plt.close('all')

# =========================================================================
# Explore situational characteristics of in motion routes
# =========================================================================

# Check for summarising data
if summariseData:

    # Extract passing plays from global play dataset
    passPlays = plays.loc[plays['isDropback']]

    # Set variable to store in motion route outcome
    inMotionRoutePlay = []

    # Loop through game and play Id's to identify whether an in motion route occurred
    # -------------------------------------------------------------------------
    print('Exploring passing plays for in motion routes...')
    for ii in tqdm(range(len(passPlays))):

        # Get game and play Id
        gameId = passPlays['gameId'].iloc[ii]
        playId = passPlays['playId'].iloc[ii]

        # Get possession team for current play
        posTeam = passPlays.loc[(passPlays['gameId'] == gameId) &
                                (passPlays['playId'] == playId)]['possessionTeam'].values[0]

        # Extract in motion player info for current play
        inMotionAtSnap = player_play.loc[
            (player_play['gameId'] == gameId) &
            (player_play['playId'] == playId) &
            (player_play['teamAbbr'] == posTeam)
            ]['inMotionAtBallSnap'].to_list()

        # Extract whether route was run at snap
        runningRoute = player_play.loc[
            (player_play['gameId'] == gameId) &
            (player_play['playId'] == playId) &
            (player_play['teamAbbr'] == posTeam)
            ]['wasRunningRoute'].to_list()

        # Look up combinations and sum
        nPlayersInMotionRouteAtSnap = np.sum([inMotionAtSnap[ii] == True and runningRoute[ii] == 1.0 for ii in range(len(inMotionAtSnap))])

        # Allocate outcome from play
        if nPlayersInMotionRouteAtSnap > 0:
            inMotionRoutePlay.append(True)
        else:
            inMotionRoutePlay.append(False)

    # Append to pass plays dataframe
    passPlays['inMotionRoutePlay'] = inMotionRoutePlay

    # Explore characteristics of in motion route plays
    # -------------------------------------------------------------------------

    # Extract the in motion route plays from passing dataset
    passPlaysInMotion = passPlays.loc[passPlays['inMotionRoutePlay']]

    # Print out summary value for number of in motion route plays
    print(f'Total number of passing plays with receiver in motion at the snap: {len(passPlaysInMotion)}')

    # Group by offensive formation to explore frequency
    groupedOffenseFormation = passPlaysInMotion.groupby('offenseFormation').count()['playId'].copy()
    groupedOffenseFormation.sort_values(ascending = False, inplace = True)
    print(f'{"*"*10} In Motion Route Plays by Offense Formation {"*"*10}')
    for formation in groupedOffenseFormation.index:
        print(f'{formation}: {groupedOffenseFormation[formation]} plays ({"{0:.2f}".format(groupedOffenseFormation[formation] / len(passPlaysInMotion) * 100)}%)')

    # Group by receiver alignment to explore frequency
    groupedReceiverAlignment = passPlaysInMotion.groupby('receiverAlignment').count()['playId'].copy()
    groupedReceiverAlignment.sort_values(ascending=False, inplace=True)
    print(f'{"*" * 10} In Motion Route Plays by Receiver Alignment {"*" * 10}')
    for alignment in groupedReceiverAlignment.index:
        print(
            f'{alignment} alignment: {groupedReceiverAlignment[alignment]} plays ({"{0:.2f}".format(groupedReceiverAlignment[alignment] / len(passPlaysInMotion) * 100)}%)')

    # Group by down to explore frequency
    groupedDown = passPlaysInMotion.groupby('down').count()['playId'].copy()
    groupedDown.sort_values(ascending=False, inplace=True)
    print(f'{"*" * 10} In Motion Route Plays by Down {"*" * 10}')
    for down in groupedDown.index:
        print(
            f'Down {down}: {groupedDown[down]} plays ({"{0:.2f}".format(groupedDown[down] / len(passPlaysInMotion) * 100)}%)')

    # Group by quarter to explore frequency
    groupedQuarter = passPlaysInMotion.groupby('quarter').count()['playId'].copy()
    groupedQuarter.sort_values(ascending=False, inplace=True)
    print(f'{"*" * 10} In Motion Route Plays by Quarter {"*" * 10}')
    for quarter in groupedQuarter.index:
        print(
            f'Quarter {quarter}: {groupedQuarter[quarter]} plays ({"{0:.2f}".format(groupedQuarter[quarter] / len(passPlaysInMotion) * 100)}%)')

# =========================================================================
# Collate individual player route data for calculating index
# =========================================================================

"""

Metrics included in this section from calculations:

    > targeted: a boolean of whether or not the player was vs. wasn't targeted on the play
    > reception: a boolean of whether or not the player did vs. didn't get the reception on the play
    > yards: with receptions, the total number of yards on the play
    > yac: with receptions, the number of yards after the catch on the play
    > peakSeparation: maximum separation from closest defender in yards from the minimum time to throw in dataset to the pass frame
    > separationAtCatch: with receptions, the separation in yards from nearest defender at the catch
    > releaseSpeed: average player velocity in the first second after the snap

"""

# Check for running calculations
if calcInMotionIndex:

    # Settings
    # -------------------------------------------------------------------------

    # Set release speed time window frames
    # Calculated on frame rate of 0.1 seconds per frame
    releaseSpeedDuration = 1.0  # seconds after snap
    releaseSpeedFramesN = releaseSpeedDuration / 0.1

    # Set distance and time threshold for open receiver
    # Calculated on frame rate of 0.1 seconds per frame
    openDistance = 3.0  # yards
    openTime = 0.5  # seconds
    openFramesN = openTime / 0.1

    # Calculate 25th percentile for time to throw in dataset
    # Calculated on frame rate of 0.1 seconds per frame
    timeToThrow25 = plays['timeToThrow'].describe()['25%']
    timeToThrow25FramesN = int(np.round(timeToThrow25 / 0.1))

    # Get the minimum time to through in the dataset
    # Calculated on frame rate of 0.1 seconds per frame
    timeToThrowMin = plays['timeToThrow'].describe()['min']
    timeToThrowMinFramesN = int(np.round(timeToThrowMin / 0.1))

    # Columns of route data to work through
    analyseColumns = ['stationRoutePlays', 'motionRoutePlays']

    # Loop through the identified route runners
    # -------------------------------------------------------------------------
    print('Calculating metrics across route runners...')
    for nflId in tqdm(routeData['nflId']):

        # Create a dictionary to store route result data in
        playerData = {'nflId': [], 'gameId': [], 'playId': [], 'inMotionAtSnap': [],
                      'targeted': [], 'reception': [], 'catch': [], 'yards': [], 'yac': [],
                      'createdSpace': [], 'createdSpaceEarly': [],
                      'peakSeparation': [], 'separationAtCatch': [],
                      'releaseSpeed': []}

        # Loop through the analysis columns
        # -------------------------------------------------------------------------
        for routeAnalysis in analyseColumns:

            # Set the in motion at snap variable for this loop
            if 'station' in routeAnalysis:
                inMotionAtSnap = False
            else:
                inMotionAtSnap = True

            # Loop through route type game and play Id's
            for gamePlayId in routeData.loc[routeData['nflId'] == nflId][routeAnalysis].values[0]:

                # Get play data
                # -------------------------------------------------------------------------

                # Extract game and play Ids to variables
                gameId = gamePlayId[0]
                playId = gamePlayId[1]

                # Get the players data for the current play
                currPlayPlayerData = player_play.loc[
                    (player_play['nflId'] == nflId) &
                    (player_play['gameId'] == gameId) &
                    (player_play['playId'] == playId),]

                # Get the tracking data for the play
                # Get the week number to look up tracking data
                weekNo = games.loc[(games['gameId'] == gameId)]['week'].values[0]
                # Get the tracking data for the play
                currPlayTrackingData = tracking[f'week{weekNo}'].loc[
                    (tracking[f'week{weekNo}']['gameId'] == gameId) &
                    (tracking[f'week{weekNo}']['playId'] == playId)
                ]
                # Get the player data from the tracking of the play
                currPlayPlayerTrackingData = currPlayTrackingData.loc[
                    currPlayTrackingData['nflId'] == nflId
                ]

                # Get targeting and catch variables
                # -------------------------------------------------------------------------

                # Check if player was targeted and got the reception
                targeted = bool(currPlayPlayerData['wasTargettedReceiver'].values[0])
                reception = bool(currPlayPlayerData['hadPassReception'].values[0])

                # Check if player made the catch on target
                if targeted and reception:
                    catch = True
                elif targeted and not reception:
                    catch = False
                else:
                    catch = np.nan

                # Check for receiving yards if appropriate
                if reception:
                    # Extract yards on play and yards after catch
                    yards = currPlayPlayerData['receivingYards'].values[0]
                    yac = currPlayPlayerData['yardageGainedAfterTheCatch'].values[0]
                else:
                    # Set values as nan
                    yards = np.nan
                    yac = np.nan

                # Get space creation and separation variables
                # -------------------------------------------------------------------------

                # Calculate openness across play and in short time frame after snap

                # Get snap and the frame of the first listed pass event
                snapFrameId = currPlayTrackingData.loc[currPlayTrackingData['frameType'] == 'SNAP']['frameId'].unique()[0]
                # Get pass event name
                passEvent = [str(eventName) for eventName in currPlayTrackingData['event'].unique().tolist() if str(eventName).startswith('pass_')][0]
                # Get the pass event frame
                passFrameId = currPlayTrackingData.loc[currPlayTrackingData['event'] == passEvent]['frameId'].unique()[0]

                # Get frame Id for 'early' after snap
                # Check if this is more than pass frame and alter if necessary
                earlyFrameId = snapFrameId + timeToThrow25FramesN
                if earlyFrameId > passFrameId:
                    earlyFrameId = passFrameId

                # Get players position from snap to pass
                playerXY = currPlayPlayerTrackingData.loc[
                    (currPlayPlayerTrackingData['frameId'] >= snapFrameId) &
                    (currPlayPlayerTrackingData['frameId'] <= passFrameId),
                ][['x', 'y']].values

                # Get the defensive team player Ids
                defensiveIds = currPlayTrackingData.loc[
                    ~currPlayTrackingData['club'].isin([routeData.loc[routeData['nflId'] == nflId]['teamName'].values[0], 'football'])
                ]['nflId'].unique().tolist()

                # Calculate distance of defensive players from receiver across frames
                minDefDist = np.zeros((len(playerXY)))
                # Loop through frames
                for frameInd in range(len(minDefDist)):
                    # Set values for current frame
                    defDist = np.zeros((len(defensiveIds)))
                    # Loop through each defender on the frame
                    for defId in defensiveIds:
                        # Get defender position at catch
                        defXY = currPlayTrackingData.loc[
                            (currPlayTrackingData['nflId'] == defId) &
                            (currPlayTrackingData['frameId'] == snapFrameId+frameInd),
                        ][['x', 'y']].values[0]
                        # Calculate distance to receiver and store in array
                        defDist[defensiveIds.index(defId)] = np.linalg.norm(playerXY[frameInd] - defXY)
                    # Set minimum defender distance for current frame
                    minDefDist[frameInd] = np.min(defDist)

                # Get the minimum defender distance in the early time frame
                if earlyFrameId == passFrameId:
                    minDefDistEarly = minDefDist
                else:
                    minDefDistEarly = minDefDist[0:earlyFrameId-snapFrameId]

                # Default creating space variable to False
                createdSpace = False
                createdSpaceEarly = False

                # Identify length of consecutive frames where player was considered open based on distance
                lenOpen = []
                lenOpenEarly = []
                for jj,kk in groupby(minDefDist >= openDistance):
                    if jj == True:
                        lenOpen.append(len(list(kk)))
                for jj, kk in groupby(minDefDistEarly >= openDistance):
                    if jj == True:
                        lenOpenEarly.append(len(list(kk)))

                # If any open periods are greater than the frames threshold change the variable to True
                if any(np.array(lenOpen) >= openFramesN):
                    createdSpace = True
                if any(np.array(lenOpenEarly) >= openFramesN):
                    createdSpaceEarly = True

                # Calculate the peak separation
                peakSeparation = np.max(minDefDist[timeToThrowMinFramesN::])

                # Get separation at catch if appropriate
                if reception:
                    # Get frame Id for catch
                    # Sometimes pass arrived is present, while other times it is pass outcome caught
                    # Probably happens when these events need to be the same frame
                    # Other times there isn't a pass outcome, so pass forward frame is used as a last ditch effort
                    try:
                        # Attempt to get the pass arrived event
                        passArrivedId = currPlayTrackingData.loc[
                            currPlayTrackingData['event'] == 'pass_arrived'
                            ]['frameId'].unique()[0]
                    except:
                        try:
                            # Attempted to get the pass outcome caught event
                            passArrivedId = currPlayTrackingData.loc[
                                currPlayTrackingData['event'] == 'pass_outcome_caught'
                                ]['frameId'].unique()[0]
                        except:
                            # Use the pass forward event as a last ditch effort
                            passArrivedId = currPlayTrackingData.loc[
                                currPlayTrackingData['event'] == 'pass_forward'
                                ]['frameId'].unique()[0]
                    # Get players position at pass arrived timing
                    playerXY = currPlayPlayerTrackingData.loc[
                        currPlayPlayerTrackingData['frameId'] == passArrivedId,
                        ][['x','y']].values[0]
                    # Get the defensive team player Ids
                    defensiveIds = currPlayTrackingData.loc[
                        ~currPlayTrackingData['club'].isin([routeData.loc[routeData['nflId'] == nflId]['teamName'].values[0],'football'])
                    ]['nflId'].unique().tolist()
                    # Calculate distance of defensive players from receiver at pass arrived event
                    defDist = np.zeros((len(defensiveIds)))
                    for defId in defensiveIds:
                        # Get defender position at catch
                        defXY = currPlayTrackingData.loc[
                            (currPlayTrackingData['nflId'] == defId) &
                            (currPlayTrackingData['frameId'] == passArrivedId),
                            ][['x', 'y']].values[0]
                        # Calculate distance to receiver and store in array
                        defDist[defensiveIds.index(defId)] = np.linalg.norm(playerXY - defXY)
                    # Store separation as the minimum defender distance
                    separationAtCatch = np.nanmin(defDist)
                else:
                    # Set value as nan
                    separationAtCatch = np.nan

                # Calculate speed variables
                # -------------------------------------------------------------------------

                # Calculate release speed
                # Get snap frame Id
                snapFrameId = currPlayTrackingData.loc[
                    currPlayTrackingData['frameType'] == 'SNAP'
                ]['frameId'].unique()[0]
                # Get players speed over the desired number of frames following
                releaseSpeed = currPlayPlayerTrackingData.loc[
                    (currPlayPlayerTrackingData['frameId'] > snapFrameId) &
                    (currPlayPlayerTrackingData['frameId'] <= snapFrameId + releaseSpeedFramesN)
                ]['s'].to_numpy().mean()

                # Store in dictionary
                playerData['nflId'].append(nflId)
                playerData['gameId'].append(gameId)
                playerData['playId'].append(gameId)
                playerData['inMotionAtSnap'].append(inMotionAtSnap)
                playerData['targeted'].append(targeted)
                playerData['reception'].append(reception)
                playerData['catch'].append(catch)
                playerData['yards'].append(yards)
                playerData['yac'].append(yac)
                playerData['createdSpace'].append(createdSpace)
                playerData['createdSpaceEarly'].append(createdSpaceEarly)
                playerData['peakSeparation'].append(peakSeparation)
                playerData['separationAtCatch'].append(separationAtCatch)
                playerData['releaseSpeed'].append(releaseSpeed)

        # Save the player data to file
        with open(os.path.join('..', 'outputs', 'player', f'{nflId}_routeData.pkl'), 'wb') as pklFile:
            pickle.dump(playerData, pklFile)

# =========================================================================
# Run some basic checks on player data
# =========================================================================

"""

The below section is commented offers some basic print outs to review
the stationary vs. in-motion summary data for a given player.

"""

# Check for summarising data
if summariseData:

    # Set player Id to look up (Travis Kelce used as an example here)
    nflId = 40011

    # Load in players data
    with open(os.path.join('..', 'outputs', 'player', f'{nflId}_routeData.pkl'), 'rb') as pklFile:
        playerData = pd.DataFrame.from_dict(pickle.load(pklFile))

    # Print out some basic descriptives
    print(f'Summary data for {players.loc[players["nflId"] == nflId]["displayName"].values[0]}')

    # Target rate
    # Get data to support descriptions
    summaryData = playerData.groupby(['inMotionAtSnap','targeted']).count()
    # Print out data
    print(f'{"*"*20} TARGET RATE {"*"*20}')
    print(f'Targeted on {summaryData.loc[(False,True)]["nflId"]} of {len(playerData.loc[playerData["inMotionAtSnap"] == False])} routes when stationary at snap '
          f'({"{0:.2f}".format(summaryData.loc[(False,True)]["nflId"] / len(playerData.loc[playerData["inMotionAtSnap"] == False]) * 100)}% of stationary routes)')
    print(f'Targeted on {summaryData.loc[(True,True)]["nflId"]} of {len(playerData.loc[playerData["inMotionAtSnap"] == True])} routes when in-motion at snap '
          f'({"{0:.2f}".format(summaryData.loc[(True,True)]["nflId"] / len(playerData.loc[playerData["inMotionAtSnap"] == True]) * 100)}% of in-motion routes)')

    # Reception rate
    # Get data to support descriptions
    summaryData = playerData.groupby(['inMotionAtSnap','reception']).count()
    # Print out data
    print(f'{"*"*20} RECEPTION RATE {"*"*20}')
    print(f'Received pass on {summaryData.loc[(False,True)]["nflId"]} of {len(playerData.loc[playerData["inMotionAtSnap"] == False])} routes when stationary at snap '
          f'({"{0:.2f}".format(summaryData.loc[(False,True)]["nflId"] / len(playerData.loc[playerData["inMotionAtSnap"] == False]) * 100)}% of stationary routes)')
    print(f'Received pass on {summaryData.loc[(True,True)]["nflId"]} of {len(playerData.loc[playerData["inMotionAtSnap"] == True])} routes when in-motion at snap '
          f'({"{0:.2f}".format(summaryData.loc[(True,True)]["nflId"] / len(playerData.loc[playerData["inMotionAtSnap"] == True]) * 100)}% of in-motion routes)')

    # Average yards
    # Get data to support descriptions
    summaryData = playerData.groupby(['inMotionAtSnap'])['yards'].mean()
    # Print out data
    print(f'{"*"*20} AVERAGE YARDS {"*"*20}')
    print(f'Average receiving yards when stationary at snap: {"{0:.2f}".format(summaryData[False])}')
    print(f'Average receiving yards when in-motion at snap: {"{0:.2f}".format(summaryData[True])}')

    # Average yards after catch
    # Get data to support descriptions
    summaryData = playerData.groupby(['inMotionAtSnap'])['yac'].mean()
    # Print out data
    print(f'{"*"*20} AVERAGE YARDS AFTER CATCH {"*"*20}')
    print(f'Average yards after catch when stationary at snap: {"{0:.2f}".format(summaryData[False])}')
    print(f'Average yards after catch when in-motion at snap: {"{0:.2f}".format(summaryData[True])}')

    # Average separation at catch
    # Get data to support descriptions
    summaryData = playerData.groupby(['inMotionAtSnap'])['separationAtCatch'].mean()
    # Print out data
    print(f'{"*"*20} AVERAGE SEPARATION AT CATCH {"*"*20}')
    print(f'Average separation at catch when stationary at snap: {"{0:.2f}".format(summaryData[False])}')
    print(f'Average separation at catch when in-motion at snap: {"{0:.2f}".format(summaryData[True])}')

    # Average release speed
    # Get data to support descriptions
    summaryData = playerData.groupby(['inMotionAtSnap'])['releaseSpeed'].mean()
    # Print out data
    print(f'{"*"*20} AVERAGE RELEASE SPEED {"*"*20}')
    print(f'Average release speed when stationary at snap: {"{0:.2f}".format(summaryData[False])}')
    print(f'Average release speed when in-motion at snap: {"{0:.2f}".format(summaryData[True])}')

# =========================================================================
# Check eligibility for in motion index
# =========================================================================

"""

There are circumstances where the IMI metrics can't be calculated for a player,
so the dataset is reduced here to only include players with at least:

    > At least 10 stationary routes & 10 in-motion routes
    > At least 1 target on stationary and 1 target on in motion routes
    > At least 1 catch on stationary and 1 catch on in motion routes

Players that don't meet these criteria are excluded at the calculation step.

There will also be some IMI indices that can't be calculated for certain players
given the limited sample size of some outcomes.

"""

# Create list to store eligibility
eligibleIMI = []

# Loop through route runners
for nflId in routeData['nflId']:

    # Load in the player route data and convert to dataframe for ease of use
    with open(os.path.join('..', 'outputs', 'player', f'{nflId}_routeData.pkl'), 'rb') as pklFile:
        playerData = pd.DataFrame.from_dict(pickle.load(pklFile))

    # Get data to run checks for including player
    # If any of these fail given the data isn't there then the player won't meet the criteria anyway
    try:

        # Get data
        stationaryRoutes = playerData.groupby('inMotionAtSnap').count().loc[False]['nflId']
        inMotionRoutes = playerData.groupby('inMotionAtSnap').count().loc[True]['nflId']
        stationaryTarget = playerData.groupby(['inMotionAtSnap', 'targeted']).count().loc[(False, True)]['nflId']
        inMotionTarget = playerData.groupby(['inMotionAtSnap', 'targeted']).count().loc[(True, True)]['nflId']
        stationaryRec = playerData.groupby(['inMotionAtSnap', 'reception']).count().loc[(False, True)]['nflId']
        inMotionRec = playerData.groupby(['inMotionAtSnap', 'reception']).count().loc[(True, True)]['nflId']

        # Check to keep player
        if all([
            stationaryRoutes >= 10, inMotionRoutes >= 10,
            stationaryTarget > 0, inMotionTarget > 0,
            stationaryRec > 0, inMotionRec > 0
        ]):
            keepPlayer = True
        else:
            keepPlayer = False
    except:
        keepPlayer = False

    # Append to list if appropriate
    if keepPlayer:
        eligibleIMI.append(nflId)

# Check number of eligible vs. ineligible
print(f'Players eligible for IMI calculations: {len(eligibleIMI)}')
print(f'Players removed from IMI calculations: {len(routeData["nflId"]) - len(eligibleIMI)}')

# =========================================================================
# Calculate in motion index metrics
# =========================================================================

"""

Here is where the individual in motion indice values are calculated. Although 
there were some eligibility checks, there will still likely be some IMI indices
that can't be calculated for certain players given the limited sample size of
some outcomes.

"""

# Check for running calculations
if calcInMotionIndex:

    # Settings
    # -------------------------------------------------------------------------

    # Set number of samples to take in Monte Carlo approach
    nSamples = 10000

    # Loop through eligible route runners
    # -------------------------------------------------------------------------
    print('Calculating IMI for eligible route runners...')
    for nflId in tqdm(eligibleIMI):

        # Load in the player route data and convert to dataframe for ease of use
        with open(os.path.join('..', 'outputs', 'player', f'{nflId}_routeData.pkl'), 'rb') as pklFile:
            playerData = pd.DataFrame.from_dict(pickle.load(pklFile))

        # Get summary data to support calculations
        playerCountData = playerData.groupby(['inMotionAtSnap','reception']).count()

        # Get supporting data for IMI calculations
        # -------------------------------------------------------------------------

        # Create dictionary to store data in
        calcImiData = {}

        # Loop through boolean variables to extract data for
        for dataVar in ['targeted', 'reception',
                        # 'createdSpace', 'createdSpaceEarly'
                        ]:
            # Create key in dictionary
            calcImiData[dataVar] = {}
            # Extract data using try:except to avoid errors and allocate zeros where appropriate
            try:
                calcImiData[dataVar]['stationaryTrue'] = playerData.groupby(['inMotionAtSnap', dataVar]).count().loc[(False, True)]['nflId']
            except:
                calcImiData[dataVar]['stationaryTrue'] = 0
            try:
                calcImiData[dataVar]['stationaryFalse'] = playerData.groupby(['inMotionAtSnap', dataVar]).count().loc[(False, False)]['nflId']
            except:
                calcImiData[dataVar]['stationaryFalse'] = 0
            try:
                calcImiData[dataVar]['inMotionTrue'] = playerData.groupby(['inMotionAtSnap', dataVar]).count().loc[(True, True)]['nflId']
            except:
                calcImiData[dataVar]['inMotionTrue'] = 0
            try:
                calcImiData[dataVar]['inMotionFalse'] = playerData.groupby(['inMotionAtSnap', dataVar]).count().loc[(True, False)]['nflId']
            except:
                calcImiData[dataVar]['inMotionFalse'] = 0

        # Loop through continuous variables to extract data for
        for dataVar in ['yards', 'yac', 'peakSeparation', 'separationAtCatch', 'releaseSpeed']:
            # Create key in dictionary
            calcImiData[dataVar] = {}
            # Extract data using (unlikely to need try/except given eligibility criteria)
            calcImiData[dataVar]['stationaryMu'] = playerData.groupby(['inMotionAtSnap']).mean(numeric_only=True)[dataVar].loc[False]
            calcImiData[dataVar]['stationarySigma'] = playerData.groupby(['inMotionAtSnap']).std(numeric_only=True)[dataVar].loc[False]
            calcImiData[dataVar]['inMotionMu'] = playerData.groupby(['inMotionAtSnap']).mean(numeric_only=True)[dataVar].loc[True]
            calcImiData[dataVar]['inMotionSigma'] = playerData.groupby(['inMotionAtSnap']).std(numeric_only=True)[dataVar].loc[True]

        # Calculate IMI values
        # -------------------------------------------------------------------------

        # Use function to calculate IMI
        imiResults = calcIMI(calcImiData, weightsIMI, nSamples, nflId)

        # Save dictionary to file
        with open(os.path.join('..', 'outputs', 'player', f'{nflId}_IMI.pkl'), 'wb') as pklFile:
            pickle.dump(imiResults, pklFile)

# =========================================================================
# Summarise IMI results
# =========================================================================

# Check for summarising data
if summariseData:

    # Read in eligible player data to create an export of average values
    # -------------------------------------------------------------------------

    # Create dictionary to store values
    playerImiResults = {'nflId': [], 'playerName': [], 'position': [], 'IMI': [],
                        'targetedIMI': [], 'receptionIMI': [],
                        'yardsIMI': [], 'yacIMI': [],
                        'peakSeparationIMI': [], 'separationAtCatchIMI': [],
                        'releaseSpeedIMI': [],
                        'nStationaryRoutes': [], 'nInMotionRoutes': [],
                        'nStationaryReceptions': [], 'nInMotionReceptions': []}

    # Loop through eligible players
    for nflId in eligibleIMI:

        # Read in players IMI results
        with open(os.path.join('..', 'outputs', 'player', f'{nflId}_IMI.pkl'), 'rb') as pklFile:
            imiResults = pickle.load(pklFile)

        # Load in the player route data and convert to dataframe for ease of use
        with open(os.path.join('..', 'outputs', 'player', f'{nflId}_routeData.pkl'), 'rb') as pklFile:
            playerData = pd.DataFrame.from_dict(pickle.load(pklFile))

        # Get summary data to include in results
        playerCountData = playerData.groupby(['inMotionAtSnap', 'reception']).count()

        # Extract average data across variables
        playerImiResults['nflId'].append(nflId)
        playerImiResults['playerName'].append(rosterData.loc[rosterData['gsis_it_id'] == str(nflId),]['player_name'].values[0])
        playerImiResults['position'].append(rosterData.loc[rosterData['gsis_it_id'] == str(nflId),]['position'].values[0])
        playerImiResults['IMI'].append(np.nanmean(imiResults['IMI']['sampleVals']))
        playerImiResults['targetedIMI'].append(np.nanmean(imiResults['targeted']['mean']))
        playerImiResults['receptionIMI'].append(np.nanmean(imiResults['reception']['mean']))
        playerImiResults['yardsIMI'].append(np.nanmean(imiResults['yards']['mean']))
        playerImiResults['yacIMI'].append(np.nanmean(imiResults['yac']['mean']))
        playerImiResults['peakSeparationIMI'].append(np.nanmean(imiResults['peakSeparation']['mean']))
        playerImiResults['separationAtCatchIMI'].append(np.nanmean(imiResults['separationAtCatch']['mean']))
        playerImiResults['releaseSpeedIMI'].append(np.nanmean(imiResults['releaseSpeed']['mean']))
        playerImiResults['nStationaryRoutes'].append(len(playerData.loc[playerData['inMotionAtSnap'] == False]))
        playerImiResults['nInMotionRoutes'].append(len(playerData.loc[playerData['inMotionAtSnap'] == True]))
        playerImiResults['nStationaryReceptions'].append(len(playerData.loc[(playerData['inMotionAtSnap'] == False) &
                                                                            (playerData['reception'] == True)]))
        playerImiResults['nInMotionReceptions'].append(len(playerData.loc[(playerData['inMotionAtSnap'] == True) &
                                                                          (playerData['reception'] == True)]))

    # Save dictionary format of player IMI results
    with open(os.path.join('..', 'outputs', 'results', 'playerIMI.pkl'), 'wb') as pklFile:
            pickle.dump(playerImiResults, pklFile)

    # Convert to dataframe
    playerImiData = pd.DataFrame.from_dict(playerImiResults)

    # Sort by overall IMI
    playerImiData.sort_values(by = 'IMI', ascending = False, inplace = True)

    # Export to file
    # Rename columns here for cleanliness
    playerImiData.rename(columns = {'nflId': 'NFL ID', 'playerName': 'Player Name', 'position': 'Playing Position', 'IMI': 'IMI',
                                    'targetedIMI': 'IMI Target Multiplier', 'receptionIMI': 'IMI Reception Multiplier',
                                    'yardsIMI': 'IMI Yards Multiplier', 'yacIMI': 'IMI YAC Multiplier',
                                    'peakSeparationIMI': 'IMI Peak Separation Multiplier', 'separationAtCatchIMI': 'IMI Separation at Catch Multiplier',
                                    'releaseSpeedIMI': 'IMI Release Speed Multiplier',
                                    'nStationaryRoutes': 'No. of Stationary Routes Run', 'nInMotionRoutes': 'No. of In Motion Routes Run',
                                    'nStationaryReceptions': 'No. of Receptions on Stationary Routes',
                                    'nInMotionReceptions': 'No. of Receptions on In Motion Routes'}).to_csv(
        os.path.join('..', 'outputs', 'results', 'playerIMI.csv'), index = False)

# Otherwise load in saved results
else:

    # Load the already extracted team data
    with open(os.path.join('..', 'outputs', 'results', 'playerIMI.pkl'), 'rb') as pklFile:
        playerImiResults = pickle.load(pklFile)

    # Convert to dataframe
    playerImiData = pd.DataFrame.from_dict(playerImiResults)

    # Sort by overall IMI
    playerImiData.sort_values(by='IMI', ascending=False, inplace=True)

# =========================================================================
# Create IMI polar plots
# =========================================================================

# Check for creating visuals
if createVisuals:

    # Create individualised plots for players
    # -------------------------------------------------------------------------

    # Set the number of variables
    imiPlotVars = ['targeted', 'reception', 'yards', 'yac', 'peakSeparation', 'separationAtCatch', 'releaseSpeed']
    nPlotVars = len(imiPlotVars)

    # Set theta values for spider plot
    theta = radar_factory(nPlotVars, frame = 'polygon') * -1

    # Identify contribution from components to players IMI
    varContribution = {var: weightsIMI[var] / np.sum([weightsIMI[jj] for jj in imiPlotVars]) for var in imiPlotVars}
    for var in imiPlotVars:
        playerImiData[f'{var}IMI_contribution'] = [playerImiData[f'{var}IMI'].iloc[ii] * varContribution[var] for ii in range(len(playerImiData))]

    # Identify max and minimum weighted contributions
    maxC = np.max([playerImiData[f'{var}IMI_contribution'].max() for var in imiPlotVars])
    minC = np.min([playerImiData[f'{var}IMI_contribution'].min() for var in imiPlotVars])

    # Loop through eligible players
    print('Creating player IMI figures...')
    for nflId in tqdm(eligibleIMI):

        # Settings
        # -------------------------------------------------------------------------

        # Read in players IMI results
        with open(os.path.join('..', 'outputs', 'player', f'{nflId}_IMI.pkl'), 'rb') as pklFile:
            imiResults = pickle.load(pklFile)

        # Create the figure
        fig = plt.figure(figsize=(14, 7))

        # Set figure face colour transparency
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(0.0)

        # Set subplot spacing
        plt.subplots_adjust(left=0.02, right=0.98, bottom=0.16, top=0.95, wspace=0.05)

        # Get player details
        playerName = routeData.loc[routeData['nflId'] == nflId]['playerName'].values[0]
        playerNo = '#' + str(int(rosterData.loc[rosterData['gsis_it_id'] == str(nflId), ['jersey_number']].values[0][0]))
        playerPos = rosterData.loc[rosterData['gsis_it_id'] == str(nflId), ['position']].values[0][0]
        playerTeam = routeData.loc[routeData['nflId'] == nflId]['teamName'].values[0]
        playerTeamFull = teamData.loc[teamData['team_abbr'] == routeData.loc[routeData['nflId'] == nflId]['teamName'].values[0]]['team_name'].values[0]

        # Get the plot colour for the player
        teamCol = teamData.loc[teamData['team_abbr'] == playerTeam,]['team_color'].values[0]
        teamCol2 = teamData.loc[teamData['team_abbr'] == playerTeam,]['team_color2'].values[0]

        # Add the player image
        # -------------------------------------------------------------------------

        # Add the axis
        imgAx = fig.add_subplot(1,2,1)

        # Set axis values for ease and consistency
        imgAx.set_xlim([0,1])
        imgAx.set_ylim([0, 1])

        # Load player image
        playerImg = plt.imread(os.path.join('..', 'img', 'player', f'{nflId}.png'))

        # Create the offset image
        offsetImg = OffsetImage(playerImg, zoom=0.35)
        offsetImg.image.axes = imgAx

        # Create the annotation box
        aBox = AnnotationBbox(offsetImg, [0.54,0.0],
                              box_alignment = (0.5, 0.0),
                              bboxprops={'lw': 0, 'fc': 'none', 'zorder': 1, 'clip_on': False}
                              )

        # Add the image
        imgAx.add_artist(aBox)

        # Turn the axis off
        imgAx.axis('off')

        # Add the polar plot
        # -------------------------------------------------------------------------

        # Add the axis and set the projection
        plotAx = fig.add_subplot(1,2,2,projection = 'radar')

        # Get the players data
        imiPlayer = playerImiData.loc[playerImiData['nflId'] == nflId,]
        playerC = [imiPlayer.iloc[0][f'{var}IMI_contribution'] for var in imiPlotVars]

        # Use max and min values to set limits
        plotAx.set_ylim([minC, maxC + 0.10])

        # Add label for each multiplier component
        plotAx.text(theta[0], maxC + 0.05, 'Targets', rotation=90,
                    color=teamCol, font='Arial', fontweight='bold', fontsize=10,
                    ha='center', va='top', zorder = 4)
        plotAx.text(theta[1], maxC + 0.05, 'Rec.', rotation=90-np.rad2deg(theta[1]*-1),
                    color=teamCol, font='Arial', fontweight='bold', fontsize=10,
                    ha='right', va='top', zorder = 4)
        plotAx.text(theta[2], maxC + 0.05, 'Yards', rotation=90 - np.rad2deg(theta[2] * -1),
                    color=teamCol, font='Arial', fontweight='bold', fontsize=10,
                    ha='right', va='center', zorder = 4)
        plotAx.text(theta[3], maxC + 0.05, 'YAC', rotation=90 - np.rad2deg(theta[3] * -1),
                    color=teamCol, font='Arial', fontweight='bold', fontsize=10,
                    ha='right', va='bottom', zorder = 4)
        plotAx.text(theta[4], maxC + 0.05, 'Peak Sep.', rotation=-90 - np.rad2deg(theta[4] * -1),
                    color=teamCol, font='Arial', fontweight='bold', fontsize=10,
                    ha='left', va='bottom', zorder = 4)
        plotAx.text(theta[5], maxC + 0.05, 'Sep. at Catch', rotation=-90 - np.rad2deg(theta[5] * -1),
                    color=teamCol, font='Arial', fontweight='bold', fontsize=10,
                    ha='left', va='bottom', zorder = 4)
        plotAx.text(theta[6], maxC + 0.05, 'Rel. Speed', rotation=-90 - np.rad2deg(theta[6] * -1),
                    color=teamCol, font='Arial', fontweight='bold', fontsize=10,
                    ha='left', va='top', zorder = 4)

        # Turn off grids and ticks
        # plotAx.set_ylim([0, 0.75])
        plotAx.grid(axis='y', linestyle=':', linewidth=0.0, color='dimgrey', alpha=0.0)
        plotAx.set_yticklabels([])

        # Remove spoke grid lines
        plotAx.set_xticklabels([])
        plotAx.grid(axis='x', linestyle=':', linewidth=0.0, color='dimgrey', alpha=0.0)

        # Set polar spine borders to team colour
        plotAx.spines['polar'].set(color=teamCol, lw=2.5)

        # Set axes background colour
        plotAx.set_facecolor('white')

        # Add a line that would represent a weighted IMI of 1
        imiOneLine = plotAx.plot(theta, [varContribution[var] for var in imiPlotVars],
                                 lw=1.0, ls=':', c='dimgrey', alpha=1.0,
                                 zorder=1, clip_on=False)

        # Plot mean IMI data and fill
        imiLine = plotAx.plot(theta, playerC,
                              lw = 2.0, ls = '-', c = teamCol2,
                              zorder = 3, clip_on = False)
        imiFill = plotAx.fill(theta, playerC,
                              facecolor = teamCol2, alpha = 0.30,
                              zorder = 2, clip_on = False,
                              label = '_nolegend')

        # Increase size of polar axis for viewability
        axPos = plotAx.get_position()
        plotAx.set_position((axPos.x0, axPos.y0, (axPos.x1-axPos.x0)*1.04, (axPos.y1-axPos.y0)*1.04))

        # Add the annotations
        # -------------------------------------------------------------------------

        # Add the line underneath axes
        # Get the bottom left of the player image axis in the figure coordinates
        imgAxPos = (imgAx.get_position().x0, imgAx.get_position().y0)
        # Get position opposite to this representing the other axis border
        endLinePos = (1 - imgAxPos[0], imgAxPos[1])
        # Transform the figure coordinate to display and then into the data coordinates
        endLinePosData = imgAx.transData.inverted().transform(fig.transFigure.transform(endLinePos))
        # Draw line on image axis by transforming position
        imgAx.plot([0.0, endLinePosData[0]], [0.0, 0.0],
                   ls='-', lw=2.5, c=teamCol, zorder=4, clip_on=False)

        # Read in and display team logo
        teamImg = plt.imread(os.path.join('..', 'img', 'team', f'{playerTeam}.png'))
        offsetImg = OffsetImage(teamImg, zoom=0.095)
        offsetImg.image.axes = imgAx
        aBox = AnnotationBbox(offsetImg, [0.0,0.0],
                              box_alignment = (0.0, 1.0),
                              bboxprops={'lw': 0, 'fc': 'none', 'zorder': 1, 'clip_on': False}
                              )
        imgAx.add_artist(aBox)

        # Add the player name
        imgAx.text(0.14, -0.02, playerName,
                   color = teamCol, font ='Quakerhack', fontsize = 44,
                   ha = 'left', va = 'top', clip_on = False)

        # Add player details
        imgAx.text(0.14, -0.16, f'{playerNo}, {playerPos}, {playerTeamFull}',
                   color = teamCol, font = 'Arial', fontweight = 'normal', fontsize=12,
                   ha = 'left', va = 'top', clip_on = False)

        # Add the players overall IMI
        imiVal = playerImiData.loc[playerImiData['nflId'] == nflId,]['IMI'].values[0]
        imgAx.text(endLinePosData[0], -0.02, f'IMI: {"{0:.3f}".format(np.round(imiVal,3))}',
                   color=teamCol2, font='Kuunari', fontsize = 44,
                   ha='right', va='top', clip_on=False)

        # Add the confidence intervals around the IMI
        lb95 = imiResults['IMI']['sampleVals'].mean() - (1.96 * (imiResults['IMI']['sampleVals'].std() / np.sqrt(len(imiResults['IMI']['sampleVals']))))
        ub95 = imiResults['IMI']['sampleVals'].mean() + (1.96 * (imiResults['IMI']['sampleVals'].std() / np.sqrt(len(imiResults['IMI']['sampleVals']))))
        imgAx.text(endLinePosData[0], -0.12, f'[{"{0:.3f}".format(np.round(lb95,3))}, {"{0:.3f}".format(np.round(ub95,3))} 95% CIs]',
                   color=teamCol2, font='Arial', fontweight='normal', fontsize=12,
                   ha='right', va='top', clip_on=False)

        # Save player figure
        # -------------------------------------------------------------------------

        # Save figure
        fig.savefig(os.path.join('..', 'outputs', 'figure', 'player', f'{nflId}_imiSummary.png'),
                    format='png', dpi=600,
                    # transparent=True
                    facecolor = fig.get_facecolor(), edgecolor='none'
                    )

        # Close figure
        plt.close('all')

    # Create overall IMI plot for all players
    # -------------------------------------------------------------------------

    # Create the figure
    fig = plt.figure(figsize=(10, 28))

    # Set figure face colour transparency
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(0.0)

    # Set subplots position and spacing
    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.925, wspace=0.20, hspace=0.55)

    # Add figure title
    fig.text(0.01, 0.97,
             'In Motion Index (IMI) of Eligible Players',
             font='Kuunari', fontsize=24,
             ha='left', va='center')

    # Add descriptive text
    fig.text(0.01, 0.945,
             'Players ordered according to overall IMI. Players required a minimum of 100 routes run in the Big Data Bowl 2025 dataset to be eligible.',
             font='Arial', fontsize=8, fontweight='normal',
             ha='left', va='center', fontstyle='italic')

    # Loop through player rankings to plot their data
    for ii in range(len(playerImiData)):

        # Get the player Id
        nflId = playerImiData.iloc[ii]['nflId']

        # Read in players IMI results
        with open(os.path.join('..', 'outputs', 'player', f'{nflId}_IMI.pkl'), 'rb') as pklFile:
            imiResults = pickle.load(pklFile)

        # Get player details
        playerName = routeData.loc[routeData['nflId'] == nflId]['playerName'].values[0]
        playerNo = '#' + str(int(rosterData.loc[rosterData['gsis_it_id'] == str(nflId), ['jersey_number']].values[0][0]))
        playerPos = rosterData.loc[rosterData['gsis_it_id'] == str(nflId), ['position']].values[0][0]
        playerTeam = routeData.loc[routeData['nflId'] == nflId]['teamName'].values[0]
        playerTeamFull = teamData.loc[teamData['team_abbr'] == routeData.loc[routeData['nflId'] == nflId]['teamName'].values[0]]['team_name'].values[0]

        # Get the plot colour for the player
        teamCol = teamData.loc[teamData['team_abbr'] == playerTeam,]['team_color'].values[0]
        teamCol2 = teamData.loc[teamData['team_abbr'] == playerTeam,]['team_color2'].values[0]

        # Add the player image
        # -------------------------------------------------------------------------

        # Add the axis
        if ii < 50:
            imgAx = fig.add_subplot(11, 10, int(np.linspace(1,199,100)[ii]))
        else:
            imgAx = fig.add_subplot(11, 10, int(np.linspace(1, 199, 100)[ii])+2)

        # Set axis values for ease and consistency
        imgAx.set_xlim([0, 1])
        imgAx.set_ylim([0, 1])

        # Load player image
        playerImg = plt.imread(os.path.join('..', 'img', 'player', f'{nflId}.png'))

        # Create the offset image
        offsetImg = OffsetImage(playerImg, zoom=0.035)
        offsetImg.image.axes = imgAx

        # Create the annotation box
        aBox = AnnotationBbox(offsetImg, [0.54, 0.0],
                              box_alignment=(0.5, 0.0),
                              bboxprops={'lw': 0, 'fc': 'none', 'zorder': 1, 'clip_on': False}
                              )

        # Add the image
        imgAx.add_artist(aBox)

        # Turn the axis off
        imgAx.axis('off')

        # Add the polar plot
        # -------------------------------------------------------------------------

        # Add the axis and set the projection
        if ii < 50:
            plotAx = fig.add_subplot(11, 10, int(np.linspace(2,200,100)[ii]), projection='radar')
        else:
            plotAx = fig.add_subplot(11, 10, int(np.linspace(2, 200, 100)[ii])+2, projection='radar')

        # Get the players data
        imiPlayer = playerImiData.loc[playerImiData['nflId'] == nflId,]
        playerC = [imiPlayer.iloc[0][f'{var}IMI_contribution'] for var in imiPlotVars]

        # Use max and min values to set limits
        plotAx.set_ylim([minC, maxC + 0.10])

        # Add label for each multiplier component
        plotAx.text(theta[0], maxC - 0.05, 'T',
                    color=teamCol, font='Arial', fontweight='bold', fontsize=2,
                    ha='center', va='center', zorder=4)
        plotAx.text(theta[1], maxC - 0.05, 'R',
                    color=teamCol, font='Arial', fontweight='bold', fontsize=2,
                    ha='center', va='center', zorder=4)
        plotAx.text(theta[2], maxC - 0.05, 'Y',
                    color=teamCol, font='Arial', fontweight='bold', fontsize=2,
                    ha='center', va='center', zorder=4)
        plotAx.text(theta[3], maxC - 0.05, 'YAC',
                    color=teamCol, font='Arial', fontweight='bold', fontsize=2,
                    ha='center', va='center', zorder=4)
        plotAx.text(theta[4], maxC - 0.05, 'PS',
                    color=teamCol, font='Arial', fontweight='bold', fontsize=2,
                    ha='center', va='center', zorder=4)
        plotAx.text(theta[5], maxC - 0.05, 'SC',
                    color=teamCol, font='Arial', fontweight='bold', fontsize=2,
                    ha='center', va='center', zorder=4)
        plotAx.text(theta[6], maxC - 0.05, 'RS',
                    color=teamCol, font='Arial', fontweight='bold', fontsize=2,
                    ha='center', va='center', zorder=4)

        # Turn off grids and ticks
        # plotAx.set_ylim([0, 0.75])
        plotAx.grid(axis='y', linestyle=':', linewidth=0.0, color='dimgrey', alpha=0.0)
        plotAx.set_yticklabels([])

        # Remove spoke grid lines
        plotAx.set_xticklabels([])
        plotAx.grid(axis='x', linestyle=':', linewidth=0.0, color='dimgrey', alpha=0.0)

        # Set polar spine borders to team colour
        plotAx.spines['polar'].set(color=teamCol, lw=0.75)

        # Set axes background colour
        plotAx.set_facecolor('white')

        # Add a line that would represent a weighted IMI of 1
        imiOneLine = plotAx.plot(theta, [varContribution[var] for var in imiPlotVars],
                                 lw=0.25, ls=':', c='dimgrey', alpha=1.0,
                                 zorder=1, clip_on=False)

        # Plot mean IMI data and fill
        imiLine = plotAx.plot(theta, playerC,
                              lw=0.75, ls='-', c=teamCol2,
                              zorder=3, clip_on=False)
        imiFill = plotAx.fill(theta, playerC,
                              facecolor=teamCol2, alpha=0.30,
                              zorder=2, clip_on=False,
                              label='_nolegend')

        # Alter polar axis position for viewability
        axPos = plotAx.get_position()
        plotAx.set_position((axPos.x0, axPos.y0 + 0.0025, (axPos.x1 - axPos.x0), (axPos.y1 - axPos.y0)))

        # Add the annotations
        # -------------------------------------------------------------------------

        # Add the line underneath axes
        # Get the bottom left of the player image axis in the figure coordinates
        imgAxPos = imgAx.get_position()
        plotAxPos = plotAx.get_position()
        # Get position opposite to this representing the other axis border
        endLinePos = (plotAxPos.x1, plotAxPos.y0)
        # Transform the figure coordinate to display and then into the data coordinates
        endLinePosData = imgAx.transData.inverted().transform(fig.transFigure.transform(endLinePos))
        # Draw line on image axis by transforming position
        imgAx.plot([0.0, endLinePosData[0]], [0.0, 0.0],
                   ls='-', lw=0.75, c=teamCol, zorder=4, clip_on=False)

        # Add the player name
        imgAx.text(0.0, -0.03, playerName,
                   color=teamCol, font='Quakerhack', fontsize=8,
                   ha='left', va='top', clip_on=False)

        # Add player details
        imgAx.text(0.0, -0.26, f'{playerNo}, {playerPos}, {playerTeamFull}',
                   color=teamCol, font='Arial', fontweight='normal', fontsize=4,
                   ha='left', va='top', clip_on=False)

        # Add the players overall IMI
        imiVal = playerImiData.loc[playerImiData['nflId'] == nflId,]['IMI'].values[0]
        imgAx.text(endLinePosData[0], -0.03, f'IMI: {"{0:.3f}".format(np.round(imiVal, 3))}',
                   color=teamCol2, font='Kuunari', fontsize=8,
                   ha='right', va='top', clip_on=False)

        # Add the confidence intervals around the IMI
        lb95 = imiResults['IMI']['sampleVals'].mean() - (
                    1.96 * (imiResults['IMI']['sampleVals'].std() / np.sqrt(len(imiResults['IMI']['sampleVals']))))
        ub95 = imiResults['IMI']['sampleVals'].mean() + (
                    1.96 * (imiResults['IMI']['sampleVals'].std() / np.sqrt(len(imiResults['IMI']['sampleVals']))))
        imgAx.text(endLinePosData[0], -0.26, f'[{"{0:.3f}".format(np.round(lb95, 3))}, {"{0:.3f}".format(np.round(ub95, 3))} 95% CIs]',
                   color=teamCol2, font='Arial', fontweight='normal', fontsize=4,
                   ha='right', va='top', clip_on=False)

    # Save group figure
    # -------------------------------------------------------------------------

    # Save figure
    fig.savefig(os.path.join('..', 'outputs', 'figure', f'allPlayers_imiSummary.png'),
                format='png', dpi=600,
                # transparent=True
                facecolor=fig.get_facecolor(), edgecolor='none'
                )

    # Close figure
    plt.close('all')

# =========================================================================
# Create leaderboards for individual multiplier components
# =========================================================================

# Check for creating visuals
if createVisuals:

    # Create the multiplier leaderboards for each variable
    # -------------------------------------------------------------------------

    # Set the variables to create leaderboards for
    leaderboardVars = ['targeted', 'reception', 'yards', 'yac', 'peakSeparation', 'separationAtCatch', 'releaseSpeed']
    leaderboardVars_label = ['Targets', 'Receptions', 'Yards', 'YAC', 'Peak Separation', 'Separation at Catch', 'Release Speed']

    # Loop through variables
    print('Creating visuals for IMI multiplier leaderboards...')
    for var in tqdm(leaderboardVars):

        # Create a copy and sort by the current multiplier
        leaderboardData = playerImiData.copy()
        leaderboardData.sort_values(by = f'{var}IMI', ascending = False, inplace = True)

        # Create the
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6,7))

        # Set subplot spacing
        plt.subplots_adjust(left = 0.125, right = 0.975, bottom = 0.075, top = 0.9)

        # Add figure title
        fig.text(0.01, 0.97,
                 f'Top 5 Players for {leaderboardVars_label[leaderboardVars.index(var)]} IMI Multiplier',
                 font = 'Kuunari', fontsize = 24,
                 ha = 'left', va = 'center')

        # Add descriptive text
        fig.text(0.01, 0.94,
                 'Data represents the average estimate \u00b1 95% confidence intervals for multiplier',
                 font='Arial', fontsize=8, fontweight='normal',
                 ha='left', va='center', fontstyle='italic')

        # Set the y limits for the top 5
        ax.set_ylim([0.5, 5.5])

        # Invert y-axis so #1 is at the top
        ax.yaxis.set_inverted(True)

        # Add player data
        # -------------------------------------------------------------------------
        for ii in range(5):

            # Get the players Id
            nflId = leaderboardData.iloc[ii]['nflId']

            # Read in players IMI results
            with open(os.path.join('..', 'outputs', 'player', f'{nflId}_IMI.pkl'), 'rb') as pklFile:
                imiResults = pickle.load(pklFile)

            # Get colouring details
            playerTeam = routeData.loc[routeData['nflId'] == nflId]['teamName'].values[0]
            teamCol = teamData.loc[teamData['team_abbr'] == playerTeam,]['team_color'].values[0]
            teamCol2 = teamData.loc[teamData['team_abbr'] == playerTeam,]['team_color2'].values[0]

            # Add point at mean
            ax.scatter(imiResults[var]['mean'], ii + 1 + 0.15,
                       s=75, marker='o', fc=teamCol, ec=teamCol2,
                       zorder=3)

            # Add the confidence intervals
            ax.plot([imiResults[var]['lowerBound'], imiResults[var]['upperBound']], [ii + 1 + 0.15, ii + 1 + 0.15],
                    c=teamCol2, lw=1.5, ls='-',
                    zorder=2)

        # Clean up axes
        # -------------------------------------------------------------------------

        # Reset x-axis to zero lower limit
        ax.set_xlim([0, ax.get_xlim()[1]])

        # Add x-axis label
        ax.set_xlabel('IMI Multiplier', fontsize = 12, fontweight = 'bold', labelpad = 7.5)

        # Set x-axis ticks to whole numbers
        if ax.get_xlim()[1] <= 12:
            ax.set_xticks(np.arange(0,np.round(ax.get_xlim()[1])+1))
        else:
            ax.set_xticks(np.arange(0, np.round(ax.get_xlim()[1]) + 1, 2))

        # Remove y-tick labels and ticks
        ax.set_yticklabels([])
        ax.tick_params(axis = 'y', length = 0)

        # Edit x-tick parameters for top and bottom
        ax.tick_params(axis = 'x', top = True, labeltop = True, direction = 'in')

        # Remove the undesired spines
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add player details
        # -------------------------------------------------------------------------

        # Re-loop through players
        for ii in range(5):

            # Get the players Id
            nflId = leaderboardData.iloc[ii]['nflId']

            # Get player details
            playerName = routeData.loc[routeData['nflId'] == nflId]['playerName'].values[0]
            playerNo = '#' + str(int(rosterData.loc[rosterData['gsis_it_id'] == str(nflId), ['jersey_number']].values[0][0]))
            playerPos = rosterData.loc[rosterData['gsis_it_id'] == str(nflId), ['position']].values[0][0]
            playerTeam = routeData.loc[routeData['nflId'] == nflId]['teamName'].values[0]
            playerTeamFull = teamData.loc[teamData['team_abbr'] == routeData.loc[routeData['nflId'] == nflId]['teamName'].values[0]]['team_name'].values[0]

            # Get the plot colour for the player
            teamCol = teamData.loc[teamData['team_abbr'] == playerTeam,]['team_color'].values[0]
            teamCol2 = teamData.loc[teamData['team_abbr'] == playerTeam,]['team_color2'].values[0]

            # Add image
            # -------------------------------------------------------------------------

            # Load player image
            playerImg = plt.imread(os.path.join('..', 'img', 'player', f'{nflId}_cropped_col0.png'))
            # Create the offset image
            offsetImg = OffsetImage(playerImg, zoom=0.04)
            offsetImg.image.axes = ax
            # Create the annotation box
            aBox = AnnotationBbox(offsetImg, [0.0, ii+1],
                                  box_alignment=(1.0, 0.5),
                                  bboxprops={'lw': 0, 'fc': 'none', 'clip_on': False}
                                  )
            # Add the image
            ax.add_artist(aBox)

            # Add player text
            # -------------------------------------------------------------------------

            # Add player name
            nameText = ax.text(0, ii+1 - 0.03, playerName,
                               color = teamCol,
                               font='Quakerhack', fontsize=16,
                               ha='left', va='bottom', clip_on=False)

            # Add player details
            detailText = ax.text(0, ii+1,
                                 f'{playerNo}, {playerPos}, {playerTeam}',
                                 color = teamCol,
                                 font='Arial', fontsize=8, fontweight='normal',
                                 ha='left', va='top', clip_on=False)

        # Save to file
        # -------------------------------------------------------------------------

        # Save figure
        fig.savefig(os.path.join('..', 'outputs', 'figure', f'topPlayers_{var}.png'),
                    format='png', dpi=600, transparent=True)

        # Close figure
        plt.close('all')

# =========================================================================
# Create quadrant plot for in motion routes vs. IMI
# =========================================================================

# Check for creating visuals
if createVisuals:

    # Create plot of in motion usage vs. IMI
    # -------------------------------------------------------------------------

    # Calculate z-score for in motion route proportion across players
    playerImiData['inMotionRouteProp'] = playerImiData['nInMotionRoutes'] / (playerImiData['nInMotionRoutes'] + playerImiData['nStationaryRoutes'])
    playerImiData['inMotionRoutePropZ'] = (playerImiData['inMotionRouteProp'] - playerImiData['inMotionRouteProp'].mean()) / playerImiData['inMotionRouteProp'].std()

    # Create figure to plot on
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (9,8))
    plt.subplots_adjust(left = 0.04, right = 0.96, bottom = 0.05, top = 0.91)

    # Add figure title
    fig.text(0.01, 0.97,
             'Player In Motion Route Efficiency: Usage vs. IMI',
             font='Kuunari', fontsize=24,
             ha='left', va='center')

    # Add descriptive text
    fig.text(0.01, 0.94,
             'In motion route usage calculated as a proportion relative to other eligible players and referenced against the player''s IMI',
             font='Arial', fontsize=8, fontweight='normal',
             ha='left', va='center', fontstyle='italic')

    # Create scatter plot using team colours
    # -------------------------------------------------------------------------

    # Loop through dataset
    for ii in range(len(playerImiData)):

        # Get player data
        inMotionRoutePropZ = playerImiData.iloc[ii]['inMotionRoutePropZ']
        imi = playerImiData.iloc[ii]['IMI']

        # Get player details
        nflId = playerImiData.iloc[ii]['nflId']
        playerName = routeData.loc[routeData['nflId'] == nflId]['playerName'].values[0]
        playerNo = '#' + str(int(rosterData.loc[rosterData['gsis_it_id'] == str(nflId), ['jersey_number']].values[0][0]))
        playerPos = rosterData.loc[rosterData['gsis_it_id'] == str(nflId), ['position']].values[0][0]
        playerTeam = routeData.loc[routeData['nflId'] == nflId]['teamName'].values[0]
        playerTeamFull = teamData.loc[teamData['team_abbr'] == routeData.loc[routeData['nflId'] == nflId]['teamName'].values[0]]['team_name'].values[0]

        # Get colouring details
        teamCol = teamData.loc[teamData['team_abbr'] == playerTeam,]['team_color'].values[0]
        teamCol2 = teamData.loc[teamData['team_abbr'] == playerTeam,]['team_color2'].values[0]

        # Add the data point
        ax.scatter(inMotionRoutePropZ, imi,
                   s = 25, marker = 'o', c = teamCol, ec = teamCol2,
                   zorder = 3, clip_on = False)

    # Set even x-axis
    if np.abs(ax.get_xlim()[1]) > np.abs(ax.get_xlim()[0]):
        ax.set_xlim([ax.get_xlim()[1]*-1, ax.get_xlim()[1]])
    else:
        ax.set_xlim([ax.get_xlim()[0], ax.get_xlim()[0]*-1])

    # Set even y-axis
    ax.set_ylim([1 - (ax.get_ylim()[1] - 1), ax.get_ylim()[1]])

    # Add quadrant lines
    # X-axis
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [1.0, 1.0], c = 'black', lw = 2.5, zorder = 2)
    ax.scatter(ax.get_xlim()[0], 1.0, s = 75, c = 'black', marker = '<', clip_on = False, zorder = 2)
    ax.scatter(ax.get_xlim()[1], 1.0, s = 75, c = 'black', marker = '>', clip_on = False, zorder = 2)
    # Y-axis
    ax.plot([0.0, 0.0], [ax.get_ylim()[0], ax.get_ylim()[1]], c = 'black', lw = 2.5, zorder = 2)
    ax.scatter(0.0, ax.get_ylim()[1], s = 75, c = 'black', marker = '^', clip_on = False, zorder = 2)
    ax.scatter(0.0, ax.get_ylim()[0], s = 75, c = 'black', marker = 'v', clip_on = False, zorder = 2)

    # Turn axis off
    ax.axis('off')

    # Add axis labelling
    # -------------------------------------------------------------------------

    # Add x-axis text
    ax.text(ax.get_xlim()[0] + 0.1, 1 - 0.03, 'Lower In Motion Route Usage',
            ha = 'left', va = 'top',
            font = 'Arial', fontsize = 10, fontweight = 'normal', fontstyle = 'italic',
            zorder = 2)
    ax.text(ax.get_xlim()[1] - 0.1, 1 + 0.03, 'Higher In Motion Route Usage',
            ha='right', va='bottom',
            font='Arial', fontsize=10, fontweight='normal', fontstyle='italic',
            zorder=2)

    # Add y-axis text
    ax.text(0 + 0.075, ax.get_ylim()[1] - 0.04, 'Higher IMI (> 1)',
            ha='left', va='top', rotation = 90,
            font='Arial', fontsize=10, fontweight='normal', fontstyle='italic',
            zorder=2)
    ax.text(0 - 0.075, ax.get_ylim()[0] + 0.04, 'Lower IMI (< 1)',
            ha='right', va='bottom', rotation=90,
            font='Arial', fontsize=10, fontweight='normal', fontstyle='italic',
            zorder=2)

    # Save to file
    # -------------------------------------------------------------------------

    # Save figure
    fig.savefig(os.path.join('..', 'outputs', 'figure', f'inMotionUsage_v_IMI.png'),
                format='png', dpi=600, transparent=True)

    # Close figure
    plt.close('all')

    # Quickly repeat above plot with player labels for reference (simplified)
    # -------------------------------------------------------------------------

    # Create figure to plot on
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 8))
    plt.subplots_adjust(left=0.04, right=0.96, bottom=0.05, top=0.91)

    # Create scatter plot using team colours
    # -------------------------------------------------------------------------

    # Loop through dataset
    for ii in range(len(playerImiData)):
        # Get player data
        inMotionRoutePropZ = playerImiData.iloc[ii]['inMotionRoutePropZ']
        imi = playerImiData.iloc[ii]['IMI']

        # Get player details
        nflId = playerImiData.iloc[ii]['nflId']
        playerName = routeData.loc[routeData['nflId'] == nflId]['playerName'].values[0]
        playerNo = '#' + str(int(rosterData.loc[rosterData['gsis_it_id'] == str(nflId), ['jersey_number']].values[0][0]))
        playerPos = rosterData.loc[rosterData['gsis_it_id'] == str(nflId), ['position']].values[0][0]
        playerTeam = routeData.loc[routeData['nflId'] == nflId]['teamName'].values[0]
        playerTeamFull = teamData.loc[teamData['team_abbr'] == routeData.loc[routeData['nflId'] == nflId]['teamName'].values[0]]['team_name'].values[
            0]

        # Get colouring details
        teamCol = teamData.loc[teamData['team_abbr'] == playerTeam,]['team_color'].values[0]
        teamCol2 = teamData.loc[teamData['team_abbr'] == playerTeam,]['team_color2'].values[0]

        # Add the data point
        ax.scatter(inMotionRoutePropZ, imi,
                   s=25, marker='o', c=teamCol, ec=teamCol2,
                   zorder=3, clip_on=False)

    # Set even x-axis
    if np.abs(ax.get_xlim()[1]) > np.abs(ax.get_xlim()[0]):
        ax.set_xlim([ax.get_xlim()[1] * -1, ax.get_xlim()[1]])
    else:
        ax.set_xlim([ax.get_xlim()[0], ax.get_xlim()[0] * -1])

    # Set even y-axis
    ax.set_ylim([1 - (ax.get_ylim()[1] - 1), ax.get_ylim()[1]])

    # Add quadrant lines
    # X-axis
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [1.0, 1.0], c='black', lw=2.5, zorder=2)
    ax.scatter(ax.get_xlim()[0], 1.0, s=75, c='black', marker='<', clip_on=False, zorder=2)
    ax.scatter(ax.get_xlim()[1], 1.0, s=75, c='black', marker='>', clip_on=False, zorder=2)
    # Y-axis
    ax.plot([0.0, 0.0], [ax.get_ylim()[0], ax.get_ylim()[1]], c='black', lw=2.5, zorder=2)
    ax.scatter(0.0, ax.get_ylim()[1], s=75, c='black', marker='^', clip_on=False, zorder=2)
    ax.scatter(0.0, ax.get_ylim()[0], s=75, c='black', marker='v', clip_on=False, zorder=2)

    # Turn axis off
    ax.axis('off')

    # Add point labels
    # -------------------------------------------------------------------------

    # Annotate all points
    for ii, txt in enumerate(playerImiData['playerName'].to_list()):
        ax.annotate(
            txt, (playerImiData['inMotionRoutePropZ'].to_numpy()[ii],
                  playerImiData['IMI'].to_numpy()[ii]),
            font = 'Arial', fontsize = 8, fontweight = 'normal',
        )

    # Save to file
    # -------------------------------------------------------------------------

    # Save figure
    fig.savefig(os.path.join('..', 'outputs', 'figure', f'inMotionUsage_v_IMI_playerReference.png'),
                format='png', dpi=600, transparent=True)

    # Close figure
    plt.close('all')

# =========================================================================
# Option to visualise the motion in a motion route play
# =========================================================================

"""

NOTE: Below is commented out but provides the option to use a function
that can draw and animate plays.

"""

# # The sample play comes from Travis Kelce (40011)
# nflId = 40011
#
# # Provide a game and play Id. This particular play is a motion route from Kelce
# gameId = 2022091110
# playId = 2720
# weekNo = games.loc[games['gameId'] == gameId]['week'].values[0]
#
# # Get the tracking data for the play
# playTracking = tracking[f'week{weekNo}'].loc[
#     (tracking[f'week{weekNo}']['gameId'] == gameId) &
#     (tracking[f'week{weekNo}']['playId'] == playId)]
#
# # Get the play description
# playDesc = plays.loc[(plays['gameId'] == gameId) &
#                      (plays['playId'] == playId)]
#
# # Get the line of scrimmage and down markers
# lineOfScrimmage = playDesc['absoluteYardlineNumber'].to_numpy()[0]
# if playTracking['playDirection'].unique()[0] == 'left':
#     # Get the yards to go and invert for field direction
#     firstDownMark = lineOfScrimmage - playDesc['yardsToGo'].to_numpy()[0]
# else:
#     # Use standard values for right directed play
#     firstDownMark = lineOfScrimmage + playDesc['yardsToGo'].to_numpy()[0]
#
# # Get home and away teams for game
# homeTeam = games.loc[games['gameId'] == gameId,]['homeTeamAbbr'].values[0]
# awayTeam = games.loc[games['gameId'] == gameId,]['visitorTeamAbbr'].values[0]
#
# # Visualise the play at the snap
# # -------------------------------------------------------------------------
#
# # Create the field to plot on
# fieldFig, fieldAx = plt.subplots(figsize=(14, 6.5))
# createField(fieldFig, fieldAx,
#             lineOfScrimmage = lineOfScrimmage, firstDownMark = firstDownMark,
#             homeTeamAbbr = homeTeam, awayTeamAbbr = awayTeam, teamData = teamData)
#
# # Draw the play frame at the snap
# snapFrameId = playTracking.loc[playTracking['frameType'] == 'SNAP',]['frameId'].unique()[0]
#
# # Draw the frame
# drawFrame(snapFrameId, homeTeam, awayTeam, teamData,
#           playTracking, 'pos',
#           lineOfScrimmage = lineOfScrimmage,
#           firstDownMark = firstDownMark)
#
# # Animate play from line set to end
# # -------------------------------------------------------------------------
#
# # Create the field to plot on
# fieldFig, fieldAx = plt.subplots(figsize=(14, 6.5))
# createField(fieldFig, fieldAx,
#             lineOfScrimmage=lineOfScrimmage, firstDownMark=firstDownMark,
#             homeTeamAbbr=homeTeam, awayTeamAbbr=awayTeam, teamData=teamData)
#
# # Identify the frame range from play
# lineSetFrameId = playTracking.loc[playTracking['event'] == 'line_set',]['frameId'].unique()[0]
# endFrameId = playTracking['frameId'].unique().max()
#
# # # Set route runner Id (Kelce)
# # routeRunnerId = 40011
#
# # Run animation function
# anim = animation.FuncAnimation(fieldFig, drawFrame,
#                                frames=range(lineSetFrameId, endFrameId + 1), repeat=False,
#                                fargs=(homeTeam, awayTeam, teamData, playTracking, 'pos',
#                                       None, None, None, #routeRunnerId,
#                                       None, lineOfScrimmage, firstDownMark))
#
# # Write to GIF file
# gifWriter = animation.PillowWriter(fps=60)
# anim.save(os.path.join('..', 'outputs', 'gif', f'samplePlay_DiggsMotion_game-{gameId}_play-{playId}.gif'),
#           dpi=150, writer=gifWriter)
#
# # Close figure after pause to avoid error
# plt.pause(1)
# plt.close()

# %% ---------- end of InMotionIndex.py ---------- %% #

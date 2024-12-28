# -*- coding: utf-8 -*-
"""

@author:

    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au

    Python script for Big Data Bowl 2025 entry calculating In Motion Index.
    This script uses the associated helperFuncs.py script for visualisations.

    Run on python version 3.10.14

    TODO:
        > Weights for overall in motion index
        > Only include pass plays in dataset
        > Correlate player and team in-motion route rate to offensive pass success metrics?
            >> Doesn't really demonstrate use of in motion index though?
        > Find players with similar HSI but different profiles via correlation - demonstrate altered player profile?
        > New metrics - more boolean style metrics?
            >> Catch rate doesn't work that well
            >> Quick separation - getting 'open' by distance threshold distance in < avg. time to throw?
            >> Leverage on man?


    NOTES:
        > Purpose statements - how does receiving performance of plauyers and teams change when in-motion at the snap, which players increase their performance
                               the most when in-motion and should therefore be a focus when in-motion

    LIMITATIONS:
        > Non-contextual (e.g. in-motion more at goal line or further away?)


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
from matplotlib.offsetbox import (AnnotationBbox, DrawingArea, OffsetImage, TextArea)
import nfl_data_py as nfl
import math
from PIL import Image, ImageChops, ImageDraw
import requests
import os
from glob import glob
from tqdm import tqdm
import pickle
from scipy import stats
import random
from itertools import groupby

# Import helper functions
from helperFuncs import createField, drawFrame, downloadTeamImages, downloadPlayerImages, cropPlayerImg, relRatiosFromCounts, relRatiosFromContinuous

# =========================================================================
# Set-up
# =========================================================================

# Set a boolean value to re-analyse data or simply load from dictionary
# Default is False --- swap to True if wishing to re-run analysis
calcHeadStartIndex = True

# Set a boolean value to reproduce visuals
# Default is False --- swap to True if wishing to re-run analysis
createVisuals = True

# Set matplotlib parameters
from matplotlib import rcParams
# rcParams['font.family'] = 'sans-serif'
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
if calcHeadStartIndex:

    # Set the minimum number of routes run to be included in the dataset
    # Note this is relatively arbitrary and could be changed to check different players
    minRoutes = 50

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
    for nflId in routeRunners:

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
            # Do this by looking for pass result indicators
            if plays.loc[(plays['gameId'] == gameId) & (plays['playId'] == playId)]['passResult'].values[0] in ['C','I','S','IN','R']:

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
             'Players required minimum of 50 routes run in the Big Data Bowl 2025 dataset',
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
if calcHeadStartIndex:

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

    # Display data for in-motion routes by teams
    # Note that this includes some zeros for teams that are no longer in competition!
    for ii in range(len(teamMotionDf)):
        # Print out summary
        print(f'#{ii+1}: {teamMotionDf.iloc[ii]["teamAbbr"]} - '
              f'{"{0:.2f}".format(teamMotionDf.iloc[ii]["propInMotion"]*100)}% of passing plays with player in motion at snap')

else:

    # Load the already extracted team data
    with open(os.path.join('..', 'outputs', 'results', 'summaryTeamPlays.pkl'), 'rb') as pklFile:
        teamPlaysDict = pd.DataFrame.from_dict(pickle.load(pklFile))

    # Convert to dataframe for later analysis
    teamMotionDf = pd.DataFrame.from_dict(teamPlaysDict)

    # Calculate proportion of in-motion plays and sort by these values
    teamMotionDf['propInMotion'] = teamMotionDf['nMotionPlays'] / (teamMotionDf['nMotionPlays'] + teamMotionDf['nStationaryPlays'])
    teamMotionDf.sort_values(by='propInMotion', ascending=False, inplace=True)

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
# Collate individual player route data for calculating index
# =========================================================================

"""

Metrics included in this section from calculations:

    > targeted: a boolean of whether or not the player was vs. wasn't targeted on the play
    > reception: a boolean of whether or not the player did vs. didn't get the reception on the play
    > yards: with receptions, the total number of yards on the play
    > yac: with receptions, the number of yards after the catch on the play
    > releaseSpeed: average player velocity in the first second after the snap
    > createdSpace: ...3 yards space at least 0.5 seconds...?
    > createdSpaceEarly: ...3 yards space at least 0.5 seconds in first X seconds after snap...?
    
TO CONSIDER ADDING:
    > separationAtCatch: with receptions, the separation in yards from nearest defender at the catch
    > some sort of separation measure against man coverage?
        >> separation relative to starting separation at catch (peak of this or something?)
        >> needs to be separation CREATED, which could get muddied if not in press coverage at snap?
    > average separation in first X seconds after snap when throw is expected?
    > average route speed?

"""

# Check for running calculations
if calcHeadStartIndex:

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

    # TODO: open early idea? How many seconds? less than average?

    # Columns of route data to work through
    analyseColumns = ['stationRoutePlays', 'motionRoutePlays']

    # Loop through the identified route runners
    # -------------------------------------------------------------------------
    print('Calculating metrics across route runners...')
    for nflId in tqdm(routeData['nflId']):

        # Create a dictionary to store route result data in
        playerData = {'nflId': [], 'gameId': [], 'playId': [], 'inMotionAtSnap': [],
                      'targeted': [], 'reception': [], 'catch': [], 'yards': [], 'yac': [],
                      'createdSpace': [], 'createdSpaceEarly': [], 'separationAtCatch': [], 'releaseSpeed': []}

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

                # Get space creation variables
                # -------------------------------------------------------------------------

                # Calculate openness across play and in short time frame after snap

                # Get snap and either the pass frame or where QB is sacked/stripped
                snapFrameId = currPlayTrackingData.loc[currPlayTrackingData['frameType'] == 'SNAP']['frameId'].unique()[0]
                if 'pass_forward' in currPlayTrackingData['event'].unique():
                    passFrameId = currPlayTrackingData.loc[currPlayTrackingData['event'] == 'pass_forward']['frameId'].unique()[0]
                elif 'qb_sack' in currPlayTrackingData['event'].unique():
                    passFrameId = currPlayTrackingData.loc[currPlayTrackingData['event'] == 'qb_sack']['frameId'].unique()[0]
                elif 'qb_strip_sack' in currPlayTrackingData['event'].unique():
                    passFrameId = currPlayTrackingData.loc[currPlayTrackingData['event'] == 'qb_strip_sack']['frameId'].unique()[0]
                # TODO: open early frame id? How many seconds?

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

                # Default creating space variable to False
                createdSpace = False

                # Identify length of consecutive frames where player was considered open based on distance
                lenOpen = []
                for jj,kk in groupby(minDefDist >= openDistance):
                    if jj == True:
                        lenOpen.append(len(list(kk)))

                # If any open periods are greater than the frames threshold change the variable to True
                if any(np.array(lenOpen) >= openFramesN):
                    createdSpace = True

                # Get separation variables
                # -------------------------------------------------------------------------

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
                playerData['createdSpaceEarly'].append('TODO')
                playerData['separationAtCatch'].append(separationAtCatch)
                playerData['releaseSpeed'].append(releaseSpeed)

        # Save the player data to file
        with open(os.path.join('..', 'outputs', 'player', f'{nflId}_routeData.pkl'), 'wb') as pklFile:
            pickle.dump(playerData, pklFile)

# =========================================================================
# Run some basic checks on player data
# =========================================================================

"""

The below section is commented out but offers some basic print outs to review
the stationary vs. in-motion summary data for a given player.

"""

# # Set player Id to look up (Travis Kelce used as an example here)
# nflId = 40011
#
# # Load in players data
# with open(os.path.join('..', 'outputs', 'player', f'{nflId}_routeData.pkl'), 'rb') as pklFile:
#     playerData = pd.DataFrame.from_dict(pickle.load(pklFile))
#
# # Print out some basic descriptives
# print(f'Summary data for {players.loc[players["nflId"] == nflId]["displayName"].values[0]}')
#
# # Target rate
# # Get data to support descriptions
# summaryData = playerData.groupby(['inMotionAtSnap','targeted']).count()
# # Print out data
# print(f'{"*"*20} TARGET RATE {"*"*20}')
# print(f'Targeted on {summaryData.loc[(False,True)]["nflId"]} of {len(playerData.loc[playerData["inMotionAtSnap"] == False])} routes when stationary at snap '
#       f'({"{0:.2f}".format(summaryData.loc[(False,True)]["nflId"] / len(playerData.loc[playerData["inMotionAtSnap"] == False]) * 100)}% of stationary routes)')
# print(f'Targeted on {summaryData.loc[(True,True)]["nflId"]} of {len(playerData.loc[playerData["inMotionAtSnap"] == True])} routes when in-motion at snap '
#       f'({"{0:.2f}".format(summaryData.loc[(True,True)]["nflId"] / len(playerData.loc[playerData["inMotionAtSnap"] == True]) * 100)}% of in-motion routes)')
#
# # Reception rate
# # Get data to support descriptions
# summaryData = playerData.groupby(['inMotionAtSnap','reception']).count()
# # Print out data
# print(f'{"*"*20} RECEPTION RATE {"*"*20}')
# print(f'Received pass on {summaryData.loc[(False,True)]["nflId"]} of {len(playerData.loc[playerData["inMotionAtSnap"] == False])} routes when stationary at snap '
#       f'({"{0:.2f}".format(summaryData.loc[(False,True)]["nflId"] / len(playerData.loc[playerData["inMotionAtSnap"] == False]) * 100)}% of stationary routes)')
# print(f'Received pass on {summaryData.loc[(True,True)]["nflId"]} of {len(playerData.loc[playerData["inMotionAtSnap"] == True])} routes when in-motion at snap '
#       f'({"{0:.2f}".format(summaryData.loc[(True,True)]["nflId"] / len(playerData.loc[playerData["inMotionAtSnap"] == True]) * 100)}% of in-motion routes)')
#
# # Average yards
# # Get data to support descriptions
# summaryData = playerData.groupby(['inMotionAtSnap'])['yards'].mean()
# # Print out data
# print(f'{"*"*20} AVERAGE YARDS {"*"*20}')
# print(f'Average receiving yards when stationary at snap: {"{0:.2f}".format(summaryData[False])}')
# print(f'Average receiving yards when in-motion at snap: {"{0:.2f}".format(summaryData[True])}')
#
# # Average yards after catch
# # Get data to support descriptions
# summaryData = playerData.groupby(['inMotionAtSnap'])['yac'].mean()
# # Print out data
# print(f'{"*"*20} AVERAGE YARDS AFTER CATCH {"*"*20}')
# print(f'Average yards after catch when stationary at snap: {"{0:.2f}".format(summaryData[False])}')
# print(f'Average yards after catch when in-motion at snap: {"{0:.2f}".format(summaryData[True])}')
#
# # Average separation at catch
# # Get data to support descriptions
# summaryData = playerData.groupby(['inMotionAtSnap'])['separationAtCatch'].mean()
# # Print out data
# print(f'{"*"*20} AVERAGE SEPARATION AT CATCH {"*"*20}')
# print(f'Average separation at catch when stationary at snap: {"{0:.2f}".format(summaryData[False])}')
# print(f'Average separation at catch when in-motion at snap: {"{0:.2f}".format(summaryData[True])}')
#
# # Average release speed
# # Get data to support descriptions
# summaryData = playerData.groupby(['inMotionAtSnap'])['releaseSpeed'].mean()
# # Print out data
# print(f'{"*"*20} AVERAGE RELEASE SPEED {"*"*20}')
# print(f'Average release speed when stationary at snap: {"{0:.2f}".format(summaryData[False])}')
# print(f'Average release speed when in-motion at snap: {"{0:.2f}".format(summaryData[True])}')

# =========================================================================
# Calculate in motion index metrics
# =========================================================================

"""

Here is where the individual in motion indice values are calculated. There are
circumstances where the HSI metrics can't be calculated for a player, so the dataset
is reduced here to only include players with at least:

    > At least 5 stationary & in-motion routes
    > 1 target on stationary routes
    > 1 target on motion routes
    > 1 catch on stationary routes
    > 1 catch on motion routes

Players that don't meet these criteria are excluded at this step.

There will also be some HSI indices that can't be calculated for certaimn players
given the limited sample size of some outcomes.

"""

# Check for running calculations
if calcHeadStartIndex:

    # Settings
    # -------------------------------------------------------------------------

    # Set number of samples to take
    nSamples = 10000

    # Set-up dictionary to store index calculations
    hsiData = {'nflId': [], 'targetedHSI': [], 'targetedHSI_lower': [], 'targetedHSI_upper': [],
               'receptionHSI': [], 'receptionHSI_lower': [], 'receptionHSI_upper': [],
               'catchRateHSI': [], 'catchRateHSI_lower': [], 'catchRateHSI_upper': [],
               'yardsHSI': [], 'yardsHSI_lower': [], 'yardsHSI_upper': [],
               'yacHSI': [], 'yacHSI_lower': [], 'yacHSI_upper': [],
               'separationAtCatchHSI': [], 'separationAtCatchHSI_lower': [], 'separationAtCatchHSI_upper': [],
               'releaseSpeedHSI': [], 'releaseSpeedHSI_lower': [], 'releaseSpeedHSI_upper': [],
               }

    # Loop through route runners
    # -------------------------------------------------------------------------
    for nflId in tqdm(routeData['nflId']):

        # Load in the player route data and convert to dataframe for ease of use
        with open(os.path.join('..', 'outputs', 'player', f'{nflId}_routeData.pkl'), 'rb') as pklFile:
            playerData = pd.DataFrame.from_dict(pickle.load(pklFile))

        # Get data to run checks for including player
        # If any of these fail the player won't meet the criteria anyway
        try:
            # Get data
            stationaryRoutes = playerData.groupby('inMotionAtSnap').count().loc[False]['nflId']
            inMotionRoutes = playerData.groupby('inMotionAtSnap').count().loc[True]['nflId']
            stationaryTarget = playerData.groupby(['inMotionAtSnap', 'targeted']).count().loc[(False, True)]['nflId']
            inMotionTarget = playerData.groupby(['inMotionAtSnap', 'targeted']).count().loc[(True, True)]['nflId']
            stationaryCatch = playerData.groupby(['inMotionAtSnap', 'reception']).count().loc[(False, True)]['nflId']
            inMotionCatch = playerData.groupby(['inMotionAtSnap', 'reception']).count().loc[(True, True)]['nflId']
            # Check to keep player
            if all([
                stationaryRoutes >= 5, inMotionRoutes >= 5,
                stationaryTarget > 0, inMotionTarget > 0,
                stationaryCatch > 0, inMotionCatch > 0
            ]):
                keepPlayer = True
            else:
                keepPlayer = False
        except:
            keepPlayer = False

        # Determine whether to include player
        if keepPlayer:

            # Set player Id in dictionary
            hsiData['nflId'].append(nflId)

            # Get summary data to support calculations
            playerCountData = playerData.groupby(['inMotionAtSnap','reception']).count()

            # Target rate boost
            # -------------------------------------------------------------------------
            try:

                # Get target vs. non-targeted numbers from stationary vs. in-motion routes
                targetStationary = playerData.groupby(['inMotionAtSnap', 'targeted']).count().loc[(False,True)]['nflId']
                nonTargetStationary = playerData.groupby(['inMotionAtSnap', 'targeted']).count().loc[(False, False)]['nflId']
                targetInMotion = playerData.groupby(['inMotionAtSnap', 'targeted']).count().loc[(True, True)]['nflId']
                nonTargetInMotion = playerData.groupby(['inMotionAtSnap', 'targeted']).count().loc[(True, False)]['nflId']

                # Calculate relative ratios for metric from counts
                hsiMetric, hsiLower, hsiUpper = relRatiosFromCounts((targetStationary,nonTargetStationary),
                                                                    (targetInMotion,nonTargetInMotion),
                                                                    (nflId+123, nflId+321), nSamples)

                # Store values in dictionary
                hsiData['targetedHSI'].append(hsiMetric)
                hsiData['targetedHSI_lower'].append(hsiLower)
                hsiData['targetedHSI_upper'].append(hsiUpper)

            except:

                # No data available so set as nan
                hsiData['targetedHSI'].append(np.nan)
                hsiData['targetedHSI_lower'].append(np.nan)
                hsiData['targetedHSI_upper'].append(np.nan)

            # Reception rate boost
            # -------------------------------------------------------------------------
            try:

                # Get receptions vs. non-reception numbers from stationary vs. in-motion routes
                receptionStationary = playerData.groupby(['inMotionAtSnap', 'reception']).count().loc[(False, True)]['nflId']
                nonReceptionStationary = playerData.groupby(['inMotionAtSnap', 'reception']).count().loc[(False, False)]['nflId']
                receptionInMotion = playerData.groupby(['inMotionAtSnap', 'reception']).count().loc[(True, True)]['nflId']
                nonReceptionInMotion = playerData.groupby(['inMotionAtSnap', 'reception']).count().loc[(True, False)]['nflId']

                # Calculate relative ratios for metric from counts
                hsiMetric, hsiLower, hsiUpper = relRatiosFromCounts((receptionStationary, nonReceptionStationary),
                                                                    (receptionInMotion, nonReceptionInMotion),
                                                                    (nflId + 1234, nflId + 4321), nSamples)

                # Store values in dictionary
                hsiData['receptionHSI'].append(hsiMetric)
                hsiData['receptionHSI_lower'].append(hsiLower)
                hsiData['receptionHSI_upper'].append(hsiUpper)

            except:

                # No data available so set as nan
                hsiData['receptionHSI'].append(np.nan)
                hsiData['receptionHSI_lower'].append(np.nan)
                hsiData['receptionHSI_upper'].append(np.nan)

            # Catch rate boos
            # -------------------------------------------------------------------------
            try:

                # Get mean & SD for yards from stationary vs. in-motion routes
                catchStationary = playerData.groupby(['inMotionAtSnap', 'targeted', 'reception']).count().loc[(False, True, True)]['nflId']
                noCatchStationary = playerData.groupby(['inMotionAtSnap', 'targeted', 'reception']).count().loc[(False, True, False)]['nflId']
                catchInMotion = playerData.groupby(['inMotionAtSnap', 'targeted', 'reception']).count().loc[(True, True, True)]['nflId']
                noCatchInMotion = playerData.groupby(['inMotionAtSnap', 'targeted', 'reception']).count().loc[(True, True, False)]['nflId']

                # Calculate relative ratios for metric from counts
                hsiMetric, hsiLower, hsiUpper = relRatiosFromCounts((catchStationary, noCatchStationary),
                                                                    (catchInMotion, noCatchInMotion),
                                                                    (nflId + 12345, nflId + 54321), nSamples)

                # Store values in dictionary
                hsiData['catchRateHSI'].append(hsiMetric)
                hsiData['catchRateHSI_lower'].append(hsiLower)
                hsiData['catchRateHSI_upper'].append(hsiUpper)

            except:

                # No data available so set as nan
                hsiData['catchRateHSI'].append(np.nan)
                hsiData['catchRateHSI_lower'].append(np.nan)
                hsiData['catchRateHSI_upper'].append(np.nan)

            # Yards boost
            # -------------------------------------------------------------------------
            try:

                # Get mean & SD for yards from stationary vs. in-motion routes
                yardsStationaryMu = playerData.groupby(['inMotionAtSnap']).mean()['yards'].loc[False]
                yardsStationarySigma = playerData.groupby(['inMotionAtSnap']).std()['yards'].loc[False]
                yardsInMotionMu = playerData.groupby(['inMotionAtSnap']).mean()['yards'].loc[True]
                yardsInMotionSigma = playerData.groupby(['inMotionAtSnap']).std()['yards'].loc[True]

                # Calculate relative ratios for metric from continuous values
                hsiMetric, hsiLower, hsiUpper = relRatiosFromContinuous((yardsStationaryMu, yardsStationarySigma),
                                                                        (yardsInMotionMu, yardsInMotionSigma),
                                                                        (nflId + 345, nflId + 543), nSamples)

                # Store values in dictionary
                hsiData['yardsHSI'].append(hsiMetric)
                hsiData['yardsHSI_lower'].append(hsiLower)
                hsiData['yardsHSI_upper'].append(hsiUpper)

            except:

                # No data available so set as nan
                hsiData['yardsHSI'].append(np.nan)
                hsiData['yardsHSI_lower'].append(np.nan)
                hsiData['yardsHSI_upper'].append(np.nan)

            # YAC boost
            # -------------------------------------------------------------------------
            try:

                # Get mean & SD for yards from stationary vs. in-motion routes
                yacStationaryMu = playerData.groupby(['inMotionAtSnap']).mean()['yac'].loc[False]
                yacStationarySigma = playerData.groupby(['inMotionAtSnap']).std()['yac'].loc[False]
                yacInMotionMu = playerData.groupby(['inMotionAtSnap']).mean()['yac'].loc[True]
                yacInMotionSigma = playerData.groupby(['inMotionAtSnap']).std()['yac'].loc[True]

                # Calculate relative ratios for metric from continuous values
                hsiMetric, hsiLower, hsiUpper = relRatiosFromContinuous((yacStationaryMu, yacStationarySigma),
                                                                        (yacInMotionMu, yacInMotionSigma),
                                                                        (nflId + 2345, nflId + 5432), nSamples)

                # Store values in dictionary
                hsiData['yacHSI'].append(hsiMetric)
                hsiData['yacHSI_lower'].append(hsiLower)
                hsiData['yacHSI_upper'].append(hsiUpper)

            except:

                # No data available so set as nan
                hsiData['yacHSI'].append(np.nan)
                hsiData['yacHSI_lower'].append(np.nan)
                hsiData['yacHSI_upper'].append(np.nan)

            # Separation at catch boost
            # -------------------------------------------------------------------------
            try:

                # Get mean & SD for yards from stationary vs. in-motion routes
                separationStationaryMu = playerData.groupby(['inMotionAtSnap']).mean()['separationAtCatch'].loc[False]
                separationStationarySigma = playerData.groupby(['inMotionAtSnap']).std()['separationAtCatch'].loc[False]
                separationInMotionMu = playerData.groupby(['inMotionAtSnap']).mean()['separationAtCatch'].loc[True]
                separationInMotionSigma = playerData.groupby(['inMotionAtSnap']).std()['separationAtCatch'].loc[True]

                # Calculate relative ratios for metric from continuous values
                hsiMetric, hsiLower, hsiUpper = relRatiosFromContinuous((separationStationaryMu, separationStationarySigma),
                                                                        (separationInMotionMu, separationInMotionSigma),
                                                                        (nflId + 456, nflId + 654), nSamples)

                # Store values in dictionary
                hsiData['separationAtCatchHSI'].append(hsiMetric)
                hsiData['separationAtCatchHSI_lower'].append(hsiLower)
                hsiData['separationAtCatchHSI_upper'].append(hsiUpper)

            except:

                # No data available so set as nan
                hsiData['separationAtCatchHSI'].append(np.nan)
                hsiData['separationAtCatchHSI_lower'].append(np.nan)
                hsiData['separationAtCatchHSI_upper'].append(np.nan)

            # Release speed boost
            # -------------------------------------------------------------------------
            try:

                # Get mean & SD for yards from stationary vs. in-motion routes
                releaseStationaryMu = playerData.groupby(['inMotionAtSnap']).mean()['releaseSpeed'].loc[False]
                releaseStationarySigma = playerData.groupby(['inMotionAtSnap']).std()['releaseSpeed'].loc[False]
                releaseInMotionMu = playerData.groupby(['inMotionAtSnap']).mean()['releaseSpeed'].loc[True]
                releaseInMotionSigma = playerData.groupby(['inMotionAtSnap']).std()['releaseSpeed'].loc[True]

                # Calculate relative ratios for metric from continuous values
                hsiMetric, hsiLower, hsiUpper = relRatiosFromContinuous((releaseStationaryMu, releaseStationarySigma),
                                                                        (releaseInMotionMu, releaseInMotionSigma),
                                                                        (nflId + 4567, nflId + 7654), nSamples)

                # Store values in dictionary
                hsiData['releaseSpeedHSI'].append(hsiMetric)
                hsiData['releaseSpeedHSI_lower'].append(hsiLower)
                hsiData['releaseSpeedHSI_upper'].append(hsiUpper)

            except:

                # No data available so set as nan
                hsiData['releaseSpeedHSI'].append(np.nan)
                hsiData['releaseSpeedHSI_lower'].append(np.nan)
                hsiData['releaseSpeedHSI_upper'].append(np.nan)

    # Convert HSI to dataframe
    hsiDf = pd.DataFrame.from_dict(hsiData)



# =========================================================================
# Visualise the motion in a motion route play
# =========================================================================

# The sample play comes from Travis Kelce (40011)
nflId = 40011

# Get the first game and play Id from Kelce for an in-motion route
# TODO: below gives a play that doesn't meet criteria --- check earlier code
# gameId, playId = routeData.loc[routeData['nflId'] == nflId,]['motionRoutePlays'].values[0][0]
gameId = 2022091110
playId = 2720
weekNo = games.loc[games['gameId'] == gameId]['week'].values[0]

# Get the tracking data for the play
playTracking = tracking[f'week{weekNo}'].loc[
    (tracking[f'week{weekNo}']['gameId'] == gameId) &
    (tracking[f'week{weekNo}']['playId'] == playId)]

# Get the play description
playDesc = plays.loc[(plays['gameId'] == gameId) &
                     (plays['playId'] == playId)]

# Get the line of scrimmage and down markers
lineOfScrimmage = playDesc['absoluteYardlineNumber'].to_numpy()[0]
if playTracking['playDirection'].unique()[0] == 'left':
    # Get the yards to go and invert for field direction
    firstDownMark = lineOfScrimmage - playDesc['yardsToGo'].to_numpy()[0]
else:
    # Use standard values for right directed play
    firstDownMark = lineOfScrimmage + playDesc['yardsToGo'].to_numpy()[0]

# Get home and away teams for game
homeTeam = games.loc[games['gameId'] == gameId,]['homeTeamAbbr'].values[0]
awayTeam = games.loc[games['gameId'] == gameId,]['visitorTeamAbbr'].values[0]

# Visualise the play at the snap
# -------------------------------------------------------------------------

# Create the field to plot on
fieldFig, fieldAx = plt.subplots(figsize=(14, 6.5))
createField(fieldFig, fieldAx,
            lineOfScrimmage = lineOfScrimmage, firstDownMark = firstDownMark,
            homeTeamAbbr = homeTeam, awayTeamAbbr = awayTeam, teamData = teamData)

# Draw the play frame at the snap
snapFrameId = playTracking.loc[playTracking['frameType'] == 'SNAP',]['frameId'].unique()[0]

# Draw the frame
drawFrame(snapFrameId, homeTeam, awayTeam, teamData,
          playTracking, 'pos',
          lineOfScrimmage = lineOfScrimmage,
          firstDownMark = firstDownMark)

# Animate play from line set to end
# -------------------------------------------------------------------------

# Create the field to plot on
fieldFig, fieldAx = plt.subplots(figsize=(14, 6.5))
createField(fieldFig, fieldAx,
            lineOfScrimmage=lineOfScrimmage, firstDownMark=firstDownMark,
            homeTeamAbbr=homeTeam, awayTeamAbbr=awayTeam, teamData=teamData)

# Identify the frame range from play
lineSetFrameId = playTracking.loc[playTracking['event'] == 'line_set',]['frameId'].unique()[0]
endFrameId = playTracking['frameId'].unique().max()

# # Set route runner Id (Kelce)
# routeRunnerId = 40011

# Run animation function
anim = animation.FuncAnimation(fieldFig, drawFrame,
                               frames=range(lineSetFrameId, endFrameId + 1), repeat=False,
                               fargs=(homeTeam, awayTeam, teamData, playTracking, 'pos',
                                      None, None, None, #routeRunnerId,
                                      None, lineOfScrimmage, firstDownMark))

# Write to GIF file
gifWriter = animation.PillowWriter(fps=60)
anim.save(os.path.join('..', 'outputs', 'gif', f'samplePlay_DiggsMotion_game-{gameId}_play-{playId}.gif'),
          dpi=150, writer=gifWriter)

# Close figure after pause to avoid error
plt.pause(1)
plt.close()

# =========================================================================
# Explore targeting rate in motion vs. stationary routes for a player
# =========================================================================

# TODO: up to here...














# %% ---------- end of headStartIndex.py ---------- %% #

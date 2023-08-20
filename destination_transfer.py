import os
import pandas as pd
import numba as nb
import numpy as np
import re
import ast
from time import time
from tqdm import tqdm

def load_transactions(transactions_path, start_time, end_time, allPaypointDf):
    """
    Load and process transaction data within a specified time range.

    Args:
        transactions_path (str): Path to the transaction data file.
        start_time (str): Start time of the desired time range (e.g 2020-02-01 04:00:00).
        end_time (str): End time of the desired time range (e.g 2020-02-02 03:59:59).

    Returns:
        pd.DataFrame: Processed transaction data within the specified time range.

    """
    # Load transaction data from the specified file using pandas
    chunksize=100000
    usersDf_iterator = pd.read_csv(transactions_path, sep='#', engine='python', usecols=[0,1,2], chunksize=chunksize)
    usersDf = []
    for df in tqdm(usersDf_iterator):
        df.columns = ['cardID', 'date', 'DPAYPOINT']
        
        # Convert the 'date' column to datetime format
        df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%dT%H:%M:%S")
        
        # Filter transactions within the specified time range
        df = df[(df['date'] > start_time) & (df['date'] < end_time)]
        usersDf.append(df)
    usersDf = pd.concat(usersDf,ignore_index=True)
    print('Full dataset size: ', len(usersDf))

    # Count the occurrences of each cardID and filter out single transactions
    v = usersDf.cardID.value_counts()
    usersDf = usersDf[usersDf['cardID'].isin(v.index[v.gt(1)])]

    # Merge with additional data (allPaypointDf) based on 'DPAYPOINT'
    usersDf = pd.merge(left=usersDf, right=allPaypointDf, how='inner', left_on='DPAYPOINT', right_on='DPAYPOINT')

    # Remove rows with missing 'LATITUD' values
    usersDf = usersDf[usersDf['LATITUD'].notna()]

    # Drop duplicate transactions based on 'cardID' and 'date'
    usersDf = usersDf.drop_duplicates(subset=['cardID', 'date'], keep="first")
    print('Dataset size after removing the single transactions: ', len(usersDf))

    # Sort the DataFrame by 'cardID' and 'date'
    usersDf = usersDf.sort_values(['cardID', 'date'], ascending=[True, True]).reset_index(drop=True)

    return usersDf


def load(station_path, stop_path, transactions_path, start_time, end_time, lines_path, near_stops_path):
    """
    Load and process various data sources related to transit stops, lines, and transactions.

    Args:
        station_path (str): Path to the rail station data file.
        stop_path (str): Path to the bus stop data file.
        transactions_path (str): Path to the transactions data file.
        start_time (str or datetime): Start time of the desired time range for transactions.
        end_time (str or datetime): End time of the desired time range for transactions.
        lines_path (str): Path to the lines data file.
        near_stops_path (str): Path to the near stops data file.

    Returns:
        tuple: A tuple containing various processed DataFrames including:
            - linesRailDf: DataFrame containing rail lines information.
            - nearStopDf: DataFrame containing information about nearby stops.
            - usersDf: DataFrame containing processed transaction data within the specified time range.
            - allUniqueStopDf: DataFrame containing unique stop information.
            - allStopWithLinesOnly: DataFrame containing stop information with lines.
            - allStopWithLines: DataFrame containing all stops with associated lines.

    """
    # Load rail station data and bus stop data, adding a 'type' column
    railStopDf = pd.read_csv(station_path, usecols=['LATITUD', 'LONGITUD', 'DPAYPOINT', 'DENOMINAPARADA', 'CODIGOMUNICIPIO'])
    railStopDf['type'] = 0

    busStopDf = pd.read_csv(stop_path, usecols=['LATITUD', 'LONGITUD', 'DPAYPOINT', 'IDLINEA', 'CODIGOMUNICIPIO'])
    busStopDf = busStopDf[busStopDf['IDLINEA'].notna()]
    busStopDf['type'] = 1

    # Concatenate rail and bus stop data, create a unique list of stops, and assign IDs
    allStopDf = pd.concat([railStopDf, busStopDf], ignore_index=True)
    allUniqueStopDf = allStopDf.drop_duplicates(subset=['LATITUD','LONGITUD','type'], keep="first")
    allUniqueStopDf = allUniqueStopDf.reset_index(drop=True)
    allUniqueStopDf['IDSTOP'] = allUniqueStopDf.index

    # Merge stop data with unique stop data
    allPaypointDf = pd.merge(allStopDf, allUniqueStopDf[['LATITUD', 'LONGITUD', 'IDSTOP']], on=['LATITUD', 'LONGITUD'], how='left')
    allPaypointDf = allPaypointDf.sort_values(['IDSTOP'], ascending=[True])

    # Load transactions data
    if os.path.isfile(os.path.join(os.getcwd(), "intermediate", f"{start_time[:10]}_filtered_data.txt")):
        usersDf =  pd.read_csv(os.path.join(os.getcwd(), "intermediate", f"{start_time[:10]}_filtered_data.txt"), sep=';')
        usersDf['date'] = pd.to_datetime(usersDf['date'], format="%Y-%m-%d %H:%M:%S")
        print("Loaded checkpoint of usersDf")
    else:
        usersDf = load_transactions(transactions_path, start_time, end_time, allPaypointDf)
        usersDf.to_csv(os.path.join(os.getcwd(), "intermediate", f"{start_time[:10]}_filtered_data.txt"), sep=";", index=False)
        print("Created checkpoint of usersDf for faster processing")

    # Load lines data and process bus lines
    linesRailDf = pd.read_csv(lines_path, names=["DENOMINAPARADA", "lines"])
    for row in range(len(linesRailDf)):
        linesRailDf.iat[row,1] = linesRailDf.iat[row,1].split(";")

    linesBusDf = allPaypointDf[allPaypointDf['type'] == 1].groupby('IDSTOP')['IDLINEA'].apply(list).to_frame()
    linesBusDf = linesBusDf.rename(columns = {'IDLINEA':'lines'})
    linesBusDf = linesBusDf.reset_index()

    # Merge lines data with stop data
    merged_linesRailDf = pd.merge(linesRailDf, allPaypointDf[['IDSTOP', 'DENOMINAPARADA']], how='left', right_on='DENOMINAPARADA', left_on='DENOMINAPARADA')
    allStopWithLinesOnly = pd.concat([merged_linesRailDf, linesBusDf], ignore_index=True)
    allStopWithLines = pd.merge(left=allPaypointDf, right=allStopWithLinesOnly, how='left', left_on='IDSTOP', right_on='IDSTOP')

    # Load and process data about nearby stops
    nearStopDf = pd.read_csv(near_stops_path)
    nearStopDf['stop'] = nearStopDf['stop'].str.strip("[]").astype(float)

    return linesRailDf, nearStopDf, usersDf, allUniqueStopDf, allStopWithLinesOnly, allStopWithLines


def distance_calculate(lat1_input, lat2_input, lon1_input, lon2_input):
    """
    Calculate the great-circle distance between two points on the Earth's surface.

    Args:
        lat1_input (float): Latitude of the first point in degrees.
        lat2_input (float): Latitude of the second point in degrees.
        lon1_input (float): Longitude of the first point in degrees.
        lon2_input (float): Longitude of the second point in degrees.

    Returns:
        float: The calculated distance in meters.

    """
    # Convert latitude and longitude values from degrees to radians
    lon1 = np.radians(lon1_input)
    lon2 = np.radians(lon2_input)
    lat1 = np.radians(lat1_input)
    lat2 = np.radians(lat2_input)
        
    # Calculate differences in longitude and latitude
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Calculate intermediate terms for haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2

    # Calculate the central angle (angular separation) using the haversine formula
    c = 2 * np.arcsin(np.sqrt(a))

    # Earth's radius in meters
    r = 6371000

    # Calculate and return the distance using the spherical law of cosines formula
    return c * r


def isin_list(list1, list2):
    """
    Check if any element of list1 is present in list2.

    Args:
        list1 (list or any): The first list or element.
        list2 (list or any): The second list or element.

    Returns:
        bool: True if any element in list1 is found in list2, False otherwise.

    """
    # Ensure both inputs are treated as lists
    if not isinstance(list1, list):
        list1 = [list1]
    if not isinstance(list2, list):
        list2 = [list2]

    isin = False
    for item in list1:
        if item in list2:
            isin = True
            break
            
    return isin


# 1. Rail-Rail
def transferRail(record, linesRailDf, nearStopDf, allUniqueStopDf, allStopWithLinesOnly):
    """
    Check if a rail transfer is feasible based on the provided record and related dataframes.

    Args:
        record (pd.Series): A pandas Series representing the record data.
        linesRailDf (pd.DataFrame): DataFrame containing rail lines data.
        nearStopDf (pd.DataFrame): DataFrame containing nearby stop data.
        allUniqueStopDf (pd.DataFrame): DataFrame containing information about all unique stops.
        allStopWithLinesOnly (pd.DataFrame): DataFrame containing stop data with associated lines.

    Returns:
        tuple: A tuple containing transfer (bool) and dest_stop (int) indicating transfer feasibility and destination stop.
    """

    # Convert date column to datetime
    recordDf = record.copy()  # Make a copy of the input record to avoid unintended modifications
    recordDf["date"] = pd.to_datetime(recordDf["date"])

    # Calculate time difference in minutes
    actualTimeDiff = (recordDf.iloc[1, recordDf.columns.get_loc("date")] - recordDf.iloc[0, recordDf.columns.get_loc("date")]).seconds / 60

    # Retrieve line lists
    currentLineList = linesRailDf[linesRailDf.DENOMINAPARADA == recordDf.iloc[0, recordDf.columns.get_loc("DENOMINAPARADA")]].iat[0, linesRailDf.columns.get_loc("lines")]
    nextLineList = linesRailDf[linesRailDf.DENOMINAPARADA == recordDf.iloc[1, recordDf.columns.get_loc("DENOMINAPARADA")]].iat[0, linesRailDf.columns.get_loc("lines")]

    # Retrieve coordinates
    lat1 = recordDf.iat[0, recordDf.columns.get_loc("LATITUD")]
    lon1 = recordDf.iat[0, recordDf.columns.get_loc("LONGITUD")]
    lat2 = recordDf.iat[1, recordDf.columns.get_loc("LATITUD")]
    lon2 = recordDf.iat[1, recordDf.columns.get_loc("LONGITUD")]

    # Calculate distance
    dist = distance_calculate(lat1, lat2, lon1, lon2)

    transfer = False
    dest_stop = -2
    maxTime = 0

    if isin_list(currentLineList, nextLineList):
        maxTime = dist / 500 + 5 + 10
        if actualTimeDiff < maxTime:
            transfer = True
    else:
        bufferZoneList = nearStopDf[nearStopDf['stop'] == recordDf.iloc[1, recordDf.columns.get_loc("IDSTOP")]].iat[0, nearStopDf.columns.get_loc("near_stops")]
        bufferZoneList = ast.literal_eval(bufferZoneList)

        for station in bufferZoneList:
            station = float(station)
            typeOfTransport = allUniqueStopDf[allUniqueStopDf['IDSTOP'] == station].iat[0, allUniqueStopDf.columns.get_loc("type")]

            if typeOfTransport == 0:
                currentCandLineDf = allStopWithLinesOnly[allStopWithLinesOnly.IDSTOP == station]
                if len(currentCandLineDf) > 0:
                    currentCandList = currentCandLineDf.iat[0, currentCandLineDf.columns.get_loc("lines")]

                    if isin_list(currentCandList, nextLineList):
                        dest_stop = station
                        cand_lat2 = allUniqueStopDf[allUniqueStopDf.IDSTOP == station].iat[0, allUniqueStopDf.columns.get_loc("LATITUD")]
                        cand_lon2 = allUniqueStopDf[allUniqueStopDf.IDSTOP == station].iat[0, allUniqueStopDf.columns.get_loc("LONGITUD")]
                        can_dist = distance_calculate(lat1, cand_lat2, lon1, cand_lon2)
                        walking_dist = distance_calculate(cand_lat2, lat2, cand_lon2, lon2)
                        maxTime = (can_dist / 500) + 5 + (walking_dist / 50) + 10
                    else:
                        maxTime = dist / 500 + 5 + 20

                    if actualTimeDiff < maxTime:
                        transfer = True
                        break

    return transfer, dest_stop


# 2. Bus-Bus
def transferBus(record, nearStopDf, allUniqueStopDf, allStopWithLinesOnly):
    """
    Check if a bus transfer is feasible based on the provided record and related dataframes.

    Args:
        record (pd.Series): A pandas Series representing the record data.
        nearStopDf (pd.DataFrame): DataFrame containing nearby stop data.
        allUniqueStopDf (pd.DataFrame): DataFrame containing information about all unique stops.
        allStopWithLinesOnly (pd.DataFrame): DataFrame containing stop data with associated lines.

    Returns:
        tuple: A tuple containing transfer (bool) and dest_stop (int) indicating transfer feasibility and destination stop.
    """

    transfer = False
    dest_stop = -2
    recordDf = record.copy()  # Make a copy of the input record to avoid unintended modifications
    recordDf["date"] = pd.to_datetime(recordDf["date"])

    if recordDf.iat[0, recordDf.columns.get_loc("IDLINEA")] != recordDf.iat[1, recordDf.columns.get_loc("IDLINEA")]:
        actualTimeDiff = (recordDf.iloc[1, recordDf.columns.get_loc("date")] - recordDf.iloc[0, recordDf.columns.get_loc("date")]).seconds / 60
        currentLine = recordDf.iloc[0, recordDf.columns.get_loc("IDLINEA")]
        lat1 = recordDf.iat[0, recordDf.columns.get_loc("LATITUD")]
        lon1 = recordDf.iat[0, recordDf.columns.get_loc("LONGITUD")]
        lat2 = recordDf.iat[1, recordDf.columns.get_loc("LATITUD")]
        lon2 = recordDf.iat[1, recordDf.columns.get_loc("LONGITUD")]

        bufferZoneList = nearStopDf[nearStopDf['stop'] == recordDf.iloc[1, recordDf.columns.get_loc("IDSTOP")]].iat[0, nearStopDf.columns.get_loc("near_stops")]
        bufferZoneList = ast.literal_eval(bufferZoneList)

        for stop in bufferZoneList:
            stop = float(stop)
            typeOfTransport = allUniqueStopDf[allUniqueStopDf['IDSTOP'] == stop].iat[0, allUniqueStopDf.columns.get_loc("type")]

            if typeOfTransport == 1:
                candStopBusLineDf = allStopWithLinesOnly[allStopWithLinesOnly.IDSTOP == stop]

                if len(candStopBusLineDf) > 0:
                    candStopBusLineList = candStopBusLineDf.iat[0, candStopBusLineDf.columns.get_loc("lines")]

                    if currentLine in candStopBusLineList:
                        dest_stop = stop
                        cand_lat2 = allUniqueStopDf[allUniqueStopDf.IDSTOP == stop].iat[0, allUniqueStopDf.columns.get_loc("LATITUD")]
                        cand_lon2 = allUniqueStopDf[allUniqueStopDf.IDSTOP == stop].iat[0, allUniqueStopDf.columns.get_loc("LONGITUD")]
                        can_dist = distance_calculate(lat1, cand_lat2, lon1, cand_lon2)
                        walking_dist = distance_calculate(cand_lat2, lat2, cand_lon2, lon2)
                        maxTime = (can_dist / 250) + 10 + (walking_dist / 50)

                        if actualTimeDiff < maxTime:
                            transfer = True
                            break

    return transfer, dest_stop


# 3 rail -> bus
def transferRailBus(record, linesRailDf, nearStopDf, allUniqueStopDf, allStopWithLinesOnly):
    """
    Check if a rail-to-bus transfer is feasible based on the provided record and related dataframes.

    Args:
        record (pd.Series): A pandas Series representing the record data.
        linesRailDf (pd.DataFrame): DataFrame containing rail lines data.
        nearStopDf (pd.DataFrame): DataFrame containing nearby stop data.
        allUniqueStopDf (pd.DataFrame): DataFrame containing information about all unique stops.
        allStopWithLinesOnly (pd.DataFrame): DataFrame containing stop data with associated lines.

    Returns:
        tuple: A tuple containing transfer (bool) and dest_stop (int) indicating transfer feasibility and destination stop.
    """

    transfer = False
    dest_stop = -2
    recordDf = record.copy()  # Make a copy of the input record to avoid unintended modifications
    recordDf["date"] = pd.to_datetime(recordDf["date"])

    actualTimeDiff = (recordDf.iloc[1, recordDf.columns.get_loc("date")] - recordDf.iloc[0, recordDf.columns.get_loc("date")]).seconds / 60
    currentLineList = linesRailDf[linesRailDf.DENOMINAPARADA == recordDf.iloc[0, recordDf.columns.get_loc("DENOMINAPARADA")]].iat[0, linesRailDf.columns.get_loc("lines")]
    lat1 = recordDf.iat[0, recordDf.columns.get_loc("LATITUD")]
    lon1 = recordDf.iat[0, recordDf.columns.get_loc("LONGITUD")]
    lat2 = recordDf.iat[1, recordDf.columns.get_loc("LATITUD")]
    lon2 = recordDf.iat[1, recordDf.columns.get_loc("LONGITUD")]

    nextBoardingStopID = recordDf.iat[0, recordDf.columns.get_loc("IDSTOP")]
    bufferZoneList = nearStopDf[nearStopDf['stop'] == nextBoardingStopID].iat[0, nearStopDf.columns.get_loc("near_stops")]
    bufferZoneList = ast.literal_eval(bufferZoneList)

    for station in bufferZoneList:
        station = float(station)
        typeOfTransport = allUniqueStopDf[allUniqueStopDf['IDSTOP'] == station].iat[0, allUniqueStopDf.columns.get_loc("type")]

        if typeOfTransport == 0:
            candStationLineDf = allStopWithLinesOnly[allStopWithLinesOnly.IDSTOP == station]

            if len(candStationLineDf) > 0:
                candStationLineList = candStationLineDf.iat[0, candStationLineDf.columns.get_loc("lines")]

                if isin_list(currentLineList, candStationLineList):
                    dest_stop = station
                    cand_lat2 = allUniqueStopDf[allUniqueStopDf.IDSTOP == station].iat[0, allUniqueStopDf.columns.get_loc("LATITUD")]
                    cand_lon2 = allUniqueStopDf[allUniqueStopDf.IDSTOP == station].iat[0, allUniqueStopDf.columns.get_loc("LONGITUD")]
                    can_dist = distance_calculate(lat1, cand_lat2, lon1, cand_lon2)
                    walking_dist = distance_calculate(cand_lat2, lat2, cand_lon2, lon2)
                    maxTime = (can_dist / 500) + (walking_dist / 50) + 10 + 10 + 5

                    if actualTimeDiff < maxTime:
                        transfer = True
                        break

    return transfer, dest_stop



# 4 bus -> rail
def transferBusRail(record, nearStopDf, allUniqueStopDf, allStopWithLinesOnly):
    """
    Check if a bus-to-rail transfer is feasible based on the provided record and related dataframes.

    Args:
        record (pd.Series): A pandas Series representing the record data.
        nearStopDf (pd.DataFrame): DataFrame containing nearby stop data.
        allUniqueStopDf (pd.DataFrame): DataFrame containing information about all unique stops.
        allStopWithLinesOnly (pd.DataFrame): DataFrame containing stop data with associated lines.

    Returns:
        tuple: A tuple containing transfer (bool) and dest_stop (int) indicating transfer feasibility and destination stop.
    """

    transfer = False
    dest_stop = -2
    recordDf = record.copy()  # Make a copy of the input record to avoid unintended modifications
    recordDf["date"] = pd.to_datetime(recordDf["date"])

    actualTimeDiff = (recordDf.iloc[1, recordDf.columns.get_loc("date")] - recordDf.iloc[0, recordDf.columns.get_loc("date")]).seconds / 60
    currentLine = recordDf.iloc[0, 7]

    lat1 = recordDf.iat[0, recordDf.columns.get_loc("LATITUD")]
    lon1 = recordDf.iat[0, recordDf.columns.get_loc("LONGITUD")]
    lat2 = recordDf.iat[1, recordDf.columns.get_loc("LATITUD")]
    lon2 = recordDf.iat[1, recordDf.columns.get_loc("LONGITUD")]

    bufferZoneList = nearStopDf[nearStopDf['stop'] == recordDf.iloc[1, recordDf.columns.get_loc("IDSTOP")]].iat[0, nearStopDf.columns.get_loc("near_stops")]
    bufferZoneList = ast.literal_eval(bufferZoneList)

    for stop in bufferZoneList:
        stop = float(stop)
        typeOfTransport = allUniqueStopDf[allUniqueStopDf['IDSTOP'] == stop].iat[0, allUniqueStopDf.columns.get_loc("type")]

        if typeOfTransport == 1:
            candStopBusLineDf = allStopWithLinesOnly[allStopWithLinesOnly.IDSTOP == stop]

            if len(candStopBusLineDf) > 0:
                candStopBusLineList = candStopBusLineDf.iat[0, candStopBusLineDf.columns.get_loc("lines")]

                if currentLine in candStopBusLineList:
                    dest_stop = stop
                    cand_lat2 = allStopWithLines[allStopWithLines.IDSTOP == stop].iat[0, allStopWithLines.columns.get_loc("LATITUD")]
                    cand_lon2 = allStopWithLines[allStopWithLines.IDSTOP == stop].iat[0, allStopWithLines.columns.get_loc("LONGITUD")]
                    can_dist = distance_calculate(lat1, cand_lat2, lon1, cand_lon2)
                    walking_dist = distance_calculate(cand_lat2, lat2, cand_lon2, lon2)

                    maxTime = (can_dist / 250) + (walking_dist / 50)
                    if actualTimeDiff < maxTime:
                        transfer = True
                        break

    return transfer, dest_stop


def detect_transfers(usersDf, linesRailDf, nearStopDf, allUniqueStopDf, allStopWithLinesOnly, allStopWithLines):
    """
    Detect and record transfer information in the user's dataframe.

    Args:
        usersDf (pd.DataFrame): DataFrame containing user transaction data.
        linesRailDf (pd.DataFrame): DataFrame containing rail lines data.
        nearStopDf (pd.DataFrame): DataFrame containing nearby stop data.
        allUniqueStopDf (pd.DataFrame): DataFrame containing information about all unique stops.
        allStopWithLinesOnly (pd.DataFrame): DataFrame containing stop data with associated lines.
        allStopWithLines (pd.DataFrame): DataFrame containing stop data with all lines.

    Returns:
        pd.DataFrame: The input users' DataFrame with added 'transfer' and 'destination' columns.
    """
    cardID_first = usersDf.iloc[0, usersDf.columns.get_loc("cardID")]
    transferList = []
    destinationList = []

    count = 0 
    for transaction in tqdm(range(len(usersDf)-1)):
        destination = -1
        transfer = False
        cardID_second = usersDf.iloc[transaction+1, usersDf.columns.get_loc("cardID")]

        if(cardID_second != cardID_first):
            firstTripStopID = usersDf.iloc[transaction-count, usersDf.columns.get_loc("IDSTOP")]

            if (usersDf.iloc[transaction-count, usersDf.columns.get_loc("type")] == 0 and usersDf.iloc[transaction, usersDf.columns.get_loc("type")] == 0) and (firstTripStopID != usersDf.iloc[transaction, usersDf.columns.get_loc("IDSTOP")]):
                currentLineList = allStopWithLines[allStopWithLines.IDSTOP == firstTripStopID].iat[0, allStopWithLines.columns.get_loc("lines")]
                nextLineList  = allStopWithLines[allStopWithLines.IDSTOP == usersDf.iloc[transaction, usersDf.columns.get_loc("IDSTOP")]].iat[0, allStopWithLines.columns.get_loc("lines")]
                if isin_list(currentLineList, nextLineList):
                    destination = firstTripStopID
                else: 
                    nearList = ast.literal_eval(nearStopDf[nearStopDf['stop'] == firstTripStopID].iat[0, nearStopDf.columns.get_loc("near_stops")])
                    for station in nearList:
                        if allStopWithLines[allStopWithLines.IDSTOP == station].iat[0, allStopWithLines.columns.get_loc("type")] == 0:
                            currentLineList = allStopWithLines[allStopWithLines.IDSTOP == station].iat[0,allStopWithLines.columns.get_loc("lines")]
                            if isin_list(nextLineList, currentLineList):
                                destination = firstTripStopID
                                break

            if (usersDf.iloc[transaction-count, usersDf.columns.get_loc("type")] == 1 and usersDf.iloc[transaction, usersDf.columns.get_loc("type")] == 1) and (firstTripStopID != usersDf.iloc[transaction, usersDf.columns.get_loc("IDSTOP")]):
                firstLine = usersDf.iloc[transaction-count, usersDf.columns.get_loc("IDLINEA")]
                lastLine  = usersDf.iloc[transaction, usersDf.columns.get_loc("IDLINEA")]
                if firstLine == lastLine:
                    destination = firstTripStopID
                else: 
                    nearList = ast.literal_eval(nearStopDf[nearStopDf['stop'] == firstTripStopID].iat[0, nearStopDf.columns.get_loc("near_stops")])
                    for stop in nearList:
                        if allStopWithLines[allStopWithLines.IDSTOP == stop].iat[0, allStopWithLines.columns.get_loc("type")] == 1:
                            bufferLineList = allStopWithLines[allStopWithLines.IDSTOP == stop].iat[0, allStopWithLines.columns.get_loc("lines")]
                            if lastLine in bufferLineList:
                                destination = firstTripStopID
                                break

            if (usersDf.iloc[transaction-count, usersDf.columns.get_loc("type")] == 1 and usersDf.iloc[transaction, usersDf.columns.get_loc("type")] == 0):
                nextLineList  = allStopWithLines[allStopWithLines.IDSTOP == usersDf.iloc[transaction, usersDf.columns.get_loc("IDSTOP")]].iat[0, allStopWithLines.columns.get_loc("lines")]
                nearList = ast.literal_eval(nearStopDf[nearStopDf['stop'] == firstTripStopID].iat[0, nearStopDf.columns.get_loc("near_stops")])
                for station in nearList:
                    if allStopWithLines[allStopWithLines.IDSTOP == station].iat[0, allStopWithLines.columns.get_loc("type")] == 0:
                        currentLineList = allStopWithLines[allStopWithLines.IDSTOP == station].iat[0, allStopWithLines.columns.get_loc("lines")]
                        if isin_list(nextLineList, currentLineList):
                            destination = firstTripStopID
                            break

            if (usersDf.iloc[transaction-count, usersDf.columns.get_loc("type")] == 0 and usersDf.iloc[transaction, usersDf.columns.get_loc("type")] == 1):
                currentLineList = allStopWithLines[allStopWithLines.IDSTOP == usersDf.iloc[transaction-count, usersDf.columns.get_loc("IDSTOP")]].iat[0, allStopWithLines.columns.get_loc("lines")]
                nextLine = usersDf.iloc[transaction, usersDf.columns.get_loc("IDLINEA")] 
                nearList = ast.literal_eval(nearStopDf[nearStopDf['stop'] == firstTripStopID].iat[0, nearStopDf.columns.get_loc("near_stops")])
                for stop in nearList:
                    if allStopWithLines[allStopWithLines.IDSTOP == stop].iat[0, allStopWithLines.columns.get_loc("type")] == 1:
                        currentLineList = allStopWithLines[allStopWithLines.IDSTOP == stop].iat[0, allStopWithLines.columns.get_loc("lines")]
                        if isin_list(nextLine, currentLineList):               
                            destination = firstTripStopID
                            break
    
            cardID_first = cardID_second
            count = 0 
        else:
            count += 1
            record = usersDf.iloc[[transaction, transaction+1]]
            if usersDf.iloc[transaction, usersDf.columns.get_loc("type")] == 0 and usersDf.iloc[transaction+1, usersDf.columns.get_loc("type")] == 0:
                transfer, destination = transferRail(record, linesRailDf, nearStopDf, allUniqueStopDf, allStopWithLinesOnly)
            elif usersDf.iloc[transaction, usersDf.columns.get_loc("type")] == 1 and usersDf.iloc[transaction+1, usersDf.columns.get_loc("type")] == 1:
                transfer, destination = transferBus(record, nearStopDf, allUniqueStopDf, allStopWithLinesOnly)
            elif usersDf.iloc[transaction, usersDf.columns.get_loc("type")] == 0 and usersDf.iloc[transaction+1, usersDf.columns.get_loc("type")] == 1:
                transfer, destination = transferRailBus(record, linesRailDf, nearStopDf, allUniqueStopDf, allStopWithLinesOnly)
            elif usersDf.iloc[transaction, usersDf.columns.get_loc("type")] == 1 and usersDf.iloc[transaction+1, usersDf.columns.get_loc("type")] == 0:
                transfer, destination = transferBusRail(record, nearStopDf, allUniqueStopDf, allStopWithLinesOnly)

        transferList.append(transfer)
        destinationList.append(destination)

    transferList.append(False)
    destinationList.append(-2)
    usersDf.insert(9, 'transfer', transferList)
    usersDf.insert(10, 'destinaiton', destinationList)
    return usersDf


if __name__=="__main__":

    for d in range(2,9):
        t1 = time()
        # Load
        config_dict = {"station_path": os.path.join(os.getcwd(), "input", "stationInfo.csv"),
                        "stop_path": os.path.join(os.getcwd(), "input", "stopInfo.csv"),
                        "transactions_path": os.path.join(os.getcwd(),"input", "transactionData.txt"),
                        "lines_path": os.path.join(os.getcwd(), "input", "lines.csv"),
                        "near_stops_path": os.path.join(os.getcwd(), "intermediate", "near.csv"),
                        "start_time":f"2020-02-0{d} 04:00:00",
                        "end_time":f"2020-02-0{d+1} 03:59:59"}
        print(f"Loading data for {config_dict['start_time'][:10]}")
        linesRailDf, nearStopDf, usersDf, allUniqueStopDf, allStopWithLinesOnly, allStopWithLines = load(**config_dict)
        print(f"Loaded data in {round((time()-t1) / 60,1)} minutes")

        # Format to match old thesis column order
        usersDf = usersDf[["cardID","date","DPAYPOINT","DENOMINAPARADA","LATITUD","LONGITUD","type","IDLINEA","IDSTOP"]]

        # Detect transfers
        usersDf = detect_transfers(usersDf, linesRailDf, nearStopDf, allUniqueStopDf, allStopWithLinesOnly, allStopWithLines)
        
        # Save
        usersDf.to_csv(config_dict["start_time"][:10] + "_matrix.csv", encoding = 'utf-8-sig') 
        print(f"Finished {config_dict['start_time'][:10]} in {round((time()-t1) / 60, 1)} minutes")

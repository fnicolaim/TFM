import pandas as pd
import numba as nb
import numpy as np
import re
import ast

railStopDf = pd.read_csv('stationInfo.csv', usecols=['LATITUD', 'LONGITUD', 'DPAYPOINT', 'DENOMINAPARADA', 'CODIGOMUNICIPIO'])
railStopDf['type'] = 0

busStopDf = pd.read_csv('stopInfo.csv', usecols=['LATITUD', 'LONGITUD', 'DPAYPOINT', 'IDLINEA', 'CODIGOMUNICIPIO'])
busStopDf = busStopDf[busStopDf['IDLINEA'].notna()]
busStopDf['type'] = 1

allStopDf = pd.concat([railStopDf, busStopDf], ignore_index=True)
allUniqueStopDf = allStopDf.drop_duplicates(subset=['LATITUD','LONGITUD','type'], keep="first")
allUniqueStopDf = allUniqueStopDf.reset_index(drop=True)
allUniqueStopDf['IDSTOP'] = allUniqueStopDf.index
# allUniqueStopDf

allPaypointDf = pd.merge(allStopDf, allUniqueStopDf[['LATITUD', 'LONGITUD', 'IDSTOP']], on=['LATITUD', 'LONGITUD'], how='left')
allPaypointDf = allPaypointDf.sort_values(['IDSTOP'], ascending=[True])
# allPaypointDf

usersDf = pd.read_csv('transactionData.txt', sep='#', engine='python', usecols=[0,1,2])
usersDf.columns = ['cardID', 'date', 'DPAYPOINT']
usersDf['date'] = pd.to_datetime(usersDf['date'])
usersDf = usersDf[(usersDf['date'] > '2020-02-01 04:00:00') & (usersDf['date'] < '2020-02-02 03:59:59')]
print('Full dataset size: ', len(usersDf))


v = usersDf.cardID.value_counts()
usersDf = usersDf[usersDf['cardID'].isin(v.index[v.gt(1)])]

usersDf = pd.merge(left=usersDf, right=allPaypointDf, how='inner', left_on='DPAYPOINT', right_on='DPAYPOINT')
usersDf = usersDf[usersDf['LATITUD'].notna()]
usersDf = usersDf.drop_duplicates(subset=['cardID','date'], keep="first")
print('Dataset size after removing the single transactions: ', len(usersDf))

usersDf = usersDf.sort_values(['cardID', 'date'], ascending=[True, True]).reset_index(drop=True)
# usersDf

linesRailDf = pd.read_csv('lines.csv', names=["DENOMINAPARADA", "lines"])
for row in range(len(linesRailDf)):
    linesRailDf.iat[row,1] = linesRailDf.iat[row,1].split(";")

linesBusDf = allPaypointDf[allPaypointDf['type'] == 1].groupby('IDSTOP')['IDLINEA'].apply(list).to_frame()
linesBusDf = linesBusDf.rename(columns = {'IDLINEA':'lines'})
linesBusDf = linesBusDf.reset_index()

merged_linesRailDf = pd.merge(linesRailDf, allPaypointDf[['IDSTOP', 'DENOMINAPARADA']], how='left', right_on='DENOMINAPARADA', left_on='DENOMINAPARADA')
allStopWithLinesOnly = pd.concat([merged_linesRailDf, linesBusDf], ignore_index=True)
# allStopWithLinesOnly

allStopWithLines = pd.merge(left=allPaypointDf, right=allStopWithLinesOnly, how='left', left_on='IDSTOP', right_on='IDSTOP')
# allStopWithLines.head()

nearStopDf = pd.read_csv('near.csv')
nearStopDf['stop'] = nearStopDf['stop'].str.strip("[]").astype(float)
# nearStopDf.head()

def distance_calculate(lat1_input,lat2_input,lon1_input,lon2_input):
  lon1 = np.radians(lon1_input)
  lon2 = np.radians(lon2_input)
  lat1 = np.radians(lat1_input)
  lat2 = np.radians(lat2_input)
      
  dlon = lon2 - lon1
  dlat = lat2 - lat1
  a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
  c = 2 * np.arcsin(np.sqrt(a))
  r = 6371000
  return c * r


def isin_list(list1, list2):
  isin = False
  for item in range(len(list1)):
    if list1[item] in list2:
      isin = True
      break 
  return isin

# 1. Rail-Rail
def transferRail(cardID, usersDf, record):
  transfer = False
  dest_stop = -2
  maxTime = 0
  recordDf = record
  recordDf["date"] = pd.to_datetime(recordDf["date"])
  
  actualTimeDiff = (recordDf.iloc[1,1] - recordDf.iloc[0,1]).seconds/60
  currentLineList = linesRailDf[linesRailDf.DENOMINAPARADA == recordDf.iloc[0,3]].iat[0,1]
  nextLineList  = linesRailDf[linesRailDf.DENOMINAPARADA == recordDf.iloc[1,3]].iat[0,1]

  lat1 = recordDf.iloc[0,4]
  lon1 = recordDf.iloc[0,5]
  lat2 = recordDf.iloc[1,4]
  lon2 = recordDf.iloc[1,5]
  dist = distance_calculate(lat1,lat2,lon1,lon2) 

  if isin_list(currentLineList, nextLineList):
    maxTime = dist/500 + 5 + 10 
    if actualTimeDiff < maxTime:
      transfer = True

  else: 
    bufferZoneList = nearStopDf[nearStopDf['stop'] == recordDf.iloc[1,8]].iat[0,1]
    bufferZoneList = ast.literal_eval(bufferZoneList)

    for station in bufferZoneList:
      float(station)
      typeOfTransport = allUniqueStopDf[allUniqueStopDf['IDSTOP'] == station].iat[0,4]

      if typeOfTransport == 0:
        currentCandLineDf = allStopWithLinesOnly[allStopWithLinesOnly.IDSTOP == station]
        if len(currentCandLineDf) > 0:
          currentCandList = currentCandLineDf.iat[0,1]

          if isin_list(currentCandList, nextLineList):
            dest_stop = station
            cand_lat2 = allUniqueStopDf[allUniqueStopDf.IDSTOP == station].iat[0,1]
            cand_lon2 = allUniqueStopDf[allUniqueStopDf.IDSTOP == station].iat[0,2]
            can_dist = distance_calculate(lat1, cand_lat2, lon1, cand_lon2)
            walking_dist = distance_calculate(cand_lat2,lat2,cand_lon2,lon2)
            maxTime = (can_dist/500) + 5 + (walking_dist/50) + 10 

          else:
            maxTime = dist/500 + 5 + 20 

        if actualTimeDiff < maxTime:
          transfer = True
          break

  return transfer, dest_stop


# 2. Bus-Bus
def transferBus(cardID, usersDf, record):
  transfer = False
  dest_stop = -2
  recordDf = record
  recordDf["date"] = pd.to_datetime(recordDf["date"])

  if record.iat[0, 7] != record.iat[1, 7]: 
    actualTimeDiff = (recordDf.iloc[1,1] - recordDf.iloc[0,1]).seconds/60
    currentLine = recordDf.iloc[0,7]
    lat1 = recordDf.iloc[0,4]
    lon1 = recordDf.iloc[0,5]
    lat2 = recordDf.iloc[1,4]
    lon2 = recordDf.iloc[1,5]

    bufferZoneList = nearStopDf[nearStopDf['stop'] == recordDf.iloc[1,8]].iat[0,1]
    bufferZoneList = ast.literal_eval(bufferZoneList)

    for stop in bufferZoneList:
      float(stop)
      typeOfTransport = allUniqueStopDf[allUniqueStopDf['IDSTOP'] == stop].iat[0,4]

      if typeOfTransport == 1:
        candStopBusLineDf = allStopWithLinesOnly[allStopWithLinesOnly.IDSTOP == stop]
    
        if len(candStopBusLineDf) > 0:
          candStopBusLineList = candStopBusLineDf.iat[0,1]
  
          if currentLine in candStopBusLineList:
            dest_stop = stop
            cand_lat2 = allUniqueStopDf[allUniqueStopDf.IDSTOP == stop].iat[0,1]
            cand_lon2 = allUniqueStopDf[allUniqueStopDf.IDSTOP == stop].iat[0,2]
            can_dist = distance_calculate(lat1, cand_lat2, lon1, cand_lon2)
            walking_dist = distance_calculate(cand_lat2,lat2,cand_lon2,lon2)
            maxTime = (can_dist/250) + 10 + (walking_dist/50)
  

            if actualTimeDiff < maxTime:
              transfer = True
              break

  return transfer, dest_stop

def transferRailBus(cardID, usersDf, record):
  transfer = False
  dest_stop = -2
  recordDf = record
  recordDf["date"] = pd.to_datetime(recordDf["date"])

  actualTimeDiff = (recordDf.iloc[1,1] - recordDf.iloc[0,1]).seconds/60
  currentLineList = linesRailDf[linesRailDf.DENOMINAPARADA == recordDf.iloc[0,3]].iat[0,1]
  lat1 = recordDf.iloc[0,4]
  lon1 = recordDf.iloc[0,5]
  nextBoardingStopID = recordDf.iat[0,8]
  lat2 = recordDf.iat[1, 4]
  lon2 = recordDf.iat[1, 5]

  bufferZoneList = nearStopDf[nearStopDf['stop'] == nextBoardingStopID].iat[0,1]
  bufferZoneList = ast.literal_eval(bufferZoneList)

  for station in bufferZoneList:
    float(station)
    typeOfTransport = allUniqueStopDf[allUniqueStopDf['IDSTOP'] == station].iat[0,4]

    if typeOfTransport == 0:
      candStationLineDf = allStopWithLinesOnly[allStopWithLinesOnly.IDSTOP == station]

      if len(candStationLineDf) > 0:
        candStationLineList = candStationLineDf.iat[0,1]
        
        if isin_list(currentLineList, candStationLineList):
          dest_stop = station
          cand_lat2 = allUniqueStopDf[allUniqueStopDf.IDSTOP == station].iat[0,1]
          cand_lon2 = allUniqueStopDf[allUniqueStopDf.IDSTOP == station].iat[0,2]
          can_dist = distance_calculate(lat1, cand_lat2, lon1, cand_lon2)
          walking_dist = distance_calculate(cand_lat2,lat2,cand_lon2,lon2)       
          maxTime =  (can_dist/500) + (walking_dist/50) + 10 + 10 + 5

          if actualTimeDiff < maxTime:
            transfer = True
            break

  return transfer, dest_stop


def transferBusRail(cardID, usersDf, record):
  transfer = False
  dest_stop = -2
  recordDf = record
  recordDf["date"] = pd.to_datetime(recordDf["date"])

  actualTimeDiff = (recordDf.iloc[1,1] - recordDf.iloc[0,1]).seconds/60
  currentLine = recordDf.iloc[0,7]

  lat1 = recordDf.iloc[0,4]
  lon1 = recordDf.iloc[0,5]
  lat2 = recordDf.iloc[1,4]
  lon2 = recordDf.iloc[1,5]

  bufferZoneList = nearStopDf[nearStopDf['stop'] == recordDf.iloc[1,8]].iat[0,1]
  bufferZoneList = ast.literal_eval(bufferZoneList)

  for stop in bufferZoneList:
    float(stop)
    typeOfTransport = allUniqueStopDf[allUniqueStopDf['IDSTOP'] == stop].iat[0,4]

    if typeOfTransport == 1:
      candStopBusLineDf = allStopWithLinesOnly[allStopWithLinesOnly.IDSTOP == stop]

      if len(candStopBusLineDf) > 0:
        candStopBusLineList = candStopBusLineDf.iat[0,1]
      
        if currentLine in candStopBusLineList:
          dest_stop = stop
          cand_lat2 = allStopWithLines[allStopWithLines.IDSTOP == stop].iat[0,1]
          cand_lon2 = allStopWithLines[allStopWithLines.IDSTOP == stop].iat[0,2]
          can_dist = distance_calculate(lat1, cand_lat2, lon1, cand_lon2)
          walking_dist = distance_calculate(cand_lat2,lat2,cand_lon2,lon2)

          maxTime = (can_dist/250) + (walking_dist/50)
          if actualTimeDiff < maxTime:
            
            transfer = True
            break

  return transfer, dest_stop


# Destination estimation
cardID_first = usersDf.iloc[0, 0]
transferList = []
destinationList = []

count = 0 
for transaction in range(len(usersDf)-1):
  destination = -1
  transfer = False
  cardID_second = usersDf.iloc[transaction+1, 0]

  if(cardID_second != cardID_first):
    firstTripStopID = usersDf.iloc[transaction-count, 8]

    if (usersDf.iloc[transaction-count, 6] == 0 and usersDf.iloc[transaction, 6] == 0) and (firstTripStopID != usersDf.iloc[transaction, 8]):
      currentLineList = allStopWithLines[allStopWithLines.IDSTOP == firstTripStopID].iat[0,-1]
      nextLineList  = allStopWithLines[allStopWithLines.IDSTOP == usersDf.iloc[transaction,8]].iat[0,-1]
      if isin_list(currentLineList, nextLineList):
        destination = firstTripStopID
      else: 
        nearList = ast.literal_eval(nearStopDf[nearStopDf['stop'] == firstTripStopID].iat[0,1])
        for station in nearList:
          if allStopWithLines[allStopWithLines.IDSTOP == station].iat[0, 4] == 0:
            currentLineList = allStopWithLines[allStopWithLines.IDSTOP == station].iat[0,-1]
            if isin_list(nextLineList, currentLineList):
              destination = firstTripStopID
              break

    if (usersDf.iloc[transaction-count, 6] == 1 and usersDf.iloc[transaction, 6] == 1) and (firstTripStopID != usersDf.iloc[transaction, 8]):
        firstLine = usersDf.iloc[transaction-count, 7]
        lastLine  = usersDf.iloc[transaction, 7]
        if firstLine == lastLine:
          destination = firstTripStopID
        else: 
          nearList = ast.literal_eval(nearStopDf[nearStopDf['stop'] == firstTripStopID].iat[0,1])
          for stop in nearList:
            if allStopWithLines[allStopWithLines.IDSTOP == stop].iat[0, 4] == 1:
              bufferLineList = allStopWithLines[allStopWithLines.IDSTOP == stop].iat[0,-1]
              if lastLine in bufferLineList:
                destination = firstTripStopID
                break

    if (usersDf.iloc[transaction-count, 6] == 1 and usersDf.iloc[transaction, 6] == 0):
        currentLine = usersDf.iloc[transaction-count, 7]
        nextLineList  = allStopWithLines[allStopWithLines.IDSTOP == usersDf.iloc[transaction, 8]].iat[0,-1]
        nearList = ast.literal_eval(nearStopDf[nearStopDf['stop'] == firstTripStopID].iat[0,1])
        for station in nearList:
          if allStopWithLines[allStopWithLines.IDSTOP == station].iat[0, 4] == 0:
            currentLineList = allStopWithLines[allStopWithLines.IDSTOP == station].iat[0,-1]
            if isin_list(nextLineList, currentLineList):
              destination = firstTripStopID
              break

    if (usersDf.iloc[transaction-count, 6] == 0 and usersDf.iloc[transaction, 6] == 1):
        currentLineList = allStopWithLines[allStopWithLines.IDSTOP == usersDf.iloc[transaction-count, 8]].iat[0,-1]
        nextLine = usersDf.iloc[transaction, 7] 
        nearList = ast.literal_eval(nearStopDf[nearStopDf['stop'] == firstTripStopID].iat[0,1])
        for stop in nearList:
          if allStopWithLines[allStopWithLines.IDSTOP == stop].iat[0, 4] == 1:
            currentLineList = allStopWithLines[allStopWithLines.IDSTOP == stop].iat[0,-1]
            if isin_list(nextLine, currentLineList):               
              destination = firstTripStopID
              break

    
    cardID_first = cardID_second
    count = 0 
  else:
    count += 1
    record = usersDf.iloc[[transaction, transaction+1]]
    if usersDf.iloc[transaction, 6] == 0 and usersDf.iloc[transaction+1, 6] == 0:
      transfer, destination = transferRail(cardID_second, usersDf, record)
    elif usersDf.iloc[transaction, 6] == 1 and usersDf.iloc[transaction+1, 6] == 1:
      transfer, destination = transferBus(cardID_second, usersDf, record)
    elif usersDf.iloc[transaction, 6] == 0 and usersDf.iloc[transaction+1, 6] == 1:
      transfer, destination = transferRailBus(cardID_second, usersDf, record)
    elif usersDf.iloc[transaction, 6] == 1 and usersDf.iloc[transaction+1, 6] == 0:
      transfer, destination = transferBusRail(cardID_second, usersDf, record)

  transferList.append(transfer)
  destinationList.append(destination)

transferList.append(False)
destinationList.append(-2)
usersDf.insert(9, 'transfer', transferList)
usersDf.insert(10, 'destinaiton', destinationList)

usersDf.to_csv('Feb4-4AMDF.csv', encoding = 'utf-8-sig') 
print(f"Finished in {round((time()-t1) / 60,1)} (m)")

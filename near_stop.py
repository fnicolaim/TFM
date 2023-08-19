import pandas as pd
import numba as nb
import numpy as np
import csv
from time import time

@nb.njit(fastmath=True, error_model='numpy')
def find_near_stops(lat, lon, stopID):

    nearStopFullList = []
    for stops in nb.prange(len(stopID)):
        stopsList = []
        nearStopList = []
        rowList = []
        stopsList.append(stopID[stops])
        rowList.append(stopsList)

        for near_stops in nb.prange(len(stopID)):
            lon1 = np.radians(lon[near_stops])
            lon2 = np.radians(lon[stops])
            lat1 = np.radians(lat[near_stops])
            lat2 = np.radians(lat[stops])

            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            c = 2 * np.arcsin(np.sqrt(a))
            r = 6371000
            distance = c * r

            if distance < 650 and (stopID[near_stops] != stopID[stops]):
                nearStopList.append(stopID[near_stops])

        rowList.append(nearStopList)

        nearStopFullList.append(rowList)
    return nearStopFullList

if __name__=="__main__":
    t1 = time()
    # Extract
    # Step added to map web source to local source
    pd.read_excel("BD_TOP_AUTOBUSES.xlsx").to_csv("stopInfo.csv", index=False)

    railStopDf = pd.read_csv('stationInfo.csv', usecols=['LATITUD', 'LONGITUD', 'DPAYPOINT', 'DENOMINAPARADA'])\
        .drop_duplicates(subset=['LATITUD', 'LONGITUD'], keep="first")
    railStopDf['type'] = 'rail'

    busStopDf = pd.read_csv('stopInfo.csv', usecols=['LATITUD', 'LONGITUD', 'DPAYPOINT', 'IDLINEA'])
    busStopDf = busStopDf[busStopDf['IDLINEA'].notna()]
    busStopDf['type'] = 'bus'

    allUniqueStopDf = pd.concat([railStopDf, busStopDf], ignore_index=True)\
        .drop_duplicates(subset=['LATITUD','LONGITUD', 'type'], keep="first")
    allUniqueStopDf = allUniqueStopDf.reset_index(drop=True)
    allUniqueStopDf['IDSTOP'] = allUniqueStopDf.index

    # Transform
    near_stops = find_near_stops(allUniqueStopDf['LATITUD'].values,
                                allUniqueStopDf['LONGITUD'].values,
                                allUniqueStopDf['IDSTOP'].values.astype('float64'))

    # Load
    fields = ['stop', 'near_stops']
    with open("near.csv", "w", newline="") as file:
        write = csv.writer(file)
        write.writerow(fields)
        write.writerows(near_stops)
    print(f"Finished in {round(time()-t1)} (s).")
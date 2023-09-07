
"""
This code contructs the odmatrices at a municipio level, the inner legs of the trip are discarded
leaving just one row per trip.
"""

import pandas as pd
import numpy as np
import os
from time import time
from datetime import datetime, timedelta
from geopy.distance import geodesic

if __name__=="__main__":

  # Load rail station data and bus stop data, adding a 'type' column
  stationInfoDf = pd.read_csv(os.path.join(os.getcwd(), "input", "stationInfo.csv"), usecols=['LATITUD', 'LONGITUD', 'DPAYPOINT', 'DENOMINAPARADA', 'CODIGOMUNICIPIO'])
  stationInfoDf['type'] = 0

  stopInfoDf = pd.read_csv(os.path.join(os.getcwd(), "input", "stopInfo.csv"), usecols=['LATITUD', 'LONGITUD', 'DPAYPOINT', 'IDLINEA', 'CODIGOMUNICIPIO'])
  stopInfoDf = stopInfoDf[stopInfoDf['IDLINEA'].notna()]
  stopInfoDf['type'] = 1

  # DF with dpaypoints and their municipio
  stopsMunicipio = pd.concat([stationInfoDf[['CODIGOMUNICIPIO','DPAYPOINT']], stopInfoDf[['CODIGOMUNICIPIO','DPAYPOINT']]]).drop_duplicates('DPAYPOINT')

  # Concatenate rail and bus stop data, create a unique list of stops, and assign IDs
  allStopDf = pd.concat([stationInfoDf, stopInfoDf], ignore_index=True)
  allUniqueStopDf = allStopDf.drop_duplicates(subset=['LATITUD','LONGITUD','type'], keep="first")
  allUniqueStopDf = allUniqueStopDf.reset_index(drop=True)
  allUniqueStopDf['IDSTOP'] = allUniqueStopDf.index

  # Generate objective dates list
  start_date = datetime(2021, 1, 1)
  end_date = datetime(2021, 6, 30)
  date_list = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range((end_date - start_date).days + 1)]

  for date in date_list:
    print(f"Starting date: {date}")
    t1 = time()
    df = pd.read_csv(os.path.join(os.getcwd(), 'intermediate', f'{date}_matrix.csv'))

    #List created to obtain the number of transfers that are the middle elements of the transfer 
    # chains which contains more than 2 consecutive transfers. Later, the intermediate transfers are dropped
    ListTrue =[]
    for row in range(len(df)-1):
      if df.iloc[row, 10] == True & df.iloc[row+1, 10] == True:
        ListTrue.append(row+1)
    df.drop(ListTrue, axis=0, inplace=True)

    df['destShifted'] = df['destinaiton'].shift(-1, axis = 0)

    df['destinaiton'] = df['destShifted'] * (df.transfer == True) + df['destinaiton'] * (df.transfer == False)
    #The records without the destination value (which means their destinations could not be estimated
    #  therefore the default values, -1, and -2 are assigned to the destination column) are removed.
    df = df[df.destinaiton > 0]
    # trip legs of the transfer chains are removed except the first one.
    df['transferShifted'] = df['transfer'].shift(1, axis = 0)
    df = df[df['transferShifted'] == False]
    # Drop unnecesary columns
    odm = df.drop(columns=[ 'cardID', 'transfer', 'destShifted', 'transferShifted'])
    odm['destinaiton'] = odm['destinaiton'].astype(int)

    # The district code of origin and destination locations are added to the matrix.
    odm = pd.merge(odm, stopsMunicipio, on=['DPAYPOINT'], how='left')
    odm = pd.merge(odm, allUniqueStopDf[['DPAYPOINT', 'IDSTOP']], left_on=['destinaiton'], right_on=['IDSTOP'], how='left')
    odm = pd.merge(odm, stopsMunicipio, left_on=['DPAYPOINT_y'], right_on=['DPAYPOINT'], how='left')
    odm = pd.merge(odm, allUniqueStopDf[["LATITUD","LONGITUD","DPAYPOINT"]],
                   left_on=["DPAYPOINT_y"], right_on="DPAYPOINT", how="left", suffixes=("","_d"))
    
    # Drop unnecesary columns and rename
    odm = odm.drop(columns=[ 'DPAYPOINT', 'DPAYPOINT_y', 'DPAYPOINT_d', 'DPAYPOINT_x',
                             'IDLINEA', 'DENOMINAPARADA', 'IDSTOP_y', "Unnamed: 0", "type",
                             "IDSTOP_x", "destinaiton" ])
    odm = odm.rename(columns={"CODIGOMUNICIPIO_x":"origen", "CODIGOMUNICIPIO_y":"destino", "date":"fecha"})
    
    odm = odm.dropna()
    
    # Vectorized distance calculation using NumPy
    coord1 = odm[['LATITUD', 'LONGITUD']].to_numpy()
    coord2 = odm[['LATITUD_d', 'LONGITUD_d']].to_numpy()
    distances = np.array([geodesic(coord1[i], coord2[i]).meters for i in range(len(coord1))])
    odm['distancia'] = distances

    odm = odm.drop(columns=["LATITUD","LATITUD_d", "LONGITUD","LONGITUD_d"])
    odm.to_csv(os.path.join(os.getcwd(),"output", f"{date}_ttp_matrix.csv"), index=False)
    print (f"Finished {date} in {round(time()-t1)} seconds")
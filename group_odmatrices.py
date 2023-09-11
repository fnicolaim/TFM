"""
This code reads and processes CSV files with travel data, applies various
 transformations, and then combines the results into a final CSV file.
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

# Generate objective dates list
s = datetime(2021, 1, 1)
e = datetime(2021, 6, 30)
dates = [(s + timedelta(days=i)).strftime('%Y-%m-%d') for i in range((e - s).days + 1)]

dfs = []
for d in tqdm(dates):
    df = pd.read_csv(os.path.join(os.getcwd(),"output",f"{d}_ttp_matrix.csv"), dtype="str")

    # Format and binarize distance
    df.distancia = df.distancia.astype(float)

    # Define the bin edges
    bins = [0, 500, 2000, 10000, float('inf')]  # The last bin includes values >= 10000

    # Define labels for the bins
    labels = ['0-500', '500-2000', '2000-10000', '10000+']

    # Binarize the 'distance' column into the specified bins
    df['distancia'] = pd.cut(df['distancia'], bins=bins, labels=labels, right=False)

    df.fecha = pd.to_datetime(df.fecha, format="%Y-%m-%d %H:%M:%S")
    
    df["trips"] = 1
    df = df.groupby([pd.Grouper(key='fecha', freq='H'),'origen', 'destino'], as_index=False).agg({"trips":"sum"})
    df = df.loc[df["trips"]!=0]
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df.to_csv(os.path.join(os.getcwd(),"output""final_odm.csv"), index=False)
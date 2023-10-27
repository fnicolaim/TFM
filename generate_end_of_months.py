"""
This snippet generates an analog file to transaction_data_YYYYMM.txt but containing only last days of months
Note: The current state of file generates last days of july, august, and september, the logic to create
first months of year must be replicated.
"""
import os
import pandas as pd

df_1_t = pd.read_csv(os.path.join(os.getcwd(),"input","transaction_data_202107.txt"), sep="#", skiprows=4260000,
                   names=["cardID","DIA","DPAYPOINT"])

# Convert the 'date' column to datetime format
df_1_t['DIA'] = pd.to_datetime(df_1_t['DIA'], format="%Y-%m-%dT%H:%M:%S")
# Filter transactions within the specified time range
df_1_t = df_1_t[df_1_t['DIA'] > "2021-07-31 04:00:00"]
print(df_1_t.shape)

df_2_h = pd.read_csv(os.path.join(os.getcwd(),"input","transaction_data_202108.txt"), sep="#", nrows=100000)
df_2_t = pd.read_csv(os.path.join(os.getcwd(),"input","transaction_data_202108.txt"), sep="#", skiprows=5600000,
                     names=["cardID","DIA","DPAYPOINT"])
# select rows of interest
df_2_h['DIA'] = pd.to_datetime(df_2_h['DIA'], format="%Y-%m-%dT%H:%M:%S")
df_2_h = df_2_h[df_2_h['DIA'] < "2021-08-01 03:59:59"]
print(df_2_h.shape)
df_2_t['DIA'] = pd.to_datetime(df_2_t['DIA'], format="%Y-%m-%dT%H:%M:%S")
df_2_t = df_2_t[df_2_t['DIA'] > "2021-08-31 04:00:00"]
print(df_2_t.shape)

df_3_h = pd.read_csv(os.path.join(os.getcwd(),"input","transaction_data_202109.txt"), sep="#", nrows=100000)
df_3_t = pd.read_csv(os.path.join(os.getcwd(),"input","transaction_data_202109.txt"), sep="#", skiprows=7400000,
                     names=["cardID","DIA","DPAYPOINT"])
# select rows of interest
df_3_h['DIA'] = pd.to_datetime(df_3_h['DIA'], format="%Y-%m-%dT%H:%M:%S")
df_3_h = df_3_h[df_3_h['DIA'] < "2021-09-01 03:59:59"]
print(df_3_h.shape)
df_3_t['DIA'] = pd.to_datetime(df_3_t['DIA'], format="%Y-%m-%dT%H:%M:%S")
df_3_t = df_3_t[df_3_t['DIA'] > "2021-09-30 04:00:00"]
print(df_3_t.shape)

df_4_h = pd.read_csv(os.path.join(os.getcwd(),"input","transaction_data_202110.txt"), sep="#", nrows=100000)

# select rows of interest
df_4_h['DIA'] = pd.to_datetime(df_4_h['DIA'], format="%Y-%m-%dT%H:%M:%S")
df_4_h = df_4_h[df_4_h['DIA'] < "2021-10-01 03:59:59"]
print(df_4_h.shape)

dfs = [df_1_t, 
       df_2_h, df_2_t,
       df_3_h, df_3_t,
       df_4_h]
df = pd.concat(dfs, ignore_index=True)
df.DIA = df.DIA.dt.strftime("%Y-%m-%dT%H:%M:%S")
df.to_csv(r"D:\Proyectos personales\TFM DEFINITIVO\input\transaction_data_edges.txt", sep="#", index=False)
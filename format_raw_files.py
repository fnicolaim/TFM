"""
This script is used to process the CRTM raw 2021 files into "transaction_data_YYYYMM.txt" format
 which is later used by the od matrices creation scripts.
"""
import os
from tqdm import tqdm
import pandas as pd

data_dir = r"d:\\Proyectos personales\\AI4GOV\\TFM\\Data"
chunksize = 100000 # Used to avoid RAM problems

for month in range(1,11):
    month = str(month).zfill(2)

    # Parse transaction data BUS
    df_iterator = pd.read_csv(rf"D:\Proyectos personales\AI4GOV\TFM\Data\0_raw\2021\TTP_BUS_PERIODO_{month}_2021.csv.gz",dtype="str",
                            chunksize=chunksize,sep=";",usecols=["IDSERIETARJETA","TXNOMBRETITULO","ACTOR","DIA","L_BIT","P_BIT"], encoding="latin")
    for _, df in tqdm(enumerate(df_iterator)):
        # Crete DPAYPOINT
        df["DPAYPOINT"] = df["ACTOR"] + "_L" + df["L_BIT"] + "_P" + df["P_BIT"]
        # Filter 3rd age titles only
        df = df.loc[df["TXNOMBRETITULO"].isin(["ABONO ANUAL TERCERA EDAD", "ABONO 30 DIAS TERCERA EDAD"])]
        # Format and drop unnecesary columns
        df = df.rename(columns={"IDSERIETARJETA": "cardID"})
        df = df.drop(columns=["ACTOR", "L_BIT", "P_BIT", "TXNOMBRETITULO"])
        # Save
        if _ == 0:
            df.to_csv(os.path.join(r"D:\Proyectos personales\TFM DEFINITIVO\input", f"TTP_BUS_PERIODO_{month}_2021.csv.gz"), index=False, sep=";")
        else:
            df.to_csv(os.path.join(r"D:\Proyectos personales\TFM DEFINITIVO\input", f"TTP_BUS_PERIODO_{month}_2021.csv.gz"), index=False, sep=";", mode="a", header=None)

    # Parse transaction data TRAIN
    df_iterator = pd.read_csv(rf"D:\Proyectos personales\AI4GOV\TFM\Data\0_raw\2021\TTP_TRENES_PERIODO_{month}_2021.csv.gz",dtype="str",
                            chunksize=chunksize,sep=";",usecols=["IDSERIETARJETA","TXNOMBRETITULO","ACTOR","DIA","L_BIT","P_BIT"], encoding="latin")
    for _,df in tqdm(enumerate(df_iterator)):
        # Crete DPAYPOINT
        df["DPAYPOINT"] = df["ACTOR"] + "_L" + df["L_BIT"] + "_P" + df["P_BIT"]
        # Filter 3rd age titles only
        df = df.loc[df["TXNOMBRETITULO"].isin(["ABONO ANUAL TERCERA EDAD", "ABONO 30 DIAS TERCERA EDAD"])]
        # Format and drop unnecesary columns
        df = df.rename(columns={"IDSERIETARJETA":"cardID"})
        df = df.drop(columns=["ACTOR","L_BIT","P_BIT", "TXNOMBRETITULO"])
        # Save
        if _ == 0:
            df.to_csv(os.path.join(r"D:\Proyectos personales\TFM DEFINITIVO\input", f"TTP_TRENES_PERIODO_{month}_2021.csv.gz"), index=False, sep=";")
        else:
            df.to_csv(os.path.join(r"D:\Proyectos personales\TFM DEFINITIVO\input", f"TTP_TRENES_PERIODO_{month}_2021.csv.gz"), index=False, sep=";", mode="a", header=None)
        
    # JOIN BUS AND TRAIN DATA
    df_TREN = pd.read_csv(os.path.join(r"D:\Proyectos personales\TFM DEFINITIVO\input", f"TTP_TRENES_PERIODO_{month}_2021.csv.gz"), dtype="str", sep=";")
    df_BUS = pd.read_csv(os.path.join(r"D:\Proyectos personales\TFM DEFINITIVO\input", f"TTP_BUS_PERIODO_{month}_2021.csv.gz"), dtype="str", sep=";")
    df_glob = pd.concat([df_BUS, df_TREN], ignore_index=True)
    df_glob = df_glob.sort_values(by="DIA")
    df_glob.DIA = pd.to_datetime(df_glob.DIA, format="%d/%m/%Y %H:%M:%S")
    df_glob.DIA = df_glob.DIA.dt.strftime('%Y-%m-%dT%H:%M:%S')
    df_glob.to_csv(os.path.join(r"D:\Proyectos personales\TFM DEFINITIVO\input", f"transaction_data_2021{month}.txt"), sep="#",index=False)
"""
This code reads and processes CSV files with travel data, applies various
 transformations, and then combines the results into a final CSV file.
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

# Generate a list of objective dates from January 1, 2021, to October 30, 2021
start_date = datetime(2021, 1, 1)
end_date = datetime(2021, 10, 30)
dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range((end_date - start_date).days + 1)]

# Initialize an empty list to store DataFrames
dfs = []

# Loop through each date in the list
for date in tqdm(dates):
    # Read a CSV file based on the date from the "output" directory and store it in a DataFrame
    df = pd.read_csv(os.path.join(os.getcwd(), "output", f"{date}_ttp_matrix.csv"), dtype="str")

    # Convert the 'distancia' column to float
    df.distancia = df.distancia.astype(float)

    # Define bin edges for distance binning
    bins = [0, 500, 2000, 10000, float('inf')]

    # Define labels for the bins
    labels = ['0-500', '500-2000', '2000-10000', '10000+']

    # Binarize the 'distance' column into specified bins with labels
    df['distancia'] = pd.cut(df['distancia'], bins=bins, labels=labels, right=False)

    # Convert the 'fecha' column to datetime format
    df.fecha = pd.to_datetime(df.fecha, format="%Y-%m-%d %H:%M:%S")

    # Add a 'trips' column with a constant value of 1
    df["trips"] = 1

    # Group the DataFrame by 'fecha', 'origen', and 'destino', aggregating 'trips' by sum
    df = df.groupby([pd.Grouper(key='fecha', freq='H'), 'origen', 'destino'], as_index=False).agg({"trips": "sum"})

    # Filter out rows where 'trips' is not equal to zero
    df = df.loc[df["trips"] != 0]

    # Append the processed DataFrame to the list
    dfs.append(df)

# Concatenate all DataFrames in the list into a single DataFrame, ignoring original indices
final_df = pd.concat(dfs, ignore_index=True)

# Write the final concatenated DataFrame to a CSV file in the "output" directory, excluding the index column
final_df.to_csv(os.path.join(os.getcwd(), "output", "final_odm.csv"), index=False)

# Merge with weather data
weather_df = pd.read_csv(os.path.join(os.getcwd(), "input", "weather_madrid_2021_01_10.txt"))
weather_df["time"] = pd.to_datetime(weather_df["time"], format="%Y-%m-%dT%H:%M")

agg_dict = {
    "temperature_2m (°C)":"mean",
    "rain (mm)":"sum",
    "snowfall (cm)":"sum",
}
weather_df = weather_df.groupby([pd.Grouper(key='time', freq='H')], as_index=False).agg(agg_dict)
weather_df = weather_df.rename(columns={"time":"fecha"})

final_df = pd.merge(final_df, weather_df, on="fecha", how="outer").sort_values(by="fecha", ascending=True)
final_df.to_csv(os.path.join(os.getcwd(),"output", "final_odm_with_weather.csv"), index=False)
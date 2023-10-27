# TFM - CRTM

## Master Project in Data Science

Original Doga thesis can be found here -> https://oa.upm.es/71384/

### Files execution goes as follows:

0. format_raw_files.py 
    Inputs: "TTP_BUS_PERIODO_MM_YYYY.csv.gz" and "TTP_TRENES_PERIODO_MM_YYYY.csv.gz" files
    Returns: "transaction_data_YYYYMM.txt" files

Concatenates raw data from trains and buses and selects only senior smartcard contracts

1. generate_end_of_months.py
    Inputs: "TTP_BUS_PERIODO_MM_YYYY.csv.gz" and "TTP_TRENES_PERIODO_MM_YYYY.csv.gz" files
    Returns: "transaction_data_edges.txt"

Same as before but for end of months

2. near_stop.py
    Inputs: "BD_TOP_AUTOBUSES" (raw file), and "stationInfo.csv"
    Returns: "near.csv"

Note: stationInfo.csv file was provided by Doga it has ad hoc prepreocessing to mathc lines.csv file, stopInfo.csv is generated on the fly in this snippet out of raw data.

3. destination_trasfer.py
    Inputs: ("transaction_data_YYYYMM.txt" or "transaction_data_edges.txt"), "stationInfo.csv", "stopInfo.csv", "lines.csv", and "near.csv"
    Returns: "YYYY-MM-DD_matrix.csv" files

Note: Change tha dates and paths to your selection in order to execute correctly (rows 540 to 555)
Note: This script generates intermediate files "YYYY_MM_DD_filtered_data.txt" which conain transaction data with dropped columns and date filters to include only the given date transactons.
HINT: Do not execute with all the ojective data at once but call it multiple times from console changing dates to use more than one core.

3. odmatrix.py
    Inputs: "YYYY-MM-DD_matrix.csv", "stationInfo.csv" and "stopInfo.csv" files
    Returns: "YYYY-MM-DD_ttp_matrix.csv" files

Transforms the dissagregated matries to origin destination matrices at a municipality level

4. group_odmatrices.py
    Inputs: "YYYY-MM-DD_ttp_matrix.csv" files
    Returns: "fina_odm" and "final_odm_with_weather.csv" files

This script reads all the generated ttp files groups them by distance, origin and destination and merges them with weather data

5. train_lstm_models.ipynb

This notebook must be executed in google colab for easiness.
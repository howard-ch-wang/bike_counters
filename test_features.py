import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import utils
# from external_data import example_estimator as ex

def _merge_external_data(X):
    df = pd.read_csv(
        "data/H_75_previous-2020-2022.csv.gz",
        parse_dates=["AAAAMMJJHH"],
        date_format="%Y%m%d%H",
        compression="gzip",
        sep=";",
    ).rename(columns={"AAAAMMJJHH": "date"})

    df = df[
        (df["date"] >= df["date"].min())
        & (df["date"] <= df["date"].max())
    ]

    weather = (
        df.drop(columns=["NUM_POSTE", "NOM_USUEL", "LAT", "LON", "QDXI3S"])
        .groupby("date")
        .mean()
        .dropna(axis=1, how="all")
        .interpolate(method="linear") #test with and without
    )

    q_indicators = [col for col in weather.columns if col.startswith('Q')]
    w_values = weather.drop(columns=q_indicators)

    cols = ['RR1', 'DRR1', 'FF', 'FXY', 'FXI', 'FXI3S', 
            'T', 'TD', 'TN', 'TX', 'DG', 'U', 'UX', 'DHUMI40',
            'DHUMI80', 'INS', 'VV', 'DVV200', 'NEIGETOT']
    # cols = w_values.columns

    X = X.merge(w_values[cols], on='date', how='left')
    return X

def create_features(X):
    X["is_rain"] = (X["RR1"] > 0).astype(int)
    X['is_snow'] = (X['NEIGETOT'] > 0).astype(int)
    X['wind_chill'] = 13.12 + 0.6215 * X['T'] - 11.37 * (X['FF']*3.6)**0.16 + 0.3965 * X['T'] * (X['FF']*3.6)**0.16
    X['is_rush_hour'] = (
        (X['date'].dt.hour >= 8) & (X['date'].dt.hour < 10) | 
        (X['date'].dt.hour >= 17) & (X['date'].dt.hour < 20)
    ).astype(int)

    return X

def covid_dates(df):

    '''Creates a binary variable - 1 if that observation happened during a lockdown in paris and 0 otherwise
    '''

    X = df.copy()

    date_ranges = [
    #("2020-03-17", "2020-05-11"),
    ("2020-10-05", "2020-12-14"),
    ("2021-03-20", "2021-05-03"),
    ]

    # Convert date ranges to pandas datetime
    date_ranges = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in date_ranges]

    # Create binary variable
    def is_in_date_range(date, ranges):
        for start, end in ranges:
            if start <= date <= end:
                return 1
        return 0

    X["in_date_range"] = X["date"].apply(lambda x: is_in_date_range(x, date_ranges))
    return X



# X, y = utils.get_train_data()
# #print(X.head())

# X = _merge_external_data(X)
# print(X)
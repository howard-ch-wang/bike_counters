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

    weather["is_rain"] = (weather["RR1"] > 0).astype(int)
    weather['is_snow'] = (weather['NEIGETOT'] > 0).astype(int)

    cols = ['RR1', 'FF', 'T', 'TD', 'U', 'NEIGETOT', 'is_rain', 'is_snow']

    X = X.merge(weather[cols], on='date', how='left')
    return X


# X, y = utils.get_train_data()
# #print(X.head())

# X = _merge_external_data(X)
# print(X)
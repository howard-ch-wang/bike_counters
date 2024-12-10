import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from pathlib import Path
import seaborn as sns
#from feature_engine.timeseries.forecasting import LagFeatures
import random
random.seed(125)

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline

# --------- HELPER FUNCTIONS --------------#


def _encode_dates(X, cols=['date']):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    for time_col in cols:
        X[f"{time_col}_year"] = X[time_col].dt.year
        X[f"{time_col}_month"] = X[time_col].dt.month
        X[f"{time_col}_day"] = X[time_col].dt.day
        X[f"{time_col}_weekday"] = X[time_col].dt.weekday
        X[f"{time_col}_hour"] = X[time_col].dt.hour

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=cols)

def train_test_split_temporal(X, y, delta_threshold="30 days"):
    
    cutoff_date = X["date"].max() - pd.Timedelta(delta_threshold)
    mask = (X["date"] <= cutoff_date)
    X_train, X_valid = X.loc[mask], X.loc[~mask]
    y_train, y_valid = y[mask], y[~mask]

    return X_train, y_train, X_valid, y_valid

def covid_dates(df):

    '''Creates a binary variable 'in_date_range' - 1 if that observation happened during a lockdown in paris and 0 otherwise
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

def get_test_data(path="data/final_test.parquet"):
    data = pd.read_parquet(path)
    X_test = data.copy()
    return X_test

def get_train_data(path="data/train.parquet"):
    #problem_title = "Bike count prediction"
    _target_column_name = "log_bike_count"
    data = pd.read_parquet(path)
    # Sort by date first, so that time based cross-validation would produce correct results
    data = data.sort_values(["date", "counter_name"])
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name, "bike_count"], axis=1)
    return X_df, y_array

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

#-------------- DATA and MODEL Loading -------------#

X, y = get_train_data()
X = _merge_external_data(X)
X = covid_dates(X)
X = create_features(X)
X_train, y_train = X, y #retraining on full dataset
y2 = np.where(y == 0, 0, 1)

date_encoder = FunctionTransformer(_encode_dates)
#make sure to pass any date columns here as well
date_cols = _encode_dates(X_train[["date"]]).columns.tolist() 

categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
categorical_cols = ["counter_name", "site_name"]
binary_cols = ['in_date_range', 'is_rain', 'is_snow', 'is_rush_hour']
location_cols = ['latitude', 'longitude']
numerical_cols_lag = ['NEIGETOT']
numerical_cols = ['RR1', 'DRR1', 'FF', 'FXY', 'FXI', 'FXI3S', 
            'T', 'TD', 'TN', 'TX', 'DG', 'U', 'UX', 'DHUMI40',
            'DHUMI80', 'INS', 'VV', 'DVV200', 'NEIGETOT', 'wind_chill']
numerical_cols = ['T', 'wind_chill']
#lag_transformer = LagFeatures(variables=numerical_cols_lag, periods=[2, 24, 48], missing_values='ignore')

#these are created later, can find these with get_features_out
lagged_cols = [
    #'RR1_lag_2', 'FF_lag_2', 'T_lag_2', 'TD_lag_2', 'U_lag_2', 
               'NEIGETOT_lag_2',
    #'RR1_lag_24', 'FF_lag_24', 'T_lag_24', 'TD_lag_24', 'U_lag_24', 
    'NEIGETOT_lag_24',
    #           'RR1_lag_48', 'FF_lag_48', 'T_lag_48', 'TD_lag_48', 'U_lag_48', 
    'NEIGETOT_lag_48']

#Choose which features to include at this stage
preprocessor = ColumnTransformer(
    [
        ("date", OneHotEncoder(handle_unknown="ignore", sparse_output=False), date_cols),
        #("install", OneHotEncoder(handle_unknown="ignore"), install_cols),
        ("cat", categorical_encoder, categorical_cols),
        ('location', 'passthrough', location_cols),
        #('numerical', 'passthrough', numerical_cols),
        #('lagged', 'passthrough', lagged_cols),
        ('binary', 'passthrough', binary_cols)
    ],
    #remainder='passthrough'
)

regressor = HistGradientBoostingRegressor(max_leaf_nodes=50, verbose=1, max_iter=500)
classifier = HistGradientBoostingClassifier(max_leaf_nodes=50, verbose=1, max_iter=500)
print(f'Building with {regressor} and {classifier}')

#---------------------TRAINING---------------#

pipe = Pipeline([
    #('lag_features', lag_transformer),
    ('date_encoder', date_encoder),
    ('preprocessor', preprocessor),
    ('regressor', regressor)
])

pipe_class = Pipeline([
    #('lag_features', lag_transformer),
    ('date_encoder', date_encoder),
    ('preprocessor', preprocessor),
    ('classifier', classifier)
])

pipe.fit(X_train, y_train, 
         #regressor__sample_weight=weights
         )

pipe_class.fit(X_train, y2)


#---------------PREDICTION------------------#


X_test = get_test_data()
X_test = _merge_external_data(X_test)
X_test = covid_dates(X_test)
X_test = create_features(X_test)
y_pred = pipe.predict(X_test)
#y_pred = np.where(y_pred < 0, 0, y_pred)
y_class = pipe_class.predict_proba(X_test)[:, 0]

y_out = np.where(y_class > 0.7, 0, y_pred)

sol = {
    'Id': list(range(len(y_pred))),
    'log_bike_count': y_out.flatten()
}

submission = pd.DataFrame(sol)
submission.set_index("Id", inplace=True)
submission.to_csv('submission.csv')
#------------------- RATIONALE SUMMARY ---------#

#This script trains a HistGradientBoostRegressor.

#We use limited features based on the original dataset, and few variables created from the weather dataset. 
# link to external dataset: https://www.data.gouv.fr/fr/datasets/r/a77b4d44-d361-4e59-b6cc-cbbf435a2d89, by Météo-France
# Royalty-free under Etalab Open License: https://www.etalab.gouv.fr/wp-content/uploads/2014/05/Licence_Ouverte.pdf

#This script does not include the experimentation and various models, features and approaches we tried, parameter tuning, 
#please refer to the scripts in https://github.com/howard-ch-wang/bike_counters, and our report for those details

#------------------ IMPORTS ------------------#
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
# from sklearn.model_selection import RandomizedSearchCV

# --------- HELPER FUNCTIONS --------------#

#from the given code in the repo
def _encode_dates(X, cols=['date']):
    """
    Extracts and encodes date-related features from specified date columns in a DataFrame.

    Args:
        X (pd.DataFrame): The input DataFrame containing date columns.
        cols (list of str, optional): List of column names to encode. Defaults to ['date'].

    Returns:
        pd.DataFrame: A new DataFrame with encoded year, month, day, weekday, and hour for each date column, 
                      and without the original date columns.
    """
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
    """
    Splits data into temporal training and validation sets based on a cutoff date.

    Args:
        X (pd.DataFrame): Feature DataFrame containing a 'date' column.
        y (pd.Series or np.ndarray): Target variable corresponding to X.
        delta_threshold (str, optional): Time delta defining the validation set period. Defaults to "30 days".

    Returns:
        tuple: (X_train, y_train, X_valid, y_valid), where each is a subset of the data.
    """
    cutoff_date = X["date"].max() - pd.Timedelta(delta_threshold)
    mask = (X["date"] <= cutoff_date)
    X_train, X_valid = X.loc[mask], X.loc[~mask]
    y_train, y_valid = y[mask], y[~mask]

    return X_train, y_train, X_valid, y_valid

def covid_dates(df):
    """
    Adds a binary column indicating whether each observation falls within predefined lockdown date ranges in Paris.

    Args:
        df (pd.DataFrame): The input DataFrame containing a 'date' column.

    Returns:
        pd.DataFrame: A new DataFrame with an added 'in_date_range' column.
    """
    X = df.copy()

    #based on: https://en.wikipedia.org/wiki/COVID-19_pandemic_in_France
    date_ranges = [
    # ("2020-03-17", "2020-05-11"),
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

def get_test_data(path="/kaggle/input/msdb-2024/final_test.parquet"):
    """
    Loads and returns the test dataset from a specified file path.

    Args:
        path (str, optional): Path to the test data file. Defaults to "/kaggle/input/msdb-2024/final_test.parquet".

    Returns:
        pd.DataFrame: A DataFrame containing the test data.
    """
    data = pd.read_parquet(path)
    X_test = data.copy()
    return X_test

def get_train_data(path="/kaggle/input/msdb-2024/train.parquet"):
    """
    Loads and prepares the training dataset for analysis or modeling.

    Args:
        path (str, optional): Path to the training data file. Defaults to "/kaggle/input/msdb-2024/train.parquet".

    Returns:
        tuple: (X_df, y_array), where X_df is the feature DataFrame, and y_array is the target variable array.
    """
    #problem_title = "Bike count prediction"
    _target_column_name = "log_bike_count"
    data = pd.read_parquet(path)
    # Sort by date first, so that time based cross-validation would produce correct results
    data = data.sort_values(["date", "counter_name"])
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name, "bike_count"], axis=1)
    return X_df, y_array

# Function to merge weather data. However most of the features did not help so are omitted later.
def _merge_external_data(X):
    """
    Merges external weather data with the input DataFrame based on the 'date' column.

    Args:
        X (pd.DataFrame): The input DataFrame containing a 'date' column.

    Returns:
        pd.DataFrame: A new DataFrame with merged weather data.
    """
    df = pd.read_csv(
        "/kaggle/input/hourly-weather/H_75_previous-2020-2022.csv",
        parse_dates=["AAAAMMJJHH"],
        date_format="%Y%m%d%H",
        # compression="gzip",
        sep=";",
    ).rename(columns={"AAAAMMJJHH": "date"})

    df = df[
        (df["date"] >= df["date"].min())
        & (df["date"] <= df["date"].max())
    ]

    #The data entries are all from weather stations in Paris(75). Drop station-wise info for simplicity.
    #Then we use the mean value across all stations in Paris for each measurement,
    #drop empty columns, and linearly interpolate some random missing values exist in the dataset.
    weather = (
        df.drop(columns=["NUM_POSTE", "NOM_USUEL", "LAT", "LON"])
        .groupby("date")
        .mean()
        .dropna(axis=1, how="all")
        .interpolate(method="linear")
    )

    #Select all relevant measurements in dataset
    cols = ['RR1', 'DRR1', 'FF', 'FXY', 'FXI', 'FXI3S', 
            'T', 'TD', 'TN', 'TX', 'DG', 'U', 'UX', 'DHUMI40',
            'DHUMI80', 'INS', 'VV', 'DVV200', 'NEIGETOT']
    # cols = w_values.columns

    X = X.merge(weather[cols], on='date', how='left')
    return X


def create_features(X):
    """
    Creates additional feature columns in the DataFrame based on weather, time, and calculated metrics.

    Args:
        X (pd.DataFrame): The input DataFrame containing relevant columns.

    Returns:
        pd.DataFrame: A new DataFrame with added feature columns including 'is_rain', 'is_snow', 'wind_chill', and 'is_rush_hour'.
    """
    #Binary
    X["is_rain"] = (X["RR1"] > 0).astype(int)  #RR1: the amount of precipitation fallen in 1 hour (in mm and 1/10 mm)
    X['is_snow'] = (X['NEIGETOT'] > 0).astype(int)  #NEIGETOT: total height of snow on the ground (in cm)
    X['is_rush_hour'] = (
        (X['date'].dt.hour >= 8) & (X['date'].dt.hour < 10) | 
        (X['date'].dt.hour >= 17) & (X['date'].dt.hour < 20)
    ).astype(int)  #Considered 8~10h & 17~20h rush hour

    #Numerical
    X['wind_chill'] = 13.12 + 0.6215 * X['T'] - 11.37 * (X['FF']*3.6)**0.16 + 0.3965 * X['T'] * (X['FF']*3.6)**0.1
    return X

#-------------- DATA and MODEL Loading -------------#

X, y = get_train_data()
X = _merge_external_data(X)
X = covid_dates(X)
X = create_features(X)
X_train, y_train = X, y #retraining on full dataset

date_encoder = FunctionTransformer(_encode_dates)
#make sure to pass any date columns here as well
date_cols = _encode_dates(X_train[["date"]]).columns.tolist() 

categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
categorical_cols = ["counter_name", "site_name"]
binary_cols = ['in_date_range', 'is_rain', 'is_snow', 'is_rush_hour']
location_cols = ['latitude', 'longitude']

# The columns below were ultimately not used in prediction, they worsened model performance consistently

# numerical_cols_lag = ['NEIGETOT']
# numerical_cols = [
#     'RR1', 'DRR1', 'FF', 'FXY', 'FXI', 'FXI3S', 
#             'T', 
#     'TD', 'TN', 'TX', 'DG', 'U', 'UX', 'DHUMI40',
#             #'DHUMI80', 
#     'INS', 
#     'VV', 'DVV200', 'NEIGETOT', 'wind_chill'
# ]
# numerical_cols = ['T', 'wind_chill', 'INS']
# #lag_transformer = LagFeatures(variables=numerical_cols_lag, periods=[2, 24, 48], missing_values='ignore')

# #these are created later, can find these with get_features_out
# lagged_cols = [
#     #'RR1_lag_2', 'FF_lag_2', 'T_lag_2', 'TD_lag_2', 'U_lag_2', 
#                'NEIGETOT_lag_2',
#     #'RR1_lag_24', 'FF_lag_24', 'T_lag_24', 'TD_lag_24', 'U_lag_24', 
#     'NEIGETOT_lag_24',
#     #           'RR1_lag_48', 'FF_lag_48', 'T_lag_48', 'TD_lag_48', 'U_lag_48', 
#     'NEIGETOT_lag_48']


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

regressor = HistGradientBoostingRegressor(max_leaf_nodes=50, 
                                          verbose=1,
                                          max_iter=5000)
print(f'Building with {regressor}')

#---------------------TUNING and TRAINING---------------#

# Define pipeline
pipe = Pipeline([
    #('lag_features', lag_transformer),
    ('date_encoder', date_encoder),
    ('preprocessor', preprocessor),
    ('regressor', regressor)
])

# Fit the search
pipe.fit(X_train, y_train)

#---------------PREDICTION------------------#

X_test = get_test_data()
X_test = _merge_external_data(X_test)
X_test = covid_dates(X_test)
X_test = create_features(X_test)
y_pred = pipe.predict(X_test)
#y_pred = np.where(y_pred < 0, 0, y_pred)

sol = {
    'Id': list(range(len(y_pred))),
    'log_bike_count': y_pred.flatten()
}

submission = pd.DataFrame(sol)
submission.set_index("Id", inplace=True)
submission.to_csv('submission.csv')
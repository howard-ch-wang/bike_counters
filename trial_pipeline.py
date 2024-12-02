import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from pathlib import Path
import utils
from external_data import example_estimator

#helper functions

def _encode_dates(X, cols=['date', 'counter_installation_date']):
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


def get_test_data(path="data/final_test.parquet"):
    data = pd.read_parquet(path)
    # Sort by date first, so that time based cross-validation would produce correct results
    #data = data.sort_values(["date", "counter_name"])
    X_test = data.copy()
    return X_test


#load the data

#data = pd.read_parquet(Path("data") / "train.parquet")
X, y = utils.get_train_data()
X = example_estimator._merge_external_data(X)
print(f'Data: {X.columns}')

X_train, y_train, X_valid, y_valid = train_test_split_temporal(X, y)

print(
    f'Train: n_samples={X_train.shape[0]},  {X_train["date"].min()} to {X_train["date"].max()}'
)
print(
    f'Valid: n_samples={X_valid.shape[0]},  {X_valid["date"].min()} to {X_valid["date"].max()}'
)


#Model setup 
#make sparse output = true for non GB reg models - it's faster

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

date_encoder = FunctionTransformer(_encode_dates)
#make sure to pass any date columns here as well
date_cols = _encode_dates(X_train[["date", 'counter_installation_date']]).columns.tolist() 

categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
categorical_cols = ["counter_name", "site_name"]
location_cols = ['latitude', 'longitude']
numerical_cols = ['t']

preprocessor = ColumnTransformer(
    [
        ("date", OneHotEncoder(handle_unknown="ignore", sparse_output=False), date_cols),
        #("install", OneHotEncoder(handle_unknown="ignore"), install_cols),
        ("cat", categorical_encoder, categorical_cols),
        ('location', 'passthrough', location_cols),
        ('numerical', StandardScaler(), numerical_cols)
    ],
    #remainder='passthrough'
)

#Ridge, HistGradientBoostingRegressor
regressor = HistGradientBoostingRegressor()
#regressor = Ridge()

print(f'Building with {regressor}')

pipe = make_pipeline(date_encoder, preprocessor, regressor)
pipe.fit(X_train, y_train)


#cross validation
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

cv = TimeSeriesSplit(n_splits=6)

# When using a scorer in scikit-learn it always needs to be better when smaller, hence the minus sign.
scores = cross_val_score(
    pipe, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error"
)
print("RMSE: ", scores)
print(f"RMSE (all folds): {-scores.mean():.3} ± {(-scores).std():.3}")

print('success')


#getting the test data, making the predictions

X_test = get_test_data()
X_test = example_estimator._merge_external_data(X_test)
y_pred = pipe.predict(X_test)

sol = {
    'Id': list(range(len(y_pred))),
    'log_bike_count': y_pred
}

submission = pd.DataFrame(sol)
submission.set_index("Id", inplace=True)
submission.to_csv('submission.csv')


# feature importances
print(pipe.steps)
feature_names = pipe.steps[1][1].get_feature_names_out()
print("Features after preprocessing:", feature_names)
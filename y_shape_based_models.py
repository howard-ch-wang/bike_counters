import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from pathlib import Path
import utils
from external_data import example_estimator
import seaborn as sns
from feature_engine.timeseries.forecasting import LagFeatures
from test_features import _merge_external_data, create_features
#np.random.seed(80)
import random
random.seed(125)

#helper functions

# def _merge_external_data(X):
#     file_path = "/Users/sam/Desktop/X/P4DS/p4ds_sam/bike_counters/data/external_data.csv"
#     df_ext = pd.read_csv(file_path, parse_dates=["date"])
#     df_ext['date'] = df_ext['date'].astype('datetime64[us]') #small date incompatibility

#     cols = ['date', 't', 'pmer', 'tend', 'cod_tend', 'dd', 'ff', 'td', 'u', 'vv']
#     X = X.copy()
#     # When using merge_asof left frame need to be sorted
#     X["orig_index"] = np.arange(X.shape[0])
#     X = pd.merge_asof(
#         X.sort_values("date"), df_ext[cols].sort_values("date"), on="date",
#     )
#     #to add more columns, need to add in the merge line. 
#     # Sort back to the original order
#     X = X.sort_values("orig_index")
#     del X["orig_index"]
#     return X

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


def get_test_data(path="data/final_test.parquet"):
    data = pd.read_parquet(path)
    # Sort by date first, so that time based cross-validation would produce correct results
    #data = data.sort_values(["date", "counter_name"])
    X_test = data.copy()
    return X_test


#load the data

#data = pd.read_parquet(Path("data") / "train.parquet")
X, y = utils.get_train_data()
y = y.reshape(-1, 1)
print(y.shape)
print(X.info())
#y = np.exp(y)
X = _merge_external_data(X)
X = covid_dates(X)
X = create_features(X)
print(f'Data: {X.columns}')
print(X.info())
X_train, y_train, X_valid, y_valid = train_test_split_temporal(X, y)
X_train, y_train = X, y

#weighing more recent observations higher - because prediction is just after training.
current_time = X_train['date'].max()
time_diff = (current_time - X_train['date']).dt.days

# Exponential decay weights
decay_rate = 0.05  # Adjust this parameter
weights = np.exp(-decay_rate * time_diff)

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
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import PoissonRegressor
from sklearn.impute import SimpleImputer


date_encoder = FunctionTransformer(_encode_dates)
#make sure to pass any date columns here as well
date_cols = _encode_dates(X_train[["date"]]).columns.tolist() 

categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
categorical_cols = ["counter_name", "site_name"]
binary_cols = ['in_date_range', 'is_rain', 'is_snow', 'is_rush_hour']
location_cols = ['latitude', 'longitude']
numerical_cols_lag = ['RR1', 'FF', 'T', 'TD', 'U', 'NEIGETOT']
numerical_cols = ['T', 'NEIGETOT']
#numerical_cols = ['RR1', 'DRR1', 'FF', 'FXY', 'FXI', 'FXI3S', 
            # 'T', 'TD', 'TN', 'TX', 'DG', 'U', 'UX', 'DHUMI40',
            # 'DHUMI80', 'INS', 'VV', 'DVV200', 'NEIGETOT', 'wind_chill']
lag_transformer = LagFeatures(variables=numerical_cols_lag, periods=[24, 48], missing_values='ignore')

#these are created later, can find these with get_features_out
#lagged_cols = ['RR1_lag_24', 'FF_lag_24', 'T_lag_24', 'TD_lag_24', 'U_lag_24', 'NEIGETOT_lag_24',
#               'RR1_lag_48', 'FF_lag_48', 'T_lag_48', 'TD_lag_48', 'U_lag_48', 'NEIGETOT_lag_48']

preprocessor = ColumnTransformer(
    [
        ("date", OneHotEncoder(handle_unknown="ignore", sparse_output=False), date_cols),
        #("install", OneHotEncoder(handle_unknown="ignore"), install_cols),
        ("cat", categorical_encoder, categorical_cols),
        ('location', 'passthrough', location_cols),
        ('numerical', 'passthrough', numerical_cols),
        #('lagged', 'passthrough', lagged_cols),
        ('covid', 'passthrough', binary_cols)
    ],
    #remainder='passthrough'
)

#Ridge, HistGradientBoostingRegressor
#regressor = HistGradientBoostingRegressor(max_leaf_nodes=50, verbose=1, max_iter=500)
#regressor = Ridge()
#regressor = utils.GMM_eval(6)
#regressor = PoissonRegressor()

import tensorflow as tf
from tensorflow.keras.initializers import HeNormal

input_shape = X_train.shape[1]
input_shape = 170 #this needs to be set based on the post processed data shape

inputs = tf.keras.Input(shape=(input_shape,))
# x = tf.keras.layers.Dense(128, activation='relu')(inputs)
# x = tf.keras.layers.Dense(64, activation='relu')(x)
# x = tf.keras.layers.Dense(32, activation='relu')(x)
#outputs = tf.keras.layers.Dense(3)(x)  # Output for p, mu, and log(sigma)

deep_model = tf.keras.Sequential([
    #tf.keras.layers.Dense(64, activation='leaky_relu', kernel_initializer=HeNormal()),
    tf.keras.layers.Dense(32, activation='leaky_relu', kernel_initializer=HeNormal()),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(16, activation='leaky_relu', kernel_initializer=HeNormal()),
    #tf.keras.layers.Dropout(0.1),
    #tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3)  # Output layer
    #tf.keras.layers.Concatenate()([tf.keras.layers.Dense(units=1, activation='sigmoid'), tf.keras.layers.Dense(units=2)])
])
regressor = tf.keras.Model(inputs=inputs, outputs=deep_model(inputs))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
regressor.compile(optimizer=optimizer , loss=utils.zero_inflated_lognormal_loss)

print(f'Building with {regressor}')

pipe = make_pipeline(lag_transformer, date_encoder, preprocessor, regressor)

pipe = Pipeline([
    #('lag_features', lag_transformer),
    ('date_encoder', date_encoder),
    ('preprocessor', preprocessor),
    #('imputer', SimpleImputer())
    #('regressor', regressor)
])

transformed_x = pipe.fit_transform(X_train, 
         #regressor__sample_weight=weights
         )

print(f'NAs in processde data: {sum(np.isnan(transformed_x))}')
print(np.isinf(transformed_x).any()) 
# Train the model
regressor.fit(transformed_x, y_train, epochs=20, batch_size=64, 
              #sample_weight=weights
              )

#print(f'lagged: {lag_transformer.get_feature_names_out()}')
model_outputs = regressor.predict(pipe.transform(X_valid))
print(model_outputs[:100])
#y_val_pred = np.where(model_outputs[:, 0] > 0.7, 0, model_outputs[:, 1])
y_val_pred = utils.zero_inflated_lognormal_pred(model_outputs)
y_val_pred = np.array(y_val_pred).flatten()
print(f'NAs in pred: {sum(np.isnan(y_val_pred))}')
print(np.isinf(y_val_pred).any()) 

#y_val_pred = np.log(y_val_pred)
#y_valid = np.log(y_valid)

from sklearn.metrics import mean_squared_error
print(f'rmse:{mean_squared_error(y_val_pred, y_valid)}')

sns.scatterplot(x=y_valid.flatten(), y=y_val_pred, color='g', marker='.', alpha=0.1, label='Model')
reference_x = np.linspace(min(y_valid.flatten()) - 1, max(y_valid.flatten()) + 1, 1200)
plt.plot(reference_x, reference_x, label='Ideal')
plt.ylabel('Predicted y values')
plt.xlabel('Actual y values')
plt.title(f'Actual y vs Pred y')
plt.legend()
plt.show()

# #cross validation
# from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# cv = TimeSeriesSplit(n_splits=6)

# # When using a scorer in scikit-learn it always needs to be better when smaller, hence the minus sign.
# scores = cross_val_score(
#     pipe, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error"
# )
# print("RMSE: ", scores)
# print(f"RMSE (all folds): {-scores.mean():.3} ± {(-scores).std():.3}")

# print('success')


#getting the test data, making the predictions

X_test = get_test_data()
X_test = _merge_external_data(X_test)
X_test = covid_dates(X_test)
X_test = create_features(X_test)
y_pred = regressor.predict(pipe.transform(X_test))

y_pred = utils.zero_inflated_lognormal_pred(y_pred)
y_pred = np.array(y_pred).flatten()

sol = {
    'Id': list(range(len(y_pred))),
    'log_bike_count': y_pred.flatten()
}

submission = pd.DataFrame(sol)
submission.set_index("Id", inplace=True)
submission.to_csv('submission.csv')


# feature importances
print(pipe.steps)
#feature_names = pipe.steps[2][1].get_feature_names_out()
#print("Features after preprocessing:", feature_names)
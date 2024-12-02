import os

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import gmr
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from gmr import GMM

problem_title = "Bike count prediction"
_target_column_name = "log_bike_count"
# A type (class) which will be used to create wrapper objects for y_pred


def get_cv(X, y, random_state=0):
    cv = TimeSeriesSplit(n_splits=8)
    rng = np.random.RandomState(random_state)

    for train_idx, test_idx in cv.split(X):
        # Take a random sampling on test_idx so it's that samples are not consecutives.
        yield train_idx, rng.choice(test_idx, size=len(test_idx) // 3, replace=False)


def get_train_data(path="data/train.parquet"):
    data = pd.read_parquet(path)
    # Sort by date first, so that time based cross-validation would produce correct results
    data = data.sort_values(["date", "counter_name"])
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name, "bike_count"], axis=1)
    return X_df, y_array


class GMM_eval:

  '''
  A class to mkae it easier for me to fit and plot the GMM

  parameters
  ----------

  n_components: int
    the number of gaussian variables that will be fitted in the model
  '''

  def __init__(self, n_components):

    self.n_components = n_components
    self.model = GMM(n_components=n_components, random_state=0)


  def fit(self, X_train, y_train):

    '''
    Fits a GMM model with 'n_components' Gaussian variables to the data
    '''

    data_gmm_train = np.column_stack((y_train, X_train))
    self.model.from_samples(data_gmm_train)

    #this list essentially tells the model which features are going to be X features

    self.X_list = list(range(1,len(data_gmm_train[0])))

    #print(f'fit and ready to predict on {self.X_list} indices')

  def predict(self, X_test):
    '''
    creates a prediction vector
    '''

    self.y_pred = self.model.predict(self.X_list, X_test)
    return self.y_pred

  def evaluate(self, y_test, dist=False):
    '''
    Return RMSE of last predict call
    '''
    y_pred = self.y_pred


    print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}')


    plt.scatter(y_pred, y_test, color='g', marker='.', alpha=0.2, label='Model')
    reference_x = np.linspace(0, 6000, 12000)
    plt.plot(reference_x, reference_x, label='Ideal')
    plt.ylabel('Actual y values')
    plt.xlabel('Predicted y values')
    plt.title(f'Actual y vs Pred y - GMR with {self.n_components} Gaussians')
    plt.legend()
    plt.show()

    if dist:
      plt.hist(y_pred)
      plt.xlabel('y value')
      plt.ylabel('frequency')
      plt.title('Prediction distribution')
      plt.show()
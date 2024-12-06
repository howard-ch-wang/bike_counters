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

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

def ziln_loss(y_true, y_pred):
    # Extract the predicted probabilities and count predictions
    p_pred = y_pred[:, 0]  # Probability of zero
    mu_pred = y_pred[:, 1]  # Mean of the count distribution
    sigma_pred = tf.exp(y_pred[:, 2])  # Standard deviation (use exp for stability)

    print(p_pred, mu_pred, sigma_pred)

    # Calculate the loss components
    zero_inflation_loss = -tf.reduce_mean(y_true * tf.math.log(p_pred) + (1 - y_true) * tf.math.log(1 - p_pred))
    count_loss = -tf.reduce_mean(tfp.distributions.Normal(mu_pred, sigma_pred).log_prob(y_true))

    # Combine the two loss components
    total_loss = zero_inflation_loss + count_loss

    print(total_loss, zero_inflation_loss, count_loss)

    return total_loss


#https://github.com/google/lifetime_value/blob/master/lifetime_value/zero_inflated_lognormal.py
def zero_inflated_lognormal_loss(labels: tf.Tensor,
                                 logits: tf.Tensor) -> tf.Tensor:
  """Computes the zero inflated lognormal loss.

  Usage with tf.keras API:

  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile('sgd', loss=zero_inflated_lognormal)
  ```

  Arguments:
    labels: True targets, tensor of shape [batch_size, 1].
    logits: Logits of output layer, tensor of shape [batch_size, 3].

  Returns:
    Zero inflated lognormal loss value.
  """
  labels = tf.convert_to_tensor(labels, dtype=tf.float32)
  positive = tf.cast(labels > 0, tf.float32)

  logits = tf.convert_to_tensor(logits, dtype=tf.float32)
  print(labels)
  print(logits.shape, labels.shape)
  logits.shape.assert_is_compatible_with(
      tf.TensorShape(labels.shape[:-1].as_list() + [3]))

  positive_logits = logits[..., :1]
  classification_loss = tf.keras.losses.binary_crossentropy(
      y_true=positive, y_pred=positive_logits, from_logits=True)

  loc = logits[..., 1:2]
  scale = tf.math.maximum(
      tf.keras.backend.softplus(logits[..., 2:]),
      tf.math.sqrt(tf.keras.backend.epsilon()))
  safe_labels = positive * labels + (
      1 - positive) * tf.keras.backend.ones_like(labels)
  regression_loss = -tf.keras.backend.mean(
      positive * tfd.LogNormal(loc=loc, scale=scale).log_prob(safe_labels),
      axis=-1)

  return (classification_loss) + regression_loss

def zero_inflated_lognormal_pred(logits: tf.Tensor) -> tf.Tensor:
  """Calculates predicted mean of zero inflated lognormal logits.

  Arguments:
    logits: [batch_size, 3] tensor of logits.

  Returns:
    preds: [batch_size, 1] tensor of predicted mean.
  """
  logits = tf.convert_to_tensor(logits, dtype=tf.float32)
  positive_probs = tf.keras.backend.sigmoid(logits[..., :1])
  loc = logits[..., 1:2]
  scale = tf.keras.backend.softplus(logits[..., 2:])
  preds = (
      positive_probs *
      tf.keras.backend.exp(loc + 0.5 * tf.keras.backend.square(scale)))
  return preds


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
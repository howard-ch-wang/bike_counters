# File Explanation

Broadly, we have feature engineering files, utility files, and experimentation scripts. 

## Feature Engineering / utility
1. test_features.py - feature engineering functions that help merge data, add features in the scripts
2. utils.py - based on the starter code, adds classes for GMM, ZILN loss and prediction functions, cv, and so on.

## Experimentation
1. GMM.py - Implements the Gaussian Mixture Regression (gmr), using wrapper classes from utils.py. - https://github.com/AlexanderFabisch/gmr
2. HGB.py - Implements HistogramGradientBoostingRegressor. This was also used to experiment with various feature combinations
3. y_shape_based_models.py - Considering the high number of 0 values, this script experiments with Zero-Inflated LogNormal Loss and prediction. helper functions in utils.py from - https://github.com/google/lifetime_value/blob/master/lifetime_value/zero_inflated_lognormal.py, referencing https://arxiv.org/abs/1912.07753
4. examine_features.ipynb - feature importance, hyperparam tuning

## Submission

Please consider the latest scripts associated with the submission. they are based on final_script.py and final_script_2_model.py, but include more documentation and clarity.

## Data added

https://www.data.gouv.fr/fr/datasets/r/a77b4d44-d361-4e59-b6cc-cbbf435a2d89, by Météo-France
Licensing: https://donneespubliques.meteofrance.fr/?fond=produit&id_produit=110&id_rubrique=37

# Starting kit on the bike counters dataset

Read the instruction from the [Kaggle challenge](https://www.kaggle.com/competitions/mdsb-2023/overview).

### Download the data

Download the data from Kaggle and put the files into the `data` folder.

Note that your goal is to train your model on `train.parquet` (and eventual external datasets)
and then make predictions on `final_test.parquet`.

### Install the local environment

To run the notebook locally you will need the dependencies listed
in `requirements.txt`. 

It is recommended to create a new virtual environement for this project. For instance, with conda,
```bash
conda create -n bikes-count python=3.10
conda activate bikes-count
```

You can install the dependencies with the following command-line:

```bash
pip install -r requirements.txt -U
```

### The starter notebook

Get started on this challenge with the `bike_counters_starting_kit.ipynb` notebook.
This notebook is just a helper to show you different methods and ideas useful for your
exploratory notebooks and your submission script.

Launch the notebook using:

```bash
jupyter lab bike_counters_starting_kit.ipynb
```

### Submissions

Upload your script file `.py` to Kaggle using the Kaggle interface directly.
The platform will then execute your code to generate your submission csv file,
and compute your score.

Note that your submission .csv file must have the columns "Id" and "bike_log_count",
and be of the same length as `final_test.parquet`.

# to do list

1. Find weather information - howard
2. DONE Create a .py with the pipeline, so we can experiment - sam
3. research some nice time series algorithms, try to implement them with the data we have - both

4. DONE merge existing external data to the basic dataset - sam
5. DONE Implement GMR, test it and make a submission - sam
6. DONE Inspect the test X distributions and the training X distributions - sam (could try again)
7. DONE Experiment with Poisson Regression, try implementing ZILN loss - sam
8. Feature importance, trying models on full data, create final script - make it look good.

# Goals

2nd - Find the data, and find a model

7th - Finalize model and data

11th - Submission
14th - Report
20th - Presentation




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

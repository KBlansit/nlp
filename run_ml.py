#!/usr/bin/env python

# import libraries
import h5py
import xgboost

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, f1_score

# script variables
DATA_PATH = "processed_data/data/data.hdf5"

CV_ITER = 10

from sklearn.svm import SVR

# model parameters
MODEL_PARAMS = {
    # experimental setup
    "objective": "reg:linear",
    "seed": 87567,

    # tree parameters
    "n_estimators": 1000, # number of trees
    "learning_rate": 0.025,
    "gamma": 7,
    "subsample": 0.5, # closer to 0.5 helps prevent overfitting
    "max_depth": 15,
    "nthread": -1,
}

def load_stored_data():
    """
    loads data
    """
    # load data
    data_f = h5py.File(DATA_PATH)

    # get data
    x = data_f["all_x"][()]
    tmp_y = data_f["all_y"][()]

    # close data connection
    data_f.close()

    # trim x data
    x = x[:, (x.sum(axis=0) != 0)]

    # make percentiles
    percentile_000 = np.percentile(tmp_y, 0)
    percentile_025 = np.percentile(tmp_y, 25)
    percentile_050 = np.percentile(tmp_y, 50)
    percentile_075 = np.percentile(tmp_y, 75)
    percentile_100 = np.percentile(tmp_y, 100)

    # determine percnetiles
    y_000_025_percent = (tmp_y >= percentile_000) & (tmp_y < percentile_025)
    y_025_050_percent = (tmp_y >= percentile_025) & (tmp_y < percentile_050)
    y_050_075_percent = (tmp_y >= percentile_050) & (tmp_y < percentile_075)
    y_075_100_percent = (tmp_y >= percentile_075) & (tmp_y <= percentile_100)

    y_stked = np.stack([
        y_000_025_percent,
        y_025_050_percent,
        y_050_075_percent,
        y_075_100_percent,
    ], axis = 1)

    y = np.where(y_stked)[1]

    return x, y

def load_gld_std():
    """
    """
    data_f = h5py.File(DATA_PATH)

    # get data
    x = data_f["gld_x"][()]
    y = data_f["gld_y"][()]

    # trim x data
    x = x[:, (x.sum(axis=0) != 0)]

    # reshape y
    y = np.expand_dims(y, axis=1)

    # close data connection
    data_f.close()

    return x, y

def main():
    """
    main script function
    """
    # load data
    x, y = load_stored_data()

    # split data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size = 0.25,
        random_state = 54892,
    )

    # run xg boost
    model = xgboost.XGBClassifier(**MODEL_PARAMS)
    model.fit(x_train, y_train)

    print("\n")
    print("\n")
    print("\n")

    print("Train results")
    print(accuracy_score(model.predict(x_train), y_train))

    print("\n")

    print("Test results")
    print(accuracy_score(model.predict(x_test), y_test))

if __name__ == '__main__':
    main()

    #!/usr/bin/env python

# import libraries
import os
import re
import json

import numpy as np
import pandas as pd

from math import ceil
from operator import mul
from functools import reduce
from scipy.spatial import distance
from collections import OrderedDict
from keras.callbacks import TensorBoard, ModelCheckpoint

# user defined libraries
from src.callbacks import HeatmapLocalizationCallBack
from src.evaluate_model import write_images, calculate_metrics, create_histograms
from src.data_generator import get_full_study_paths, get_batched_hdf5, data_generator

class CrossValidation:
    def __init__(self, model_func, settings, write_path):
        """
        INPUTS:
            model_func:
                the model function to return the current model
            cmd_args:
                the command line arguements
            settings:
                the ai settings file
            write_path:
                the path to write out to
        EFFECT:
            initializes the cross validation to run KxK cross fold validation
        """

        # return self data
        self.model_func = model_func
        self.settings = settings
        self.eval_metrics = pd.DataFrame()

        # add to settings
        self.settings['write_path'] = write_path

    def _initialize_split(self, data, prop_validation):
        """
            data:
                the hdf5 file path to load
            prop_validation:
                proportion to use as validation
        EFFECT:
            sets single self.folds_dict of testing keys
        """
        # determine values
        val_arry = np.sort(np.array(list(data.keys())))
        np.random.shuffle(val_arry)

        # get index to split
        split_indx = round(len(val_arry) * prop_validation)

        # save into fold dict
        self.folds_dict = {
            0: val_arry[split_indx:],
            1: val_arry[:split_indx],
        }

        # remember number of iters
        self.iters = 1

    def _initialize_folds(self, data, k):
        """
        INPUTS:
            data:
                the hdf5 file path to load
            k:
                the number K cross folds
        EFFECT:
            sets the self.folds_dict of testing keys
        """
        # determine values
        val_arry = np.sort(np.array(list(data.keys())))
        np.random.shuffle(val_arry)

        # determine correct index
        slice_idx = np.tile(np.arange(0, k), ceil(len(val_arry)/k))[:len(val_arry)]

        # intiialize dict
        self.folds_dict = {x: [] for x in range(0, k)}

        # map of test keys to lists
        [self.folds_dict[v].append(k) for k, v in OrderedDict(zip(val_arry, slice_idx)).items()]

        # remember number of iters
        self.iters = k

    def _return_cv_iter(self, k):
        """
        returns the iterable for train and testing keys
        """
        self.curr_iter = 0

        while self.curr_iter < self.iters:
            # make a last of train keys
            train_lsts = [v for k, v in self.folds_dict.items() if k != self.curr_iter]
            train = np.array([item for sublist in train_lsts for item in sublist])

            # make a list of test keys
            test = np.array(self.folds_dict[self.curr_iter])

            # incriment
            self.curr_iter += 1

            yield train, test

    def run_cv(self, data):
        """
        INPUTS:
            data:
                the hdf5 file path to load
            k:
                the number K cross folds
        EFFECT:
            trains models, tests models, and writes evaluation information
        """

        # initialize data splits
        if self.settings['single_run']:
            self._initialize_split(data, self.settings['VALIDATION_PROP'])
        else:
            self._initialize_folds(data, self.settings['CROSS_VALIDATION'])

        # initialize output dict
        rslt_df = pd.DataFrame()

        for curr_train, curr_test in self._return_cv_iter(self.iters):
            # label information
            print("Iteration: {}".format(self.curr_iter))

            curr_iter_name = "iter_{}".format(self.curr_iter)
            iter_path = self.settings['write_path'] + "/" + curr_iter_name

            # create path
            os.makedirs(iter_path + "/specs")

            # reinitialize model
            temp_model = self.model_func()

            # make log iter path
            log_iter_path = re.search('[^/]+$', self.settings['write_path']).group(0)
            log_iter_path = log_iter_path + "/" + curr_iter_name

            # save settings
            with open("{}/specs/input_settings.json".format(iter_path), 'w') as fp:
                json.dump(self.settings, fp)

            # save model structure
            with open("{}/specs/model_structure.json".format(iter_path), 'w') as fp:
                json.dump(temp_model.to_json(), fp)

            # reinitialize callbacks
            model_check = ModelCheckpoint(filepath="{}/specs/model.hdf5".format(iter_path))
            dist_callback = HeatmapLocalizationCallBack(
                data=data,
                test_keys=curr_test,
                settings=self.settings,
                iter_path=log_iter_path,
                func_lst=["mean_dist_anatomies", "tensorboard_images"],
            )

            # get full paths for training and tessting data
            full_training_paths = get_full_study_paths(data, curr_train)
            full_testing_paths = get_full_study_paths(data, curr_test)

            # fit model
            temp_model.fit_generator(
                generator=data_generator(data, full_training_paths, self.settings),
                steps_per_epoch=len(full_training_paths)/self.settings["BATCH_SIZE"],
                epochs=self.settings["EPOCHS"],
                verbose=1,
                callbacks=[
                    dist_callback,
                    model_check,
                ],
            )

    def write_results(self):
        """
        writes out summaries
        """

        # set prop thresh
        prop_thresh = self.settings["DIST_THRESH"]

        # initialize data frame
        stats_df = pd.DataFrame()

        # stats for distance
        stats_df['median_dist'] = self.eval_metrics.groupby(['anatomy', 'iter'])['heatmap_distance'].agg(np.median)
        stats_df['prop_over_{}'.format(prop_thresh)] = self.eval_metrics.groupby(['anatomy', 'iter'])['heatmap_distance'].agg(lambda x: 100 * sum(x>prop_thresh)/len(x))

        # determine columns to look at
        iter_cols = stats_df.columns

        # reset index
        stats_df = stats_df.reset_index()

        # make a dictionary for results
        summary_dict = {}
        for col in iter_cols:
            summary_dict[col] = pd.DataFrame()
            summary_dict[col]['min'] = stats_df.groupby(['anatomy'])[col].agg(min)
            summary_dict[col]['median'] = stats_df.groupby(['anatomy'])[col].agg(np.median)
            summary_dict[col]['max'] = stats_df.groupby(['anatomy'])[col].agg(max)

        # write
        [v.to_csv("{}/{}_results.csv".format(self.settings['write_path'], k), index=False) for k, v in summary_dict.items()]

        # write histograms
        create_histograms(self.eval_metrics, self.settings['write_path'], "heatmap_distance")

        # write actual individual results
        self.eval_metrics.to_csv(self.settings['write_path'] + "/result_data.csv", index=False)

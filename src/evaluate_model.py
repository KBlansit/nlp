#!/usr/bin/env python

# import libraries
import os
import itertools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from math import ceil
from itertools import product
from multiprocessing import Pool
from functools import reduce, partial

# user defined libraries
from src.image_buffer import image_heatmap
from src.metrics import heatmap_distance, max_index
from src.data_generator import get_batched_hdf5, data_generator
from src.utility import create_matrix_index, write_image_buffer, shorten_anatomy_name

def calculate_metrics(data, test_paths, pred_mtx, settings, curr_iter):
    """
    INPUTS:
        data:
            the dhf5 object
        test_paths:
            the list of test keys to use
        pred_mtx:
            the predicted matrix
        settings:
            the setting dict
        curr_iter:
            the current iteration
    OUTPUT:
        a pandas dataframe of the evualation metrics
    """
    # get actual matrix
    actual_mtx = get_batched_hdf5(data, test_paths)[1]

    # initialize result dataframe
    rslt_df = pd.DataFrame()

    # keep around for easier access
    curr_anatomies = settings["landmark_order"]

    # determine matrix shape
    mtx_shape = [
        settings["INPUT_Y_SIZE"],
        settings["INPUT_X_SIZE"],
    ]

    # loop over anatomies
    for anatomy_indx in range(len(curr_anatomies)):
        curr_actual = create_matrix_index(actual_mtx, anatomy_indx = anatomy_indx)
        curr_pred = create_matrix_index(pred_mtx, anatomy_indx = anatomy_indx)

        max_actual = max_index(curr_actual)
        max_pred = max_index(curr_pred)

        tmp_df = pd.DataFrame({
            "name": test_paths,
            "anatomy": curr_anatomies[anatomy_indx],
            "heatmap_distance": heatmap_distance(max_actual, max_pred),
        })

        rslt_df = rslt_df.append(tmp_df, ignore_index=True)

    # append iteration
    rslt_df['iter'] = curr_iter

    return rslt_df

def image_handler(input_mtx, output_mtx, pred_mtx, write_path_lst, settings, index_tuple):
    """
    INPUTS:
        input_mtx:
            the image matrix
        output_mtx:
            the target heatmap
        pred_mtx:
            the prediction matrix; if None, only show input data
        write_path_lst:
            list of paths to write
        settings:
            the settings file
        index_tuple:
            tuple:
                [0]: anatomy_indx
                [1]: study_indx
    EFFECT:
        writes image
    """
    # if 2D, set z_slc to None
    if not "INPUT_Z_SIZE" in settings.keys():
        #get max index
        z_slc = None

    # if 3D, need z_slc
    else:
        tmp_mtx = create_matrix_index(
            output_mtx,
            study_indx=index_tuple[1],
            anatomy_indx=index_tuple[0],
        )
        z_slc = np.unravel_index(tmp_mtx.argmax(), tmp_mtx.shape)[2]

    # make image buffer
    img_buffer = image_heatmap(
        anatomy_indx = index_tuple[0],
        input_mtx = np.squeeze(input_mtx),
        output_mtx = output_mtx,
        pred_mtx = pred_mtx,
        study_indx = index_tuple[1],
        z_indx = z_slc,
        settings = settings,
    )

    # write image
    write_image_buffer(
        img_buffer,
        "{}/{}.png".format(
            write_path_lst[index_tuple[1]],
            settings['landmark_order'][index_tuple[0]]
        ),
    )

def write_images(data, test_paths, pred_mtx, settings, iter_path):
    """
    INPUTS:
        data:
            the dhf5 object
        test_paths:
            the list of test keys to use
        pred_mtx:
            the predicted matrix
        settings:
            the setting dict
        iter_path:
            the current iteration
    EFFECT:
        writes the output_images
    """
    # communicate current status
    print("Exporting images:")

    # parse settings
    if settings['images'] == 'none':
        return
    elif settings['images'] == '5':
        curr_test_paths = np.random.choice(test_paths, 5, replace=False)
    elif settings['images'] == '20':
        curr_test_paths = np.random.choice(test_paths, 20, replace=False)
    elif settings['images'] == 'all':
        curr_test_paths = test_paths
    else:
        raise AssertionError("Image settings can only be none, 5, 20, or all. Got {}".format(settings['images']))

    # get input and output images
    input_mtx, output_mtx = get_batched_hdf5(data, curr_test_paths)

    # construct directories
    [os.makedirs("{}/{}".format(iter_path, x)) for x in curr_test_paths]

    # construct write paths
    curr_write_paths = ["{}/{}".format(iter_path, x) for x in curr_test_paths]

    # keep around for easier access
    curr_anatomies = settings["landmark_order"]

    # pool multiprocessing
    p = Pool()

    # make partial
    func = partial(
        image_handler, # main function
        input_mtx,
        output_mtx,
        pred_mtx,
        curr_write_paths,
        settings,
    )

    # loop over anatomies and studies
    loop_lst = [curr_anatomies, curr_write_paths]
    iters = product(*[range(len(x)) for x in loop_lst])

    # run multiprocessing function
    p.map(func, iters)

    # clean up
    p.close()
    p.join()

def create_histograms(df, write_path, col_name):
    """
    INPUTS:
        df:
            the input dataframe
        write_path:
            the path to write to
    EFFECT:
        creates histograms for data
    """
    anatomies = df.anatomy.unique().tolist()
    iters = df.iter.unique().tolist()

    # do per iter per anatomy
    # axs indexed by [row, col]
    fig, axs = plt.subplots(
        nrows=len(iters) + 1, # number of iterations plus one for all
        ncols=len(anatomies), # number of anatomy
        sharey=True,
        sharex=True,
    )

    # set row names
    row_names = ["CV: {}".format(x + 1) for x in range(len(iters))] # https://i.imgur.com/ehiodI5.png
    row_names.insert(len(row_names), "All")
    [ax.set_ylabel(row, rotation=90, size='medium') for ax, row in zip(axs[:,0], row_names)]

    # set column names
    [ax.set_title(col) for ax, col in zip(axs[0], shorten_anatomy_name(anatomies))]

    # loop through anatomies and iters
    for i, j in product(*[range(len(x)) for x in [anatomies, iters]]):
        # determine current anatomy
        curr_anatomy = anatomies[i]

        # determine current iter
        curr_iter = iters[j]

        # get vals
        vals = df.loc[(df['anatomy'] == curr_anatomy) & (df['iter'] == curr_iter)][col_name]
        sns.distplot(vals, ax=axs[j, i])

        # do it for all values
        vals = df.loc[(df['anatomy'] == curr_anatomy)][col_name]
        sns.distplot(vals, ax=axs[len(iters), i])

    # save figure
    fig.savefig(write_path + "/" + col_name + ".png")

def evualate_model(model, data, test_paths, settings):
    """
    """

    # predict on testing data
    pred_mtx = temp_model.predict_generator(
        data_generator(data, test_paths, settings),
        steps=ceil(len(test_paths)/settings["BATCH_SIZE"]),
    )

    # write out images
    write_images(data, test_paths, pred_mtx, settings, iter_path)

    # return evualation data frame
    self.eval_metrics = self.eval_metrics.append(
        calculate_metrics(data, full_testing_paths, pred_mtx, self.settings, self.curr_iter),
        ignore_index=True,
    )

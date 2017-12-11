#!/ussr/bin/env python

# import libraries
import h5py
import json

import numpy as np

from math import ceil, floor

# import user defined libraries
from src.utility import recursive_flattten
from src.image_augmentation import image_augment

def load_data(data_path):
    """
    INPUTS:
        data_path:
            the path for the HD5F file
    OUTPUT:
        the data
    """

    # read in file
    return h5py.File(data_path + "/data.hdf5", "r")

def get_full_study_paths(data, curr_keys, testing=False):
    """"
    INPUTS:
        curr_values:
            list of the current study-series keys to use
        data:
            the h5py data object
        testing:
            if we are testing, choose a single image from each series
    OUTPUT:
        list of study paths
    """
    # note:
    # data for cMRI scans grouped by study -> series -> image
    # therefore want to use all series and images per study

    # initialize list
    mtx_path_lst = []

    # create list of paths
    for study_series in curr_keys:
        frame_lst = list(data[study_series].keys())
        frame_lst = ["{}/{}".format(study_series, x) for x in frame_lst]
        if testing:
            frame_lst = [np.random.choice(frame_lst)]
        mtx_path_lst.append(frame_lst)

    # flatten
    mtx_path_lst = list(recursive_flattten(mtx_path_lst))

    # shuffle
    np.random.shuffle(mtx_path_lst)

    return mtx_path_lst

def validation_data(data, train_prop):
    """
    INPUTS:
        data:
            the h5py data object
        train_prop:
            the proprotion of data to use in training data set
    OUTPUT:
        tuple:
            0: training study id keys
            1: testing study id keys
    """
    # determine values
    val_arry = np.array(list(data.keys()))
    np.random.shuffle(val_arry)

    # determine train and test vals
    train_idx = ceil(train_prop * val_arry.shape[0])

    train_vals = val_arry[:train_idx]
    test_vals = val_arry[train_idx:]

    return train_vals, test_vals

def get_batched_hdf5(data, curr_paths, settings=None):
    """"
    INPUTS:
        data:
            the h5py data object
        curr_values:
            list of the current study keys to use
        settings:
            the setting dictionary
    OUTPUT:
        tuple:
            0: stacked input_mtx
            1: stacked output_mtx
    """

    # return lists of input and output
    input_lst = [data[x + "/input"][()] for x in curr_paths]
    output_lst = [data[x + "/output"][()] for x in curr_paths]
    dicom_info_lst = [data[x + "/dicom_info"][()] for x in curr_paths]

    # reshape matrix
    input_lst = [np.expand_dims(x.astype('float32'), axis=-1) for x in input_lst]
    output_lst = [x.astype('float32') for x in output_lst]
    dicom_info_lst = [json.loads(x) for x in dicom_info_lst]

    # numebr of iterations
    iters = range(len(curr_paths))

    # apply image augmentations
    input_lst = [image_augment(input_lst[x], dicom_info_lst[x], settings) for x in iters]
    output_lst = [image_augment(output_lst[x], dicom_info_lst[x], settings) for x in iters]

    input_rtn = np.stack(input_lst, axis=0)
    output_rtn = np.stack(output_lst, axis=0)

    return input_rtn, output_rtn

def data_generator(data, mtx_path_lst, settings):
    """"
    INPUTS:
        data:
            the h5py data object
        mtx_path_lst:
            the list of paths to use
        batch_size:
            the number of datasets used
   OUTPUT:
        tuple:
            0: input into network
            1: output into network
    """
    # get batch size
    batch_size = settings["BATCH_SIZE"]

    # convert mtx_path_lst into numpy array
    mtx_path_lst = np.array(mtx_path_lst)

    # make batch sized indicies
    min_cuts = floor(len(mtx_path_lst)/ batch_size)
    slices = np.arange(0, min_cuts*batch_size).reshape(min_cuts, batch_size).tolist()

    # if there's a remainder, append at end of list
    if len(mtx_path_lst) % batch_size:
        slices.append(np.arange(min_cuts*batch_size, len(mtx_path_lst)).tolist())

    # loop through values
    while 1:
        for curr_idx in slices:
            yield get_batched_hdf5(data, mtx_path_lst[curr_idx], settings)

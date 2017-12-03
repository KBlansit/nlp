#!/usr/bin/env python

# import libraries
import numpy as np
import pandas as pd

# script variables
CHUNK_SIZE = 10 * 6

CCS_CATEGORIES = 2621

def create_data_dir(path, dir_name=False):
    """
    INPUTS:
        path:
            the path to create the output directory
        options:
            either False in the case of no output name given or a str for output name
    EFFECT:
        creates a new directory
    OUTPUT:
        path to new directory
    """

    if not dir_name:
        # create new path of Day Month Year
        write_path = path + "/" + time.strftime("%d-%m-%Y")
    elif type(dir_name) == str:
        # create new path of supplied name
        write_path = path + "/" + dir_name
    else:
        raise AssertionError("{} is not a string or False".format(dir_name))

    # if case output directory exists, make a new one
    if os.path.exists(write_path):
        # determine max iteration
        tmp_path = write_path
        i = 1
        while os.path.exists(tmp_path):
            tmp_path = write_path + " - " + str(i)
            i = i + 1
        # assign tmp to current
        write_path = tmp_path

    # make new directory and return path
    os.makedirs(write_path)
    return write_path

def hot_encode_list(input_lst):
    """
    takes a list and creates a hot encoded matrix
    """
    # zero out mtx
    hot_encode_mtx = np.zeros([CCS_CATEGORIES])

    # encode ccs levels
    hot_encode_mtx[input_lst] = 1

    return hot_encode_mtx

def read_large_csv(file_path):
    """
    reads large dataframe
    """
    # chunk data
    reader = pd.read_csv(file_path, chunksize=CHUNK_SIZE)
    return pd.concat(reader, ignore_index=True)

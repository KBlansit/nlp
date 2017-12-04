#!/usr/bin/env python

# import libraries
import os
import h5py
import math
import argparse
import tempfile

import numpy as np
import pandas as pd

from functools import partial
from multiprocessing import Pool

# import user defined libraries
from initialize_data import DATA_PATH

from src.run_clamp import run_clamp
from src.process_named_entity_files import read_all_processed_files
from src.utility import hot_encode_list, create_data_dir

TEMP_DIR = "tmp_dir"
OUTPUT_DIR = "processed_data"

HOUR_STRETCH = 8

NUM_OF_ITERS = 10

# parse command line args
cmd_parse = argparse.ArgumentParser(description = "Application for making training data")
cmd_parse.add_argument("-d", "--data_path", help = "output data path name", type=str)
cmd_parse.add_argument("-t", "--temp_dir", help = "use a temp dir", type=bool, default=False)
cmd_parse.add_argument("-c", "--overrride_clamp", help = "override temp ", type=bool, default=False)
cmd_parse.add_argument("-u", "--override_umls", help = "overrides umls", type=bool, default=False)
cmd_args = cmd_parse.parse_args()

def read_in_data():
    """
    reads in data from hdf5 file
    """
    # console message
    print("Reading in data")
    note_df = pd.read_hdf(DATA_PATH, "note_df")
    icu_df = pd.read_hdf(DATA_PATH, "icu_df")

    return note_df, icu_df

def merge_note_and_icu_dfs(note_df, icu_df):
    """
    merges two the two dataframe files
    """
    # console message
    print("Merging data")

    # merge data
    merged_df = pd.merge(note_df, icu_df, on = "HADM_ID", how = "inner")

    # remove note chart data before and after icu stay
    merged_df = merged_df[merged_df["CHARTTIME"] < merged_df["OUTTIME"]]
    merged_df = merged_df[merged_df["CHARTTIME"] >= merged_df["INTIME"]]

    # convert HADM_ID to str
    merged_df["HADM_ID"] = merged_df["HADM_ID"].astype(int).astype(str)

    return merged_df

def write_id_notes(curr_df, curr_id, write_dir):
    """
    given a dataframe, creates
    """
    # append write dir
    write_dir = os.path.join(write_dir, str(curr_df.iloc[0]['iter']))

    # write note
    for index, row in curr_df.iterrows():
        f_name = os.path.join(write_dir, "{}_{}.txt".format(curr_id, index))
        pd.Series(row["TEXT"]).to_csv(f_name, index=False)

def process_initial_impression_notes(merged_df, file_path):
    """
    takes first 8 hours of notes and writes out for NLP pipeline
    """
    # console message
    print("Processing initial data")
    # subset data to get initial impression
    initial_df = merged_df[merged_df["CHARTTIME"] <= merged_df["INTIME"] + pd.Timedelta(hours=HOUR_STRETCH)]

    # make iter index
    initial_df["iter"] = (initial_df.reset_index().index % NUM_OF_ITERS)

    # move to dict
    df_dict = {}
    for curr_id in initial_df["HADM_ID"].unique():
        df_dict[curr_id] = initial_df[initial_df["HADM_ID"] == curr_id].reset_index()

    # make output dirs
    for curr_sub_dir in range(0, NUM_OF_ITERS):
        curr_sub_dir = os.path.join(file_path, str(curr_sub_dir))
        if not os.path.exists(curr_sub_dir):
            os.makedirs(curr_sub_dir)

    # console message
    print("Writing notes")

    # write out clinical notes
    [write_id_notes(v, k, file_path) for k, v in df_dict.items()]

def async_clamp_run(input_file_base):
    """
    async runs clamp
    """
    # create list of files to iterate over
    iter_lst = [os.path.join(input_file_base, str(x)) for x in range(0, NUM_OF_ITERS)]

    # create directory to store results
    output_dir = os.path.join(input_file_base, "named_entities")
    os.makedirs(output_dir)

    # create func
    func = partial(run_clamp, output_dir)

    # create pool object
    p = Pool(NUM_OF_ITERS)

    # map function
    p.map(func, iter_lst)

    # clean up
    p.close()

def create_input_matrix(id_dict, icu_df):
    """
    takes dictionary of
    """
    # store as tuple
    id_tuples = [(k, v) for k, v in id_dict.items()]

    # hot encode
    encode_lst = [hot_encode_list(x[1]) for x in id_tuples]

    # turn into matrix
    all_x = np.vstack(encode_lst)

    # get ids into a list
    id_series = pd.DataFrame({"HADM_ID": [int(x[0]) for x in id_tuples]})

    # turn into
    all_y = pd.merge(id_series, icu_df, how='left')['LOS'].values

    return all_x, all_y

def main():
    """
    main function
    """
    # check all data is present
    assert os.path.isfile(DATA_PATH)

    # read in data
    note_df, icu_df = read_in_data()

    # override existing clamp
    if cmd_args.overrride_clamp:
        # make temp file
        if cmd_args.temp_dir:
            tmp_d = tempfile.TemporaryDirectory(dir=TEMP_DIR)
            tmp_path = tmp_d.name
        else:
            tmp_path = TEMP_DIR

        # merge data
        merged_df = merge_note_and_icu_dfs(note_df, icu_df)

        # process merge data
        process_initial_impression_notes(merged_df, tmp_path)

        # run clamp
        async_clamp_run(tmp_path)
    else:
        tmp_path = TEMP_DIR

    # process named entity files
    id_dict = read_all_processed_files(tmp_path)

    # write new data set and return object
    if cmd_args.data_path is None:
        write_path = create_data_dir(OUTPUT_DIR, 0)
    else:
        write_path = create_data_dir(OUTPUT_DIR, cmd_args.output)

    # make new data directory
    data_file = h5py.File(write_path + "/data.hdf5", "w")

    # make training data
    all_x, all_y = create_input_matrix(id_dict, icu_df)

    # move to hd5f
    data_file['all_x'] = all_x
    data_file['all_y'] = all_y

    # close file
    data_file.close()

if __name__ == '__main__':
    main()

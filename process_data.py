#!/usr/bin/env python

# import libraries
import os
import h5py
import math
import tempfile

import numpy as np
import pandas as pd

from functools import partial
from multiprocessing import Pool

# import user defined libraries
from src.run_clamp import run_clamp
from src.utility import read_large_csv, hot_encode_list, create_data_dir

# script variables
DATA_FILES = {
    "icu_df":   "data/ICUSTAYS.csv",
    "notes_df": "data/NOTEEVENTS.csv",
}

NOTE_TYPES = ["Physician "]

ICU_COLS = ["HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME", "LOS"]
NOTE_COLS = ["HADM_ID", "CHARTTIME", "TEXT", "CATEGORY"]

TEMP_DIR = "tmp_dir"
OUTPUT_DIR = "processed_data"

HOUR_STRETCH = 8

NUM_OF_ITERS = 10

# parse command line args
cmd_parse = argparse.ArgumentParser(description = 'Application for making training data')
cmd_parse.add_argument('-o', '--output', help = 'output data directory name', type=str)
cmd_args = cmd_parse.parse_args()

def read_icu():
    """
    reads icu information
    """
    # console message
    print("Reading in icu information")

    # read in data
    icu_df = pd.read_csv(DATA_FILES["icu_df"])

    # convert to datetime
    icu_df["INTIME"] = pd.to_datetime(icu_df["INTIME"])
    icu_df["OUTTIME"] = pd.to_datetime(icu_df["OUTTIME"])

    # use longest LOS if multiple
    icu_df = icu_df.sort_values("LOS", ascending=False).groupby("HADM_ID", as_index=False).first()

    return icu_df[ICU_COLS]

def read_clinical_notes():
    """
    reads and validates notes
    """
    # console message
    print("Reading in clinical notes")

    # read in data
    note_df = read_large_csv(DATA_FILES["notes_df"])

    # convert to datetime
    note_df["CHARTTIME"] = pd.to_datetime(note_df["CHARTTIME"])

    # subset on note type and remove error notes
    note_df = note_df[note_df["CATEGORY"].isin(NOTE_TYPES)]
    node_df = note_df[note_df["ISERROR"] != 1]

    return note_df[NOTE_COLS]

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

def process_initial_impression_notes(merged_df):
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

    # make temp file
    tmp_d = tempfile.TemporaryDirectory(dir=TEMP_DIR)

    # make output dirs
    for curr_sub_dir in range(0, NUM_OF_ITERS):
        curr_sub_dir = os.path.join(tmp_d.name, str(curr_sub_dir))
        if not os.path.exists(curr_sub_dir):
            os.makedirs(curr_sub_dir)

    # console message
    print("Writing notes")

    # write out clinical notes
    [write_id_notes(v, k, tmp_d.name) for k, v in df_dict.items()]

    return tmp_d

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
    id_series = pd.Series({'HADM_ID': [x[0] for x in id_tuples]})

    # turn i
    all_y = pd.merge(id_series, icu_df, how='left')['LOS'].values

    return all_x, all_y

def main():
    """
    main function
    """
    # check all data is present
    for f_path in DATA_FILES.values():
        assert os.path.isfile(f_path)

    for f_path in MAPPING_FILES.values():
        assert os.path.isfile(f_path)

    # read in data
    note_df = read_clinical_notes()
    icu_df = read_icu()

    # merge data
    merged_df = merge_note_and_icu_dfs(note_df, icu_df)

    # process merge data
    tmp_d = process_initial_impression_notes(merged_df)

    # run clamp
    async_clamp_run(tmp_d.name)

    # process named entity files
    id_dict = read_all_processed_files(tmp_d.name)

    # write new data set and return object
    if cmd_args.output is None:
        write_path = create_data_dir(OUTPUT_DIR, 0)
    else:
        write_path = create_data_dir(OUTPUT_DIR, cmd_args.output)

    # make new data directory
    data_file = h5py.File(write_path + "/data.hdf5", "w")

    # make training data
    all_x, all_y = create_input_matrix(id_dict, merged_df)

    # move to hd5f
    data_file['all_x'] = all_x
    data_file['all_y'] = all_y

    # close file
    data_file.close()

if __name__ == '__main__':
    main()

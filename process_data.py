#!/usr/bin/env python

# import libraries
import os
import math
import tempfile

import numpy as np
import pandas as pd

# import user defined libraries
from src.import_data import read_large_csv

# script variables
DATA_FILES = {
    "icu_df":   "data/ICUSTAYS.csv",
    "notes_df": "data/NOTEEVENTS.csv",
}

MAPPING_FILES = {
    "ccs_mapping": "ccs_dx_icd10.csv",
}

NOTE_TYPES = ["Physician "]

ICU_COLS = ["HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME", "LOS"]
NOTE_COLS = ["HADM_ID", "CHARTTIME", "TEXT", "CATEGORY"]

def write_id_notes(curr_df, curr_id, write_dir = None):
    """
    given a dataframe, creates
    """
    # convert None to '' if necessary
    write_dir = '' if write_dir is None else "{}/".format(write_dir)

    # write note
    for index, row in curr_df.iterrows():
        pd.Series(row["TEXT"]).to_csv("{}{}_{}.csv".format(write_dir, curr_id, index), index=False)

def process_initial_impression_notes(merged_df):
    """
    takes first 8 hours of notes and writes out for NLP pipeline
    """
    # console message
    print("Processing initial data")
    # subset data to get initial impression
    initial_df = merged_df[merged_df["CHARTTIME"] <= merged_df["INTIME"] + pd.Timedelta(hours=8)]

    # move to dict
    df_dict = {}
    for curr_id in initial_df["HADM_ID"].unique():
        df_dict[curr_id] = initial_df[initial_df["HADM_ID"] == curr_id].reset_index()

    # write out clinical notes
    [write_id_notes(v, k, "tmp") for k, v in df_dict.items()]

def read_icu():
    """
    reads icu information
    """
    # console message
    print("Reading in icu information")

    # read in data
    icu_df = read_large_csv(DATA_FILES["icu_df"])

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

def main():
    """
    main function
    """
    # check all data is present
    for f_path in DATA_FILES.values():
        assert os.path.isfile(f_path)

    for f_path in MAPPING_FILES.values():
        assert os.path.isfile(f_path)

if __name__ == '__main__':
    main()

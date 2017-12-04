#!/usr/bin/env python

# import libraries
import os
import h5py

import pandas as pd

# import user defined libraries
from src.utility import read_large_csv

# script variables
DATA_FILES = {
    "icu_df":   "data/ICUSTAYS.csv",
    "notes_df": "data/NOTEEVENTS.csv",
}

DATA_PATH = "data/data_files.hdf"

NOTE_TYPES = ["Physician "]

ICU_COLS = ["HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME", "LOS"]
NOTE_COLS = ["HADM_ID", "CHARTTIME", "TEXT", "CATEGORY"]

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
    note_df = note_df[note_df["ISERROR"] != 1]

    return note_df[NOTE_COLS]

if __name__ == '__main__':
    # read in data
    note_df = read_clinical_notes()
    icu_df = read_icu()

    # remove old file if necessary
    if os.path.exists(DATA_PATH):
        os.remove(DATA_PATH)

    # create new hdf5 file
    data_f = h5py.File("data/data_files.hdf5", "w")

    # move to hdf5 file
    note_df.to_hdf(DATA_PATH, "note_df")
    icu_df.to_hdf(DATA_PATH, "icu_df")

    # close
    data_f.close()

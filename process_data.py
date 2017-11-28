#!/usr/bin/env python

# import libraries
import os
import tempfile
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

"""
NOTES
- Must cut off notes that occur after discharge
- Must cut off notes that occur before
- Uses longest ICU stay if multiple
ideas:
    use all valid types of notes within first x hours (multiple of 6?)

 """

 HOUR_PERIODICITY = 8

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

#!/usr/bin/env python

# import libraries
import os
import pandas as pd

# import user defined libraries
from src.import_data import read_large_csv

# script variables
DATA_FILES = {
    "notes_df": "data/NOTEEVENTS.csv",
}

MAPPING_FILES = {
    "ccs_mapping": "ccs_dx_icd10.csv"
}

def main():
    """
    main function
    """
    # check all data is present
    for f_path in DATA_FILES.values():
        assert os.path.isfile(f_path)

    for f_path in MAPPING_FILES.values():
        assert os.path.isfile(f_path)

    # get los data
    df_icu = read_large_csv("data/ICUSTAYS.csv")
    icu_cols = ["HADM_ID", "INTIME", "OUTTIME", "LOS"]

if __name__ == '__main__':
    main()

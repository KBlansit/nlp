#!/usr/bin/env python
import pandas as pd

# globals
CHUNK_SIZE = 10 * 6
DATA_FILES = {
    notes_df: "data/NOTEEVENTS.csv",
}

def read_large_csv(file_name):
    """
    reads large dataframe
    """

    # chunk data
    reader = pd.read_csv(file_name, chunksize=CHUNK_SIZE)
    return pd.concat(reader, ignore_index=True)

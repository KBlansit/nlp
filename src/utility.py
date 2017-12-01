#!/usr/bin/env python

# import libraries
import pandas as pd

# globals
CHUNK_SIZE = 10 * 6

def read_large_csv(file_path):
    """
    reads large dataframe
    """

    # chunk data
    reader = pd.read_csv(file_path, chunksize=CHUNK_SIZE)
    return pd.concat(reader, ignore_index=True)

#!/usr/bin/env python

# import libraries
import os
import yaml
import pickle
import xgboost
import tempfile

import numpy as np
import pandas as pd

# user defined libraries
from src.run_clamp import run_clamp
from src.utility import hot_encode_list
from src.process_named_entity_files import read_all_processed_files

INPUT_DIR = "input_dir"
VALID_IDX_PATH = "models/valid_idx.csv"
MODEL_PATH = "models/model.xgb"

def main():
    """
    main inferecne function
    """
    # read in model
    print("Reading in data")
    model = pickle.load(open("models/model.xgb", "rb"))
    with open("models/time_dict.yaml", "r") as f:
        time_dict = yaml.load(f)
    vld_idx = pd.read_csv(VALID_IDX_PATH)

    # make temp dir
    tmp_d = tempfile.TemporaryDirectory()

    # join paths
    output_dir = os.path.join(tmp_d.name, "named_entities")

    # make output directory
    os.makedirs(output_dir)

    # run clamp pipeline
    run_clamp(output_dir, INPUT_DIR)

    # process named entities
    print("Encoding")
    id_dict = read_all_processed_files(tmp_d.name)

    # encode list
    print("Predicted length of stay in ICU:")
    encode_lst = [hot_encode_list(x) for x in id_dict.values()]
    input_x = np.expand_dims(encode_lst[0][vld_idx["vld_idx"]], axis=0)

    # inference
    predicted_probs = model.predict_proba(input_x)[0]

    # iterate over probs
    for i in range(len(time_dict.keys())):
        print("{}: {}%".format(time_dict[str(i)], round(100 * predicted_probs[i])))

if __name__ == '__main__':
    main()

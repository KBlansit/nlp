#!/usr/bin/env python

# import libraries
import numpy as np
import pandas as pd

# script variables
GOLD_STD_DATA_PATH = "data/DIAGNOSES_ICD.csv"
ICD_9_MAP = "mappings/icd9_ccs_mapping.csv"

def load_icd9_data(hadm_id_lst):
    """
    loads data
    """
    diags = pd.read_csv(GOLD_STD_DATA_PATH)

    icd_map = pd.read_csv(ICD_9_MAP)
    icd_map["ICD-9-CM_CODE"] = icd_map['ICD-9-CM_CODE'].str.strip()

    gld_std_ccs = pd.merge(
        diags[diags["HADM_ID"].isin(hadm_id_lst)][["HADM_ID", "ICD9_CODE"]],
        icd_map[["ICD-9-CM_CODE", "CCS_CATEGORY"]],
        left_on="ICD9_CODE",
        right_on="ICD-9-CM_CODE",
        how = "inner",
    )[["HADM_ID", "CCS_CATEGORY"]]

    gld_dict = {}

    for curr_id in gld_std_ccs["HADM_ID"].unique():
        curr_df = gld_std_ccs[gld_std_ccs["HADM_ID"] == curr_id]
        gld_dict[curr_id] = curr_df["CCS_CATEGORY"].unique().tolist()

    return gld_dict

if __name__ == '__main__':
    main()

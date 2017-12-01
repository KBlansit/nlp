#!/usr/bin/env python

# import libraries
import os
import re

import numpy as np

# import user defined libraries
from src.map_cui_codes import CUI_CCS_Mapper

# script variables
PROBLEM_REGEX = re.compile("(?:semantic=problem\\tassertion=present\\tcui=)(C[0-9]+)")

ID_REGEX = re.compile("(^[0-9]+)")

DIRECTORY_NAME = "named_entities"

def read_all_processed_files(file_path):
    """
    reads all processed files to find ccs codes
    """

    # append file directory
    file_path = os.path.join(file_path, DIRECTORY_NAME)

    # list all files
    all_files = os.listdir(file_path)

    # keep only txt files
    all_files = [x for x in all_files if x.endswith('.txt')]

    # initialize id_dict
    id_dict = {}

    # initialize mapper
    mapper = CUI_CCS_Mapper()

    # iterate over files
    for curr_file in all_files:
        # extract id
        curr_id = ID_REGEX.search(curr_file).group()

        # test to see if already in id_dict
        if curr_id in id_dict.keys():
            ccs_lst = id_dict[curr_id]
        else:
            ccs_lst = []

        # determine full path
        curr_file = os.path.join(file_path, curr_file)

        # opn file
        with open(curr_file) as f:
            # iterate through lines
            for line in f:
                # test against regex
                regex_rslt = PROBLEM_REGEX.search(line)
                if regex_rslt:
                    cui_code = regex_rslt.group()
                    ccs_lst = ccs_lst + mapper.get_ccs_codes(cui_code)

        # find unique ccs codes
        ccs_lst = list(set(ccs_lst))

        # return to dictionary
        id_dict[curr_id] = ccs_lst

    return id_dict

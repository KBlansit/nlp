#!/usr/bin/env python

# import libraries
import re
import yaml
import requests

import numpy as np
import pandas as pd

from bs4 import BeautifulSoup

# script variables
CONFIG_PATH = "config_parameters.yaml"

TGT_URL = "https://utslogin.nlm.nih.gov/cas/v1/api-key"

UMLSKS_KEY = {"service": "http://umlsks.nlm.nih.gov"}

BASE_CUI_URI = ""

CODE_REGEX = re.compile("([0-9]+\.?[0-9]+$)")

ICD_10_CODE_REGEX = re.compile("([A-Z, a-z, 0-9]+\.?[0-9]+$)")

MAPPING_FILES = {
    "ccs_mapping_icd_10": "mappings/icd10_ccs_mapping.csv",
    "cui_map": "mappings/cui_map.csv",
}

# load API_KEY
with open(CONFIG_PATH, "r") as config_f:
    API_KEY = {"apikey": yaml.load(config_f)["API_KEY"]}

class CUI_CCS_Mapper:
    """
    class for storing map between CUI and CCS
    """
    def __init__(self):
        """
        init staticmethod
        """
        # initialize cui ccs dict map
        self.cui_css_map = {}

        # read in ccs icd maps
        # NOTE: remove first line of mapping file, remove all quotes
        # console message
        print("Loading CCS Mapping Data")

        # read in mapping files
        for k, v in MAPPING_FILES.items():
            setattr(self, k, pd.read_csv(v))

    def make_tgt_request(self):
        """
        get a tft key
        """
        # make post request
        r = requests.post(TGT_URL, data=API_KEY)

        # make sure request is okay
        assert r.ok

        # parse document
        soup = BeautifulSoup(r.text, "html.parser")

        # set target link
        self.target_link = soup.find("form").get("action")

    def make_sst_request(self):
        """
        gets a single time use key
        """
        # make post request
        r = requests.post(self.target_link, data=UMLSKS_KEY)

        # make sure request is okay
        assert r.ok

        return r.text

    def get_icd_10_code(self, cui_code):
        """
        takes cui and returns a list of cui codes
        """
        # get single use token
        sst = self.make_sst_request()

        # format url
        url = "https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{}/atoms?language=ENG&sabs=ICD10CM&ticket={}".format(cui_code, sst)

        # send get request
        r = requests.get(url)

        # initialize output list
        curr_ccs_lst = []

        # check if we get a valid response
        if r.ok:

            # get ccs map
            ccs_map = self.ccs_mapping_icd_10
            code_col = "ICD-10-CM_CODE"

            # iterate through results
            for rslt_code in r.json()["result"]:
                # get icd10 code
                match = ICD_10_CODE_REGEX.search(rslt_code["code"]).groups()[0]

                # strip decimal
                match = match.replace(".", "")

                # map to ccs
                ccs_codes = ccs_map[ccs_map[code_col].str.strip() == match]["CCS_CATEGORY"]

                # convert to list
                curr_ccs_lst = curr_ccs_lst + ccs_codes.astype("str").tolist()

        return curr_ccs_lst

    def request_ccs_code(self, cui_code):
        """
        returns list of ccs associated with cui code
        """
        # get single use token
        sst = self.make_sst_request()

        # format url
        url = "https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{}/atoms?sabs=CCS_10&ticket={}".format(cui_code, sst)

        # initialize list of ccs codes
        curr_ccs_lst = []

        # send get request
        r = requests.get(url)

        # check if we get a valid response
        if r.ok:
            # iterate over results
            for rslt_code in r.json()["result"]:
                if rslt_code["termType"] == "SD":
                    curr_ccs_lst.append(CODE_REGEX.search(rslt_code["code"]).group())

        # if cannot directly infer mapping, look though icd 10 code
        else:
            curr_ccs_lst = curr_ccs_lst + self.get_icd_10_code(cui_code)

        return curr_ccs_lst

    def get_ccs_codes(self, cui_code):
        """
        map a cui code to ccs concept codes
        """

        # determine if already defined
        if cui_code in self.cui_css_map.keys():
            return self.cui_css_map[cui_code]
        # else construct request
        else:
            # get codes
            rslt_df = self.cui_map[self.cui_map["CUI"] == cui_code]

            # determine if we have any ccs codes
            ccs_indx = (rslt_df["SAB"] == "CCS_10") & (rslt_df["TTY"] == "SD")
            icd_10_indx = rslt_df["SAB"] == "ICD10CM"

            if sum(ccs_indx):
                # directly read codes
                ccs_lst = rslt_df[ccs_indx]["CODE"].values.astype('int').tolist()
            elif sum(icd_10_indx):
                # indirectly read codes
                icd_codes = np.unique(rslt_df[icd_10_indx]["CODE"].values).tolist()

                # remove period
                icd_codes = [x.replace(".", "") for x in icd_codes]

                ccs_icd10_df = pd.merge(
                    pd.DataFrame({"ICD-10-CM_CODE": icd_codes}),
                    self.ccs_mapping_icd_10,
                    how = "inner",
                )

                # determine if we have valid rows
                if ccs_icd10_df.shape[0]:
                    ccs_lst = ccs_icd10_df["CCS_CATEGORY"].values.astype('int').tolist()
                # if icd10 codes associated with ccs, no codes
                else:
                    ccs_lst = []
            # if no ccs codes and no back mapping through icd10, no code
            else:
                ccs_lst = []

            # remove duplicates
            ccs_lst = list(set(ccs_lst))

            # add ccs codes to dictionary
            self.cui_css_map[cui_code] = ccs_lst

            return ccs_lst

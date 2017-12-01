#!/usr/bin/env python

# import libraries
import re
import requests

from bs4 import BeautifulSoup

# script variables
CONFIG_PATH = "config_parameters.yaml"

TGT_URL = "https://utslogin.nlm.nih.gov/cas/v1/api-key"

UMLSKS_KEY = {'service':'http://umlsks.nlm.nih.gov'}

CODE_REGEX = re.compile("([0-9]+\.?[0-9]+$)")

MAPPING_FILES = {
    "ccs_mapping_icd_9": "mappings/icd9_ccs_mapping.csv",
    "ccs_mapping_icd_10": "mappings/icd10_ccs_mapping.csv",
}

ICD_TYPES = [9, 10]

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
        for k, v in MAPPING_FILES.itme():
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
        soup = BeautifulSoup(r.text, 'html.parser')

        # set target link
        self.target_link = soup.find('form').get('action')

    @staticmethod
    def make_sst_request(target_link):
        """
        gets a single time use key
        """
        # make post request
        r = request.post(target_link, data=UMLSKS_KEY)

        # make sure request is okay
        assert r.ok

        return r.text

    def icd_code_ccs_mapper(self, cui_code, code_type):
        """
        maps icd code to ccs code
        """
        # validate code
        if code_type not in ICD_TYPES:
            raise AssertionError("{} is not a valid code type".format(code_type))

        # get single use token
        sst = self.make_sst_request(target_link)

        # send get request
        r = requests.get()

        # determine map type
        curr_ccs_map = getattr(self, "ccs_mapping_icd_{}".format(code_type))

        # determine code type
        code_type = "ICD-{}-CM_CODE".format(code_type)

        # format urls
        url = "https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{}\
            /atoms?language=ENG&sabs=ICD{}CM&ticket={}".format(cui_code, code_type, sst)

        # initialize list of ccs codes
        curr_css_lst = []

        # iterate over results
        for rslt_code in r.json()['results']:
            # get icd code
            icd_code = CODE_REGEX.search(rslt_code['code']).group().replace('.', '')

            # get ccs code
            ccs_lst.append(ccs_map[ccs_map[] == icd_code]["CCS_CATEGORY"].values[0])

        return curr_ccs_lst

    def get_ccs_code(self, cui_code):
        """
        map a cui code to ccs concept codes
        """

        # determine if already defined
        if cui_code in self.cui_css_map.keys():
            return self.cui_css_map["cui_code"]
        # else construct request
        else:
            # get a new tgt key if necessary
            if not hasattr(self, target_link):
                self.make_tgt_request()

            # iterate over icd types
            ccs_lst = [self.icd_code_ccs_mapper(cui_code, x) for x in ICD_TYPES]

            # unlist list of lists
            ccs_lst = [x for y in ccs_lst for x in y]

            # find unique codes
            ccs_lst = list(set(ccs_lst))

            # add ccs codes to dictionary
            self.cui_css_map[cui_code] = ccs_lst

            return ccs_lst

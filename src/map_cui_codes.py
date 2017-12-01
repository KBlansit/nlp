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

        # read in ccs icd9 map
        # NOTE: remove first line of mapping file, remove all quotes
        # console message
        print("Loading CCS Mapping Data")

        # read file
        self.ccs_map = pd.read_csv(file_path)

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

    def icd_9_code_ccs_mapper(self, icd_9_code):
        """
        maps icd 9 code to ccs code
        """
        # find code map
        return ccs_map[ccs_map['ICD-9-CM_CODE'] == icd_9_code][' CCS_CATEGORY'].values[0]

    def get_icd_9_code(cui_code):
        """
        map a cui code to ccs concept codes
        """

        # determine if already defined
        if cui_code in self.cui_css_map.keys():
            return self.cui_css_map['cui_code']
        # else construct request
        else:
            # get a new tgt key if necessary
            if not hasattr(self, target_link):
                self.make_tgt_request()

            # get single use token
            sst = make_sst_request(target_link)

            # format url
            url = "https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{}\
                /atoms?language=ENG&sabs=CCS&ticket={}".format(cui_code, sst)

            # make get request
            r = requests.get(url)

            # make sure request is okay
            assert r.ok

            # iterate over results
            ccs_lst = []
            for rslt_code in r.json()['results']:
                icd_9_code = CODE_REGEX.search(rslt_code['code']).group().replace('.', '')
                ccs_lst.append(self.icd_9_code_ccs_mapper(icd_9_code))

            # find unique codes
            ccs_lst = list(set(ccs_lst))

            # add ccs codes to dictionary
            self.cui_css_map[cui_code] = ccs_lst

            return ccs_lst

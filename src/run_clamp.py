#!/usr/bin/env python
import os
import yaml

INPUT_DIR = "input"
OUTPUT_DIR = "output"

def run_clamp():
    """
    runs clamp
    """
    # read in umls config data
    yaml.read()

    # read in umls config data
    open(path, "r") as config_f:
        config = yaml.load(config_f)

    # initialize cmd_dict
    cmd_dict = {}

    # input and output information
    cmd_dict['input_cmd'] = INPUT_DIR
    cmd_dict['output_cmd'] = OUTPUT_DIR

    # clamp information
    cmd_dict['clamp_bin'] = "bin/clamp-nlp-1.3.1-jar-with-dependencies.jar"
    cmd_dict['pipeline'] = "pipeline/clamp-ner-attribute.pipeline.jar"
    cmd_dict['umls_index'] = "resource/umls_index/"

    # umls information
    cmd_dict['umls_name'] = config["UMLS_USER_NAME"]
    cmd_dict['umls_pass'] = config["UMLS_USER_PASS"]

    cmd = 'java -DCLAMPLicenceFile="CLAMP.LICENSE" -Xmx2g -cp {clamp_bin} edu.uth.clamp.nlp.main.PipelineMain'.format(**cmd_dict)
    cmd = cmd + " \
        -i {input}\
        -o {output}\
        -p {pipeline}\
        -U {umls_name}\
        -P {umls_pass}\
        -I {umls_index}\
    ".format(**cmd_dict)

    os.system(cmd)

#!/usr/bin/env python
import os
import yaml

INPUT_DIR = "/home/kblansit/nlp/ClampCmd_1.3.1/input"
OUTPUT_DIR = "ClampCmd_1.3.1/output"

CONFIG_PATH = "config_parameters.yaml"

def run_clamp():
    """
    runs clamp
    """
    # console message
    print("Running Clamp")

    # read in umls config data
    with open(CONFIG_PATH, "r") as config_f:
        config = yaml.load(config_f)

    # initialize cmd_dict
    cmd_dict = {}

    # input and output information
    cmd_dict['input_cmd'] = INPUT_DIR
    cmd_dict['output_cmd'] = OUTPUT_DIR

    # clamp information
    cmd_dict['clamp_bin'] = "ClampCmd_1.3.1/bin/clamp-nlp-1.3.1-jar-with-dependencies.jar"
    cmd_dict['pipeline'] = "ClampCmd_1.3.1/pipeline/clamp-ner-attribute.pipeline.jar"
    cmd_dict['umls_index'] = "ClampCmd_1.3.1/resource/umls_index/"

    # umls information
    cmd_dict['umls_name'] = config["UMLS_USER_NAME"]
    cmd_dict['umls_pass'] = config["UMLS_USER_PASS"]

    cmd = 'java -DCLAMPLicenceFile="ClampCmd_1.3.1/CLAMP.LICENSE" -Xmx2g -cp {clamp_bin} edu.uth.clamp.nlp.main.PipelineMain'.format(**cmd_dict)
    cmd = cmd + " \
        -i {input_cmd}\
        -o {output_cmd}\
        -p {pipeline}\
        -U {umls_name}\
        -P {umls_pass}\
        -I {umls_index}\
    ".format(**cmd_dict)

    # change directory and run script
    os.system(cmd)

if __name__ == '__main__':
    run_clamp()

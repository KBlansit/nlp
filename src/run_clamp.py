#!/usr/bin/env python
import os
import yaml

CONFIG_PATH = "config_parameters.yaml"

CLAMP_DIR = "ClampCmd_1.3.1"

def run_clamp(output_dir, input_dir):
    """
    runs clamp
    NOTE: Remember that mapping using partial uses last arguement
    """
    # console message
    print("Running Clamp")

    # read in umls config data
    with open(CONFIG_PATH, "r") as config_f:
        config = yaml.load(config_f)

    # initialize cmd_dict
    cmd_dict = {}

    # input and output information
    cmd_dict['input_cmd'] = input_dir
    cmd_dict['output_cmd'] = output_dir

    # clamp information
    cmd_dict['clamp_bin'] = os.path.join(CLAMP_DIR, "bin/clamp-nlp-1.3.1-jar-with-dependencies.jar")
    cmd_dict['pipeline'] = os.path.join(CLAMP_DIR, "pipeline/clamp-ner-attribute.pipeline.jar")
    cmd_dict['umls_index'] = os.path.join(CLAMP_DIR, "resource/umls_index/")
    cmd_dict['license'] = os.path.join(CLAMP_DIR, "CLAMP.LICENSE")

    # umls information
    cmd_dict['umls_name'] = config["UMLS_USER_NAME"]
    cmd_dict['umls_pass'] = config["UMLS_USER_PASS"]

    cmd = 'java -DCLAMPLicenceFile={license} -Xmx2g -cp {clamp_bin} edu.uth.clamp.nlp.main.PipelineMain'.format(**cmd_dict)
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

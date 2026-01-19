import numpy as np
from math import cos, sin, pi, sqrt, atan2, asin, log10, acos, copysign
from typing import Union
import robot_visualization 
import argparse
import json
import time
import dataset_generation
import hayati_model
from dataset_generation import FIELDNAMES_OPTIONS, read_dataset

def main(args):
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    model = hayati_model.HayatiModel(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Name of .json configuration file. Default: ARM95.json", default="ARM95.json")
    args = parser.parse_args()
    main(args)
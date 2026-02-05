import numpy as np
from math import cos, sin, pi, sqrt, atan2, asin, log10, acos, copysign
from typing import Union
import robot_visualization 
import argparse
import json
import time
import hayati_model
from random import uniform

DEG = np.pi / 180

def main(args):
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    model = hayati_model.HayatiModel(config)
    create_real_DH(model)

def create_real_DH(model: hayati_model.HayatiModel, angle_delta=1.5*DEG, len_delta=0.2/1000, tool_offset = True):
    for i in range(6):
        if(model.nominal_dh[i][-1] == 0):
            d_or_beta = len_delta
        else:
            d_or_beta = angle_delta
        print("  [")
        print(f"    {model.nominal_dh[i][0] + len_delta * uniform(-1, 1):0.6f}, {model.nominal_dh[i][1] + angle_delta * uniform(-1, 1):0.6f}, {model.nominal_dh[i][2] + d_or_beta * uniform(-1, 1):0.6f}, {model.nominal_dh[i][3] + angle_delta * uniform(-1, 1):0.6f}, {model.nominal_dh[i][-1]}")
        print("  ],")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Name of .json configuration file. Default: ARM95.json", default="ARM95.json")
    args = parser.parse_args()
    main(args)
import numpy as np
import matplotlib.pyplot as plt 
from math import cos, sin, pi, sqrt, atan2, asin, log10, acos, copysign
from typing import Union
from math_routines import x_rot, y_rot, z_rot, arbitrary_axis_rot, trans
from robotic_transformations import dh_trans, hayati_trans
import robot_visualization 
import pygame
import joystick 
import argparse
import json
import time
from ctypes import c_float
import multiprocessing as mp
import dataset_generation
import hayati_model

def main(args):
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    model = hayati_model.HayatiModel(config)
    robot_visualization.vizualize(model, "nominal")
    if args.generate:
        dataset_generation.generate_dataset(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Name of .json configuration file. Default: ARM95.json", default="ARM95.json")
    parser.add_argument("-g", "--generate", help="Generate dataset for selected method. Default: false", type=bool, default=False)
    args = parser.parse_args()
    main(args)
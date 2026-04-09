import matplotlib.pyplot as plt
import numpy as np
from math import acos
from hayati_model import HayatiModel
import argparse
import json
from math_routines import DEG

def wire_limits(model, angles):    
    # z limits
    matrix = model.get_transition_matrix(angles, "nominal")
    z_dir = matrix[0:3, 2]
    scalar_z = np.vdot(z_dir, -model.zero_wire_direction)
    
    #np.clip(scalar_z, -1.0, 1.0)
    #alpha_z = acos(scalar_z)
    #angle_cons_z = np.array([alpha_z, model.angle_limit_z - alpha_z]) # so angle would fit

    wire = matrix[0:3, 3] - model.zero_offset_nominal
    wire_len = np.linalg.norm(wire) 
    scalar_wire = np.vdot(-(wire / wire_len), z_dir)
    np.clip(scalar_wire, -1, 1)
    alpha_z = acos(scalar_wire)
    angle_cons_z = np.array([alpha_z, model.angle_limit_z - alpha_z]) # so angle would fit
    
    # wire limits        
    len_cons = np.array([wire_len - model.wire_limits[0], model.wire_limits[1] - wire_len]) # so wire len would fit

    scalar_wire_ang = np.vdot(wire / wire_len, model.zero_wire_direction)  # > 0 so wire stretch forward        
    alpha = acos(scalar_wire_ang)
    print(alpha / DEG)
    angle_cons_wire = np.array([alpha, model.angle_limit_wire - alpha]) # so angle would fit

    return np.concatenate(([scalar_z], [scalar_wire], angle_cons_z,  [scalar_wire_ang], angle_cons_wire, len_cons)) #angle_cons_wire,



parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="Name of .json configuration file. Default: src/config/ARM95_calibration.json", default="src/config/ARM95_calibration.json")
args = parser.parse_args()

with open(args.config, 'r') as config_file:
    config = json.load(config_file)
model = HayatiModel(config)


angles = np.array([15.127,	23.399,	136.408,	20.193,	-48.524,	-102.140]) * DEG
print(wire_limits(model, angles))



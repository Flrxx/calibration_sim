import numpy as np
import matplotlib.pyplot as plt 
from math import cos, sin, pi, sqrt, atan2, asin, log10, acos, copysign
from typing import Union
from math_routines import x_rot, y_rot, z_rot, arbitrary_axis_rot, trans
from robotic_transformations import dh_trans, hayati_trans
import multiprocessing as mp
import copy
from math_routines import DEG
from scipy.optimize import minimize
import argparse
import json
import hayati_model

def is_pose_valid(model, angles):
    # НОВОЕ: Сначала проверяем динамические лимиты (это быстро)
    # if model.violates_dynamic_limits(angles):
    #     return False
        
    matrix = model.get_transition_matrix(angles, "nominal")
    z_dir = matrix[0:3, 2]
    wire = matrix[0:3, 3] - model.zero_offset_nominal
    wire_len = np.linalg.norm(wire) 
    print(wire_len)
    
    #if wire_len < model.wire_limits[0] or wire_len > model.wire_limits[1]: return False

    wire_norm = wire / wire_len
    scalar_wire_forward = np.vdot(wire_norm, model.zero_wire_direction)
    if scalar_wire_forward <= 0: return False
        
    alpha = acos(np.clip(scalar_wire_forward, -1.0, 1.0))
    if alpha > model.angle_limit_wire: return False
    print(alpha / DEG)

    scalar_z = np.vdot(-wire_norm, z_dir)
    alpha_z = acos(np.clip(scalar_z, -1.0, 1.0))
    
    print(alpha_z / DEG)
    if alpha_z > model.angle_limit_z: return False
        
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/config/ARM95_calibration.json", help="Путь к конфигу модели")
    args = parser.parse_args()

    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    model = hayati_model.HayatiModel(config)

    angle = np.array([28.131,20.228,137.658,-17.419,-0.000,-17.591]) * DEG
    
    print(is_pose_valid(model, angle))

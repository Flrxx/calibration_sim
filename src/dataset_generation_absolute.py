import csv
import argparse
import json
import numpy as np
from random import uniform, random
from math_routines import extract_zyx_euler
import hayati_model
from typing import Union
from csv_routines import write_dataset, add_line_csv, read_dataset
import copy
from math_routines import DEG

FIELDNAMES_OPTIONS = {
    "random" : ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'px_r', 'py_r', 'pz_r', 'rx_r', 'ry_r', 'rz_r'],
    "circles": ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'px_r', 'py_r', 'pz_r', 'joint'],
    "random_poses" : ['q1', 'q2', 'q3', 'q4', 'q5', 'q6']
}

def generate_real_dh(model:hayati_model.HayatiModel):
    new_dh = copy.deepcopy(model.nominal_dh)
    for i in range(6):
        if(model.nominal_dh[i][-1] == 0):
            d_or_beta = model.distance_delta
        else:
            d_or_beta = model.angle_delta
        new_dh[i][0] += model.distance_delta * uniform(-1, 1) 
        new_dh[i][1] += model.angle_delta * uniform(-1, 1)
        new_dh[i][2] += d_or_beta * uniform(-1, 1) 
        new_dh[i][3] += model.angle_delta * uniform(-1, 1)

        # print("[")
        # print(f"    {new_dh[i][0]:0.6f}, {new_dh[i][1]:0.6f}, {new_dh[i][2]:0.6f}, {new_dh[i][3]:0.6f}, {model.nominal_dh[i][-1]}")
        # print("],")
    return new_dh

def generate_dataset(model: hayati_model.HayatiModel):
    print("Generating main dataset...")
    write_dataset(generate_random_dataset(model), model.test_dataset_file, FIELDNAMES_OPTIONS["random_poses"])
    # print("Generating circles datasets...")
    # write_dataset(generate_base_circles_dataset(model), model.base_circles_dataset_file, FIELDNAMES_OPTIONS["circles"])
    # write_dataset(generate_tool_circles_dataset(model), model.tool_circles_dataset_file, FIELDNAMES_OPTIONS["circles"])
    print('Done')

# def optimal_random_dataset(model, samples_num):
#     backup_base = model.nominal_base_params.copy()
#     model.nominal_base_params = np.zeros(6)

#     # best_value = BIG_NUMBER
#     # prev_best_value = BIG_NUMBER * 10
#     dataset = generate_random_dataset(samples)

#     # while prev_best_value - best_value > 0.01:
#     #     dataset, val = self.conf_plus(dataset)
#     #     print(val)
#     #     dataset, val = self.conf_minus(dataset)
#     #     print(val)
#     #     prev_best_value = best_value
#     #     best_value = val

#     model.nominal_base_params = backup_base
#     return dataset

def generate_random_dataset(model, disable_limits=False):
    dataset = np.zeros((model.test_samples_number, len(FIELDNAMES_OPTIONS["random_poses"])))
    for i in range(model.test_samples_number):
        dataset[i] = make_random_sample(model, disable_limits)
    return dataset

def make_random_sample(model, disable_limits):
    while True:
        angle_set = np.array([model.joint_limits_general_l[axis] + random() * (model.joint_limits_general_h[axis] - model.joint_limits_general_l[axis]) for axis in range(6)], dtype='float')
        nominal_pose = model.get_transition_matrix(angle_set, 'nominal')
        if model.satisfies_cartesian_limits(nominal_pose) or disable_limits:
            # real_pose = model.get_transition_matrix(angle_set, 'real')
            # real_position = real_pose[:3, 3]
            # real_orientation = extract_zyx_euler(real_pose[:3, :3])
            # return np.concatenate((angle_set, real_position, real_orientation), axis=0)
            return angle_set
        
def generate_base_circles_dataset(model):
    return np.concatenate((make_circle(model, 1, [1.57, 0, 1.57, 0, 1.57, 0]),
                            make_circle(model, 2, [1.57, 0, 1.57, 0, 1.57, 0])), axis=0)

def generate_tool_circles_dataset(model):
    return np.concatenate((make_circle(model, 5, [0, 0, 1.57, -1.57, 0, 0]),
                            make_circle(model, 6, [0, 0, 1.57, -1.57, 1.57, 0])), axis=0)

def make_circle(model, axis, initial_position):
    dataset = np.array([], dtype='float').reshape(0, len(FIELDNAMES_OPTIONS["circles"]))
    inc = (model.joint_limits_circle_h[axis - 1] - model.joint_limits_circle_l[axis - 1])/model.circle_samples_number
    angles = initial_position
    angles[axis - 1] = model.joint_limits_circle_l[axis - 1]
    for _ in range(model.circle_samples_number):
        real_position = model.get_transition_matrix(angles, 'real')[:3, 3]
        dataset = np.concatenate((dataset, np.concatenate((angles, real_position, [axis]), axis=0).reshape(1, -1)), axis=0)
        angles[axis - 1] += inc
    return dataset

def main(args):
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    model = hayati_model.HayatiModel(config)

    if args.generate:
        generate_dataset(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Name of .json configuration file. Default: ARM95.json", default="ARM95.json")
    parser.add_argument("-g", "--generate", help="Generate poses. Default: true", type=bool, default=True)
    args = parser.parse_args()
    main(args)
        

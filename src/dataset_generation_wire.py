import csv
import argparse
import json
import os
import numpy as np
from math import sqrt
from random import uniform
from math_routines import extract_zyx_euler
import hayati_model
from typing import Union
from scipy.optimize import minimize
from csv_routines import write_dataset, add_line_csv, read_dataset

np.set_printoptions(precision=3, suppress=True, formatter={'all': lambda x: f'{x:0.3f}'})

ERRORS_LIST = ['theta_6', 'd_6', 'theta_5', 'theta_4', 'a_3', 'd_4', 'a_2', 'theta_3', 'theta_2', 'a_1']

FIELDNAMES_OPTIONS = {
    "wire_contributions" : ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'index', *ERRORS_LIST],
    "wire_samples": ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'index', 'd'],
    "random_poses" : ['q1', 'q2', 'q3', 'q4', 'q5', 'q6'],
    "test_result": ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'd_nom', 'd_est', 'delta']
}

DEG = np.pi / 180

def calculate_jac_DH_params(model: hayati_model.HayatiModel, angles: Union[list, np.ndarray], eps = 1e-6):
    jac_DH = np.zeros((3, len(ERRORS_LIST)))
    initial_coords = model.get_transition_matrix(angles, "nominal")[0:3, 3]
    for i, e in enumerate(ERRORS_LIST):        
        model.change_nominal_dh(e, eps)                                     #change DH
        new_coords = model.get_transition_matrix(angles, "nominal")[0:3, 3]
        for j in range(3):
            jac_DH[j, i] = (new_coords[j] - initial_coords[j]) / eps
        model.change_nominal_dh(e, -eps)                                    #reset DH
        pass
    return jac_DH

def calculate_wire_direction(model: hayati_model.HayatiModel, angles: Union[list, np.ndarray]):
    #zero_offset = model.get_transition_matrix(model.zero_wire_angles, "nominal")[0:3, 3]
    point_offset = model.get_transition_matrix(angles, "nominal")[0:3, 3]
    wire_vector = (np.array([point_offset[0] - model.zero_offset_nominal[0],
                                point_offset[1] - model.zero_offset_nominal[1],
                                point_offset[2] - model.zero_offset_nominal[2]])).reshape(3, 1)
    wire_norm = np.linalg.norm(wire_vector)
    if wire_norm == 0:
        print('Wire is not stretched')
        return wire_vector * 0
        #raise ValueError('Wire is not stretched')
    return wire_vector / wire_norm

def calculate_contribution(model: hayati_model.HayatiModel, angles: Union[list, np.ndarray]):
    jac_DH = calculate_jac_DH_params(model, angles)
    #jac_DH_0 = calculate_jac_DH_params(model, model.zero_wire_angles)
    wire_direction = calculate_wire_direction(model, angles)
    contribution = np.dot(wire_direction.T, jac_DH - model.jac_DH_0)
    return contribution.flatten()

def calculate_optimal_position(model: hayati_model.HayatiModel, i_target: int, angles_start:Union[list, np.ndarray], bounds:Union[list, np.ndarray],
                                            prev_eps: float, future_eps: float):
    
    def objective(angles):
        output = calculate_contribution(model, angles)
        return -(output[i_target]**2)

    cons = []
    if i_target > 0:
        def past_limits(angles):
            past_values = calculate_contribution(model, angles)[:i_target]
            upper = prev_eps - past_values
            lower = past_values + prev_eps
            return np.concatenate([upper, lower])        
        cons.append({'type': 'ineq', 'fun': past_limits})

    if i_target < len(ERRORS_LIST) - 1:
        def future_limits(angles):
            past_values = calculate_contribution(model, angles)[i_target + 1:]
            upper = future_eps - past_values
            lower = past_values + future_eps
            return np.concatenate([upper, lower])
    cons.append({'type': 'ineq', 'fun': future_limits})

    def z_direction_limit(angles):    # so z_new face forward
        z_dir = model.get_transition_matrix(angles, "nominal")[0:3, 2]
        dot = np.vdot(z_dir, -model.zero_wire_direction)
        return dot #z_dir[0]    
    cons.append({'type': 'ineq', 'fun': z_direction_limit})
    
    def wire_direction_limit(angles):    # so wire stretch forward
        wire_dir = model.get_transition_matrix(angles, "nominal")[0:3, 3] - model.zero_offset_nominal 
        res = np.vdot(wire_dir, model.zero_wire_direction)
        return res
    cons.append({'type': 'ineq', 'fun': wire_direction_limit})

    res = minimize(objective, angles_start, method='SLSQP', 
        bounds=bounds, constraints=cons, tol=1e-6, options={'ftol': 1e-6})
    if res.success:
        return(res.x)
    else:
        return([None for _ in range(6)])
    
def generate_single_layer(model: hayati_model.HayatiModel, i_target: int, user_joint_mask, tries_num, prev_eps, future_eps, angle_diff):
    samples = []
    samples_contributions = []
    angles = model.zero_wire_angles.copy()
    bounds = np.array([[angles[i], angles[i]] for i in range(6)])
    for _ in range(tries_num):
        for i, joint in enumerate(user_joint_mask):
            if(joint):
                angles[i] = uniform(model.bounds[i][0], model.bounds[i][1])
                bounds[i] = model.bounds[i]

        new_sample = calculate_optimal_position(model, i_target, angles, bounds, prev_eps, future_eps)
        if(new_sample[0] != None):      # optimiztion success
            # m = model.get_transition_matrix(new_sample, "nominal")
            # z_dir = m[0:3, 2] 
            # res1 = np.vdot(z_dir, -model.zero_wire_direction)
            # if (res1 < 0):          # if z_new faced backwards
            #     continue        
            # wire_dir = m[0:3, 3] - model.zero_offset_nominal  
            # res =   np.vdot(wire_dir, model.zero_wire_direction)
            # if(res <= 0):    #if wire is stretched backwards
            #     continue

            single_contribution = calculate_contribution(model, new_sample)
            # if abs(single_contribution[i_target]) < future_eps:   # low contribution
            #     continue

            similar_index = get_similar_index(samples, new_sample, angle_diff*DEG)
            if similar_index == None:
                samples.append(new_sample.copy())
                samples_contributions.append(single_contribution.copy())
            elif abs(single_contribution[i_target]) > abs(samples_contributions[similar_index][i_target]):
                samples[similar_index] = new_sample.copy()
                samples_contributions[similar_index] = single_contribution.copy()

    return {"samples": samples, "contributions": samples_contributions}

def get_similar_index(samples, new_sample: np.ndarray, delta):     
    if len(samples) == 0:
        return None
    
    samples_array = np.array(samples)
    diffs = np.abs(samples_array - new_sample)
    
    # This creates a boolean array where True means the row is similar
    is_similar_mask = np.all(diffs <= delta, axis=1)
    if not np.any(is_similar_mask):
        return None
    
    return np.argmax(is_similar_mask)

def generate_samples(model: hayati_model.HayatiModel, i_target: int, user_joint_mask, max_tries=100, prev_eps=0.0005, future_eps=0.0004, angle_diff=5):
    res = generate_single_layer(model, i_target, user_joint_mask, max_tries, prev_eps, future_eps, angle_diff)
    printable_res = [np.concatenate((s / DEG, [i_target], c.flatten())).tolist() for s, c in zip(res["samples"], res["contributions"])]
    printable_res.sort(key = lambda x: -abs(x[7 + i_target]))
    write_dataset(printable_res, "datasets/ARM95/wire/generation_output.csv", FIELDNAMES_OPTIONS["wire_contributions"], tolerance = ".3f")
    print(f"Found {len(printable_res)} poses")

def measure_real_distance(model: hayati_model.HayatiModel, angles: Union[list, np.ndarray], abs_tolerance, resolution):
    [x, y, z] = model.get_transition_matrix(angles, "real")[0:3, 3]
    distance = sqrt((x - model.zero_offset_real[0])**2 + (y - model.zero_offset_real[1])**2 + (z - model.zero_offset_real[2])**2)
    distance *= 1 + uniform(-1, 1) * abs_tolerance          # fit tolerance
    distance = round(distance / resolution) * resolution    # fit resolution
    return distance

def measure_all_distances(model: hayati_model.HayatiModel, dataset_contributions: str, abs_tolerance: float, resolution: float):
    dataset = read_dataset(dataset_contributions, FIELDNAMES_OPTIONS['wire_contributions'])[:, 0:8]
    dataset[:, 0:6] *= DEG
    for sample in dataset:
        sample[7] = measure_real_distance(model, sample[0:6], abs_tolerance, resolution)
    return dataset

def make_calibration_dataset(model: hayati_model.HayatiModel, dataset_contributions: str, calibration_dataset="datasets/ARM95/wire/calibration_dataset.csv", abs_tolerance=0.08/100, resolution=0.05/1000):
    model.jac_DH_0 = calculate_jac_DH_params(model, model.zero_wire_angles)
    dataset = measure_all_distances(model, dataset_contributions, abs_tolerance, resolution)
    dataset[:, 0:6] /= DEG
    write_dataset(dataset, calibration_dataset, FIELDNAMES_OPTIONS["wire_samples"], tolerance=".6f") 

def main(args):
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    model = hayati_model.HayatiModel(config)
    model.jac_DH_0 = calculate_jac_DH_params(model, model.zero_wire_angles)

    mode = "find"
    #mode = "check"
    #mode = "distances"

    if mode == "find":
        print("generation_output.csv will be erased, continue? y/n")
        #ans = input()
        #if ans == 'y':
        user_joint_mask = np.array([0, 0, 0, 1, 1, 1])
        i_target = 0
        max_tries = 100
        prev_eps = 0.0005
        future_eps = 0.0004
        angle_diff = 5
        generate_samples(model, i_target, user_joint_mask, max_tries=max_tries, prev_eps=prev_eps, future_eps=future_eps, angle_diff=angle_diff) 

    elif mode == "check":
        angles = np.array([0 * DEG, 36.1 * DEG, 96.8 * DEG, -90 * DEG, -41.976 * DEG, -81.238 * DEG])
        contribution = calculate_contribution(model, angles)
        add_line_csv(np.concatenate((angles / DEG, [0], contribution.flatten())).tolist(),"datasets/ARM95/wire/contribution_dataset.csv", FIELDNAMES_OPTIONS['wire_contributions'], ".3f")
        
    elif mode == "distances":
        make_calibration_dataset(model, dataset_contributions="datasets/ARM95/wire/contribution_dataset.csv")

    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Name of .json configuration file. Default: ARM95.json", default="ARM95.json")
    parser.add_argument("-g", "--generate", help="Generate dataset for selected method. Default: true", type=bool, default=True)
    args = parser.parse_args()
    main(args)
        

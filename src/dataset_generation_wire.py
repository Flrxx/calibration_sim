import csv
import argparse
import json
import numpy as np
from math import sqrt
from random import uniform
from math_routines import extract_zyx_euler
import hayati_model
from typing import Union
from scipy.optimize import minimize

np.set_printoptions(precision=3, suppress=True, formatter={'all': lambda x: f'{x:0.3f}'})

ERRORS_LIST = ['theta_6', 'd_6', 'theta_5', 'theta_4', 'a_3', 'd_4', 'a_2', 'theta_3', 'theta_2', 'a_1']

FIELDNAMES_OPTIONS = {
    "wire" : ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', *ERRORS_LIST],
    "circles": ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'px_r', 'py_r', 'pz_r', 'joint'],
}

DEG = np.pi / 180

def write_dataset(dataset: np.ndarray, filename: str, fieldnames: list[str], tolerance=".8f"):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in dataset:
            writer.writerow({name: f"{row[index]:{tolerance}}" for index, name in enumerate(fieldnames)})

def measure_distance(model: hayati_model.HayatiModel, angles: Union[list, np.ndarray], abs_tolerance = 0.0005):
    [x, y, z] = model.get_transition_matrix(angles, "real")[0:3, 3]
    distance = sqrt((x - model.zero_offset_real[0])^2 + (y - model.zero_offset_real[1])^2 + (z - model.zero_offset_real[2])^2)
    distance *= 1 + uniform(-1, 1) * abs_tolerance
    return distance

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
    zero_offset = model.get_transition_matrix(model.zero_wire_angles, "nominal")[0:3, 3]
    point_offset = model.get_transition_matrix(angles, "nominal")[0:3, 3]
    wire_vector = (np.array([point_offset[0] - zero_offset[0],
                                point_offset[1] - zero_offset[1],
                                point_offset[2] - zero_offset[2]])).reshape(3, 1)
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

    res = minimize(objective, angles_start, method='SLSQP', 
        bounds=bounds, constraints=cons, tol=1e-6, options={'ftol': 1e-6})
    if res.success:
        return(res.x)
    else:
        return([None for _ in range(6)])
    
def generate_single_layer(model: hayati_model.HayatiModel, i_target: int, user_joint_mask, tries_num, prev_eps, future_eps, angle_diff=5):
    error_letter, error_num = ERRORS_LIST[i_target].split("_")
    error_num = int(error_num) - 1
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
        if(new_sample[0] != None):
            if has_difference(samples, new_sample, angle_diff*DEG):
                samples.append(new_sample)
                single_contribution = calculate_contribution(model, new_sample)
                samples_contributions.append(single_contribution)

    return {"samples": samples, "contributions": samples_contributions}

def has_difference(samples, new_sample: np.ndarray, delta):
    if len(samples) == 0:
        return True    
    samples_array = np.array(samples)
    diffs = np.abs(samples_array - new_sample)
    is_similar = np.any(np.all(diffs <= delta, axis=1))
    return not is_similar

def generate_dataset(model: hayati_model.HayatiModel):
    print("Generating wire dataset...")
    write_dataset(generate_optimal_dataset(model), model.dataset_file, FIELDNAMES_OPTIONS["wire"])
    print('Done')

def generate_optimal_dataset(model: hayati_model.HayatiModel, disable_limits=False):
    dataset = np.zeros((model.general_samples_number, len(FIELDNAMES_OPTIONS["wire"])))
    for i in range(ERRORS_LIST):
        
        for j in range(6):                      # Single dataset layer
            generate_single_layer()
            calculate_optimal_position(model, )

    for i in range(model.general_samples_number):
        dataset[i] = make_optimal_sample(model, disable_limits)
    return dataset

def test_contribution(model: hayati_model.HayatiModel, angles: Union[list, np.ndarray]):
    print(calculate_contribution(model, angles))

def test_generation(model: hayati_model.HayatiModel, i_target: int, user_joint_mask, max_tries=100, prev_eps=0.0005, future_eps=0.0005):
    res = generate_single_layer(model, i_target, user_joint_mask, tries_num=max_tries, prev_eps=prev_eps, future_eps=future_eps)
    printable_res = [np.concatenate((s / DEG, c.flatten())).tolist() for s, c in zip(res["samples"], res["contributions"])]
    printable_res.sort(key = lambda x: -abs(x[6 + i_target]))
    write_dataset(printable_res, "output.csv", FIELDNAMES_OPTIONS["wire"], tolerance = ".3f")
    print("Done")
 
def main(args):
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    model = hayati_model.HayatiModel(config)
    model.jac_DH_0 = calculate_jac_DH_params(model, model.zero_wire_angles)

    #angles = np.array([0 * DEG, 36.1 * DEG, 96.8 * DEG, -90 * DEG, 90 * DEG, 0 * DEG])
    #test_contribution(model, angles)

    user_joint_mask = np.array([0, 0, 0, 1, 1, 1])
    i_target = 0
    max_tries = 30
    prev_eps = 0.0005
    #future_eps = 0.0005
    test_generation(model, i_target, user_joint_mask, max_tries=max_tries, prev_eps=prev_eps) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Name of .json configuration file. Default: ARM95.json", default="ARM95.json")
    parser.add_argument("-g", "--generate", help="Generate dataset for selected method. Default: true", type=bool, default=True)
    args = parser.parse_args()
    main(args)
        

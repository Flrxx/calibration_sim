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

FIELDNAMES_OPTIONS = {
    "wire" : ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'd'],
    "circles": ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'px_r', 'py_r', 'pz_r', 'joint'],
}

ERRORS_LIST = ['theta_6', 'd_6', 'theta_5', 'theta_4', 'a_3', 'd_4', 'a_2', 'theta_3', 'theta_2', 'a_1']

DEG = np.pi / 180

def generate_dataset(model: hayati_model.HayatiModel):
    print("Generating wire dataset...")
    write_dataset(generate_optimal_dataset(model), model.dataset_file, FIELDNAMES_OPTIONS["wire"])
    print("Generating circles datasets...")
    print('Done')

def generate_optimal_dataset(model: hayati_model.HayatiModel, disable_limits=False):
    dataset = np.zeros((model.general_samples_number, len(FIELDNAMES_OPTIONS["wire"])))
    zero_offset = model.get_transition_matrix(model.zero_wire_angles, "real")[0:3, 3]
    for i in range(model.general_samples_number):
        dataset[i] = make_optimal_sample(model, zero_offset, disable_limits)
    return dataset

def measure_distance(model: hayati_model.HayatiModel, zero_offset: Union[list, np.ndarray], angles: Union[list, np.ndarray], abs_tolerance = 0.0005):
    [x, y, z] = model.get_transition_matrix(angles, "real")[0:3, 3]
    distance = sqrt((x - zero_offset[0])^2 + (y - zero_offset[1])^2 + (z - zero_offset[2])^2)
    distance *= 1 + uniform(-1, 1) * abs_tolerance
    return distance

def calculate_jac_DH_params(model: hayati_model.HayatiModel, angles: Union[list, np.ndarray], eps = 1e-6):
    jac_DH = np.zeros((3, len(ERRORS_LIST)))
    initial_coords = model.get_transition_matrix(angles, "nominal")[0:3, 3]
    for i, e in enumerate(ERRORS_LIST):        
        model.change_nominal_dh(e, eps)                                         #change DH
        new_coords = model.get_transition_matrix(angles, "nominal")[0:3, 3]
        for j in range(3):
            jac_DH[j, i] = (new_coords[j] - initial_coords[j]) / eps
        model.change_nominal_dh(e, -eps)                                        #reset DH
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
    jac_DH_0 = calculate_jac_DH_params(model, model.zero_wire_angles)
    wire_direction = calculate_wire_direction(model, angles)
    contribution = np.dot(wire_direction.T, jac_DH - jac_DH_0)
    return contribution

def test(model: hayati_model.HayatiModel):
    angles_start = np.array([0 * DEG, 0 * DEG, 90 * DEG, 0 * DEG, 90 * DEG, 0 * DEG])

    return calculate_optimal_position(model, 3)

def calculate_optimal_position(model: hayati_model.HayatiModel, target_index: int, penalty_weight_left=100.0,  penalty_weight_right=1000.0):
    angles_start = np.array([0 * DEG, 0 * DEG, 90 * DEG, 0 * DEG, 90 * DEG, 0 * DEG])
    bounds = [(model.joint_limits_general_l[i], model.joint_limits_general_h[i]) for i in range(6)]
    def cost(angles):
        psi = calculate_contribution(model, angles)
        target_val = psi[0, target_index]
        left_vals = psi[0, :target_index]
        right_vals = psi[0, target_index + 1:]

        cost = -target_val + penalty_weight_left * np.sum(left_vals**2) + penalty_weight_right * np.sum(right_vals**2)
        return cost
    
    res = minimize(cost, angles_start, method='L-BFGS-B', bounds=bounds)
    return res.x

def calculate_optimal_position_2(model: hayati_model.HayatiModel, target_index: int):
    angles_start = np.array([0 * DEG, 0 * DEG, 90 * DEG, 0 * DEG, 90 * DEG, 0 * DEG])
    bounds = [(model.joint_limits_general_l[i], model.joint_limits_general_h[i]) for i in range(6)]
    def cost(angles):
        psi = calculate_contribution(model, angles)
        return -psi[0, target_index]
    def constraint_eq(angles):
        psi = calculate_contribution(model, angles)
        # Возвращаем все элементы, КРОМЕ целевого
        return np.delete(psi, target_index)
    
    cons = ({'type': 'eq', 'fun': constraint_eq})
    res = minimize(cost, angles_start, method='SLSQP', constraints=cons, bounds=bounds)
    return res.x

def main(args):
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    model = hayati_model.HayatiModel(config)
    angles = np.array([0 * DEG, 36.1 * DEG, 96.8 * DEG, -90.0 * DEG, 90 * DEG, 0 * DEG])
    print(ERRORS_LIST)
    print(calculate_contribution(model, angles))
    #res = test(model)
    # print(res/DEG)
    # print(calculate_contribution(model, res))
    # if args.generate:
    #     generate_dataset(model)
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Name of .json configuration file. Default: ARM95.json", default="ARM95.json")
    parser.add_argument("-g", "--generate", help="Generate dataset for selected method. Default: true", type=bool, default=True)
    args = parser.parse_args()
    main(args)
        

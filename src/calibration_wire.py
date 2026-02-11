import numpy as np
from math import cos, sin, pi, sqrt, atan2, asin, log10, acos, copysign
from typing import Union
from scipy.optimize import minimize_scalar, minimize
import argparse
import json
import time
import hayati_model
from random import uniform
from dataset_generation_absolute import generate_real_DH
from dataset_generation_wire import make_calibration_dataset, FIELDNAMES_OPTIONS, ERRORS_LIST
from csv_routines import read_dataset, write_dataset

DEG = np.pi / 180

def print_real_DH(model: hayati_model.HayatiModel, angle_delta=1*DEG, len_delta=0.5/1000, tool_offset = False):
    for i in range(6):
        if(model.nominal_dh[i][-1] == 0):
            d_or_beta = len_delta
        else:
            d_or_beta = angle_delta
        print("  [")
        print(f"    {model.nominal_dh[i][0] + len_delta * uniform(-1, 1):0.6f}, {model.nominal_dh[i][1] + angle_delta * uniform(-1, 1):0.6f}, {model.nominal_dh[i][2] + d_or_beta * uniform(-1, 1):0.6f}, {model.nominal_dh[i][3] + angle_delta * uniform(-1, 1):0.6f}, {model.nominal_dh[i][-1]}")
        print("  ],")

def execute_calibration(model: hayati_model.HayatiModel, calibration_dataset: str):
    dataset = read_dataset(calibration_dataset, FIELDNAMES_OPTIONS["wire_samples"]) 
    for i in range (len(ERRORS_LIST)):
        single_param_dataset = dataset[dataset[:, 6] == i]
        if(len(single_param_dataset) == 0):
            continue

        optimizing_param = calibrate_single_param(model, single_param_dataset)        
        
        param_num, param_letter_num = model.params_from_letters(ERRORS_LIST[i])
        nominal_value = model.nominal_dh[param_num][param_letter_num]
        real_value = model.real_dh[param_num][param_letter_num]
        final_value = optimizing_param.x[0]
        error = final_value - real_value
        start_error = nominal_value - real_value

        print(f"Nominal value: {nominal_value:.6f}")      
        print(f"Found value: {final_value:.6f}")
        print(f"Real value: {real_value:.6f}")

        if (real_value != 0):         
            start_error_rel = start_error/real_value 
            end_error_rel = error/real_value 
        else:
            start_error_rel = 1.0
            end_error_rel = 1.0

        print(f"Start error: {start_error:.6f}; {start_error_rel* 100:.2f}%")
        print(f"End error: {error:.6f}; {end_error_rel * 100:.4}%")

        print("------------------------------------")

        model.estimated_dh[param_num][param_letter_num] = final_value        

def calibrate_single_param(model: hayati_model.HayatiModel, dataset: Union[np.ndarray, list], delta=10/1000):
    angles_array = dataset[:,0:6] * DEG
    param_index = int(dataset[0,6])
    param_num, param_letter_num = model.params_from_letters(ERRORS_LIST[param_index])
    real_distances = dataset[:,7]

    def loss_function(optimizing_param):
        model.estimated_dh[param_num][param_letter_num] = optimizing_param[0]

        zero_pos = model.get_transition_matrix(model.zero_wire_angles, "estimated")[0:3, 3]
        d_theor_list = np.zeros((len(angles_array)))
        for i, angles in enumerate(angles_array):
            d_theor_list[i] = calculate_estimated_distance(model, angles, zero_pos)
        
        diff = d_theor_list - real_distances
        error = np.sum(diff**2)
        #print(f"Суммарная ошибка квадратов: {error:}")
        
        return error

    print(f"Optimizing param {ERRORS_LIST[param_index]}...")

    res = minimize(loss_function, x0=model.nominal_dh[param_num][param_letter_num], 
                   method='Nelder-Mead', tol=1e-10)
    # res = minimize_scalar(
    #     loss_function, 
    #     bounds=(model.nominal_dh[param_num][param_letter_num] - delta, model.nominal_dh[param_num][param_letter_num] + delta),  # Границы поиска параметра
    #     method='bounded'  #tol=1e-8, options={'ftol': 1e-8}
    # )
    return res

def calculate_estimated_distance(model: hayati_model.HayatiModel, angles: Union[np.ndarray, list], zero_pos: Union[np.ndarray, list]):
    [x, y, z] = model.get_transition_matrix(angles, "estimated")[0:3, 3]
    #[x_0, y_0, z_0] = model.get_transition_matrix(model.zero_wire_angles, "estimated")[0:3, 3]
    distance = sqrt((x - zero_pos[0])**2 + (y - zero_pos[1])**2 + (z - zero_pos[2])**2)
    return distance

def check_results(model: hayati_model.HayatiModel, results="results/ARM95/test_result.csv", random_dataset="datasets/ARM95/absolute/random_samples.csv"):
    poses = read_dataset(random_dataset, FIELDNAMES_OPTIONS["random_poses"])
    
    nominal_data = np.array([model.get_transition_matrix(pose, "nominal")[0:3, 3] for pose in poses])
    real_data = np.array([model.get_transition_matrix(pose, "real")[0:3, 3] for pose in poses])
    estimated_data = np.array([model.get_transition_matrix(pose, "estimated")[0:3, 3] for pose in poses])

    nominal_diff = (real_data - nominal_data) * 1000 # mm
    estimated_diff = (real_data - estimated_data) * 1000 # mm
    nominal_dist = np.linalg.norm(nominal_diff, axis=1).reshape(len(poses), 1)
    estimated_dist = np.linalg.norm(estimated_diff, axis=1).reshape(len(poses), 1)
    delta = nominal_dist - estimated_dist

    printable_res = np.concatenate((poses/DEG, nominal_dist, estimated_dist, delta), axis=1)
    write_dataset(printable_res, results, FIELDNAMES_OPTIONS["test_result"], tolerance=".3f")

def main(args):
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    model = hayati_model.HayatiModel(config)
    #create_real_DH(model, angle_delta=0.5*DEG, len_delta=0.5/1000)

    if args.generate == "True":
        new_dh = generate_real_DH(model, angle_delta=0.5*DEG, len_delta=0.5/1000)
        model.real_dh = new_dh.copy()
        model.zero_offset_real = model.get_transition_matrix(model.zero_wire_angles, "real")[0:3, 3]

    make_calibration_dataset(model, "datasets/ARM95/wire/contribution_dataset.csv", abs_tolerance=0.00/100, resolution=0.00/1000)
    execute_calibration(model, "datasets/ARM95/wire/calibration_dataset.csv")
    check_results(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Name of .json configuration file. Default: ARM95.json", default="ARM95.json")
    parser.add_argument("-g", "--generate", help="Generate new real DH", default="False")
    args = parser.parse_args()
    main(args)
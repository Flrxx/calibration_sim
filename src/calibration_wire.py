import numpy as np
from math import cos, sin, pi, sqrt, atan2, asin, log10, acos, copysign
from typing import Union
from scipy.optimize import minimize_scalar, minimize
import argparse
import json
import time
import hayati_model
from random import uniform
from dataset_generation_wire import FIELDNAMES_OPTIONS, ERRORS_LIST
from csv_routines import read_dataset

DEG = np.pi / 180

def main(args):
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    model = hayati_model.HayatiModel(config)
    #create_real_DH(model)

    execute_calibration(model, "datasets/ARM95/wire/calibration_dataset.csv")

def create_real_DH(model: hayati_model.HayatiModel, angle_delta=2*DEG, len_delta=2/1000, tool_offset = False):
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
        print(f"Start error: {start_error:.6f}; {start_error/real_value * 100:.2f}%")

        print(f"Found value: {final_value:.6f}")
        print(f"Real value: {real_value:.6f}")
        print(f"Error: {error:.6f}; {error/real_value * 100:.4}%")
        print("------------------------------------")

        model.estimated_dh[param_num][param_letter_num] = final_value        

def calibrate_single_param(model: hayati_model.HayatiModel, dataset: Union[np.ndarray, list], delta=0.1):
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
        #print(f"Суммарная ошибка квадратов: {error:.6f}")
        
        return error

    print(f"Optimizing param {ERRORS_LIST[param_index]}...")

    res = minimize(loss_function, x0=model.nominal_dh[param_num][param_letter_num], 
                   method='Nelder-Mead')
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Name of .json configuration file. Default: ARM95.json", default="ARM95.json")
    args = parser.parse_args()
    main(args)
import numpy as np
from math import cos, sin, pi, sqrt, atan2, asin, log10, acos, copysign
from typing import Union
from scipy.optimize import minimize_scalar, minimize, Bounds
import argparse
import json
import time
import hayati_model
from random import uniform
from dataset_generation_absolute import generate_real_dh
from dataset_generation_wire import make_calibration_dataset, measure_all_distances
from csv_routines import read_dataset, write_dataset
from multiprocessing import Pool, cpu_count
import copy
from math_routines import DEG

def execute_calibration(model: hayati_model.HayatiModel, dataset=[]):
    if len(dataset) == 0:
        dataset = read_dataset(model.calibration_dataset, model.fieldnames_options["wire_samples"]) 
        dataset[:,0:6] = dataset[:,0:6] * DEG
    param_errors = np.zeros(len(model.error_list))
    for i in range (len(model.error_list)):
        single_param_dataset = dataset[dataset[:, 6] == i]
        if(len(single_param_dataset) == 0):
            continue

        optimizing_param = calibrate_single_param(model, single_param_dataset)        
        
        param_num, param_letter_num = model.params_from_letters(model.error_list[i])
        nominal_value = model.nominal_dh[param_num][param_letter_num]
        real_value = model.real_dh[param_num][param_letter_num]
        final_value = optimizing_param.x[0]
        final_error = final_value -real_value
        start_error = nominal_value - real_value  

        if (real_value != 0):         
            start_error_rel = start_error/real_value
            final_error_rel = final_error/real_value
            # print(f"start {i}: {start_error_rel}")
            # print(f"end {i}: {final_error_rel}")

        else:
            start_error_rel = 100
            final_error_rel = 100

        # print(f"Nominal value: {nominal_value:.6f}")      
        # print(f"Found value: {final_value:.6f}")
        # print(f"Real value: {real_value:.6f}")

        # print(f"Start final_error: {start_error:.6f}; {start_error_rel* 100:.2f}%")
        # print(f"End final_error: {final_error:.6f}; {end_error_rel * 100:.4}%")

        # print("------------------------------------")

        model.estimated_dh[param_num][param_letter_num] = final_value 
        if start_error_rel != 0:
            param_errors[i] = (abs(start_error_rel) - abs(final_error_rel)) / abs(start_error_rel) * 100 #%
        else:
            param_errors[i] = 100 #%  
    return [param_num, param_letter_num], param_errors 

def calibrate_single_param(model: hayati_model.HayatiModel, dataset: Union[np.ndarray, list]):
    param_index = int(dataset[0,6])
    param_num, param_letter_num = model.params_from_letters(model.error_list[param_index])
    real_distances = dataset[:,7]

    def loss_function(optimizing_param):
        model.estimated_dh[param_num][param_letter_num] = optimizing_param[0]

        zero_pos = model.get_transition_matrix(model.zero_wire_angles, "estimated")[0:3, 3]
        d_theor_list = np.zeros((len(dataset)))
        for i, angles in enumerate(dataset):
            d_theor_list[i] = calculate_estimated_distance(model, angles, zero_pos)
        
        diff = d_theor_list - real_distances
        error = np.sum(diff**2)
        
        return error

    #print(f"Optimizing param {model.error_list[param_index]}...")
    if (param_index in model.angle_idexes):
        bounds = Bounds(model.nominal_dh[param_num][param_letter_num] - 2 * DEG, 
                        model.nominal_dh[param_num][param_letter_num] + 2 * DEG) # [-2 * DEG, 2 * DEG]
    else:
        bounds = Bounds(model.nominal_dh[param_num][param_letter_num] - 2/1000, 
                        model.nominal_dh[param_num][param_letter_num] + 2/1000)

    res = minimize(loss_function, x0=model.nominal_dh[param_num][param_letter_num], 
                   method='Nelder-Mead', tol=1e-10, bounds=bounds)
    return res

def calculate_estimated_distance(model: hayati_model.HayatiModel, angles: Union[np.ndarray, list], zero_pos: Union[np.ndarray, list]):
    [x, y, z] = model.get_transition_matrix(angles, "estimated")[0:3, 3]
    #[x_0, y_0, z_0] = model.get_transition_matrix(model.zero_wire_angles, "estimated")[0:3, 3]
    distance = sqrt((x - zero_pos[0])**2 + (y - zero_pos[1])**2 + (z - zero_pos[2])**2)
    return distance

def check_results(model: hayati_model.HayatiModel, last_indexes: list):
    poses = read_dataset(model.test_dataset_file, model.fieldnames_options["random_poses"])
    
    nominal_data = np.array([model.get_transition_matrix(pose, "nominal")[0:3, 3] for pose in poses])
    real_data = np.array([model.get_transition_matrix(pose, "real")[0:3, 3] for pose in poses])
    estimated_data = np.array([model.get_transition_matrix(pose, "estimated")[0:3, 3] for pose in poses])

    swap_data = model.estimated_dh[last_indexes[0]][last_indexes[1]]
    model.estimated_dh[last_indexes[0]][last_indexes[1]] = model.nominal_dh[last_indexes[0]][last_indexes[1]]
    estimated_data_prev = np.array([model.get_transition_matrix(pose, "estimated")[0:3, 3] for pose in poses])
    model.estimated_dh[last_indexes[0]][last_indexes[1]] = swap_data

    nominal_diff = (real_data - nominal_data) * 1000 # mm
    estimated_diff = (real_data - estimated_data) * 1000 # mm
    estimated_diff_prev = (real_data - estimated_data_prev) * 1000 # mm

    nominal_dist = np.linalg.norm(nominal_diff, axis=1).reshape(len(poses), 1)
    estimated_dist = np.linalg.norm(estimated_diff, axis=1).reshape(len(poses), 1)
    estimated_dist_prev = np.linalg.norm(estimated_diff_prev, axis=1).reshape(len(poses), 1)
    #delta = nominal_dist - estimated_dist

    # nom_params_errors = model.real_dh - model.nominal_dh
    # est_params_errors = model.real_dh - model.estimated_dh

    #params_errors = (nom_params_errors - est_params_errors)

    #printable_res = np.concatenate((poses/DEG, nominal_dist, estimated_dist, delta), axis=1)
    #avg_error = np.sum(delta) / len(delta)
    avg_nom_error = np.sum(nominal_dist)/len(nominal_dist)
    avg_est_error = np.sum(estimated_dist) / len (estimated_dist)
    avg_est_error_prev = np.sum(estimated_dist_prev) / len (estimated_dist_prev)
    #median = np.array([0, 0, 0, 0, 0, 0, avg_nom_error, avg_est_error, avg_error]).reshape(1, 9)
    #printable_res = np.append(printable_res, median, axis=0)
    #write_dataset(printable_res, model.results_file, model.fieldnames_options["test_result"], tolerance=".3f")
    return avg_nom_error, avg_est_error, avg_est_error_prev

def _calibrate_batch(model: hayati_model.HayatiModel, tries_count, generate):
    copy_model = copy.deepcopy(model)
    local_param_errors = np.zeros(len(model.error_list))

    local_nominal_error = 0.0
    local_estimated_error = 0.0
    local_estimated_error_prev = 0.0
    for _ in range(tries_count):
        if generate == "true":
            new_dh = generate_real_dh(copy_model)
            copy_model.change_real_dh(new_dh)
        dataset = measure_all_distances(copy_model)
        last_indexes, param_errors_new = execute_calibration(copy_model, dataset=dataset)
        local_param_errors += param_errors_new
        #param_errors += execute_calibration_group(copy_model, [[0, 6]], dataset=dataset)
        new_nom, new_est, new_esp_prev = check_results(copy_model, last_indexes)
        local_nominal_error += new_nom
        local_estimated_error += new_est
        local_estimated_error_prev += new_esp_prev
        copy_model.reset_estimated_dh()
    
    local_param_errors /= tries_count
    local_nominal_error /= tries_count
    local_estimated_error /= tries_count
    local_estimated_error_prev /= tries_count
    return local_nominal_error, local_estimated_error, local_estimated_error_prev, local_param_errors

def make_many_calibration_attempts(model: hayati_model.HayatiModel, tries_num: int, genarate="false"):
    num_cores = cpu_count()
    chunk_size = max(1, (tries_num // num_cores))

    tasks = []
    for i in range(min(num_cores, tries_num)):
        current_tries = chunk_size
        if current_tries > 0:
            tasks.append((model, chunk_size, genarate))        

    with Pool(processes=num_cores) as pool:
        results = pool.starmap(_calibrate_batch, tasks)

    nominal_error = [x[0] for x in results]
    estimated_error = [x[1] for x in results]
    estimated_error_prev = [x[2] for x in results]
    params_error = np.array([x[3] for x in results])

    nominal_error_median = sum(nominal_error)/len(nominal_error)
    estimated_error_median = sum(estimated_error)/len(estimated_error)
    estimated_error_prev_median = sum(estimated_error_prev)/len(estimated_error_prev)
    params_error_median = np.sum(params_error, axis=0)/len(params_error)

    return nominal_error_median, estimated_error_median, estimated_error_prev_median, params_error_median

def main(args):
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    model = hayati_model.HayatiModel(config)

    nominal_error, estimated_error, estimated_error_prev, params_error = make_many_calibration_attempts(model, args.num_of_tries, genarate=args.generate)  # ,params_error
    print(f"Nominal error: {nominal_error:.6f}\nAfter calibration: {estimated_error:.6f}\n\nDifference: {(nominal_error - estimated_error):.6f}")

    print(f"Previous difference: {(nominal_error - estimated_error_prev):.6f}\n\nContribution of new: {(estimated_error_prev - estimated_error):.6f}")
    print(params_error)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Name of .json configuration file. Default: src/config/ARM95_calibration.json", default="src/config/ARM95_calibration.json")
    parser.add_argument("-g", "--generate", help="Generate new real DH. Default: 'true'", default="true")
    parser.add_argument("-n", "--num_of_tries", help="How many tries to make. Default: 1", type=int, default=10)
    args = parser.parse_args()
    main(args)
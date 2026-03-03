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
        # print(i)
        # print(param_errors[i])   
    return param_errors    

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

def execute_calibration_group(model: hayati_model.HayatiModel, bourders_list: Union[np.ndarray, list], dataset=[]):
    if len(dataset) == 0:
        dataset = read_dataset(model.calibration_dataset, model.fieldnames_options["wire_samples"]) 
        dataset[:,0:6] = dataset[:,0:6] * DEG
    param_errors = np.zeros(len(model.error_list))
    for bourders in bourders_list:
        mask = (dataset[:, 6] >= bourders[0]) & (dataset[:, 6] < bourders[1])
        group_dataset = dataset[mask]
        if(len(group_dataset) == 0):
            continue
        target_indexes = set(group_dataset[:, 6])
        target_params = [model.error_list[int(i)] for i in target_indexes]
        param_num_list = []
        param_letter_list = []
        for target in target_params:
            param_num, param_letter_num = model.params_from_letters(target)
            param_num_list.append(param_num)
            param_letter_list.append(param_letter_num)

        optimizing_params = calibrate_group_of_params(model, group_dataset, param_num_list, param_letter_list) 
        estimated_values_list = optimizing_params.x
        
        nominal_values_list = np.zeros(len(param_letter_list))
        real_values_list = np.zeros(len(param_letter_list))
        for i in range(len(param_num_list)):
            nominal_values_list[i] = model.nominal_dh[param_num_list[i]][param_letter_list[i]]
            real_values_list[i] = model.real_dh[param_num_list[i]][param_letter_list[i]]
            model.estimated_dh[param_num_list[i]][param_letter_list[i]] = estimated_values_list[i] 

        final_error = estimated_values_list - real_values_list
        start_error = nominal_values_list - real_values_list  
        #final_error = np.zeros(len(model.error_list))
        if (np.all(real_values_list != 0)):         
            start_error_rel = start_error/real_values_list
            final_error_rel = final_error/real_values_list
            # print("---")
            # print(start_error_rel)
            # print(final_error_rel)
            # print(f"start {i}: {start_error_rel}")
            # print(f"end {i}: {final_error_rel}")

        else:
            start_error_rel = None
            final_error_rel = None

        # print(f"Nominal value: {nominal_value:.6f}")      
        # print(f"Found value: {final_value:.6f}")
        # print(f"Real value: {real_value:.6f}")

        # print(f"Start final_error: {start_error:.6f}; {start_error_rel* 100:.2f}%")
        # print(f"End final_error: {final_error:.6f}; {end_error_rel * 100:.4}%")

        # print("------------------------------------")

        if (np.all(start_error_rel != 0)):
            for num, index in enumerate(target_indexes):
                param_errors[int(index)] = (abs(start_error_rel[num]) - abs(final_error_rel[num])) / abs(start_error_rel[num]) * 100 #%
        else:
            param_errors[bourders[0]: bourders[1]] = None #%
        # print(i)
        # print(param_errors[i])   
    return param_errors    

def calibrate_group_of_params(model: hayati_model.HayatiModel, dataset: Union[np.ndarray, list], param_num_list, param_letter_list):
    #param_index = int(dataset[0,6])
    real_distances = dataset[:,7]

    def loss_function(optimizing_params):
        for i in range(len(param_num_list)):
            model.estimated_dh[param_num_list[i]][param_letter_list[i]] = optimizing_params[i]

        zero_pos = model.get_transition_matrix(model.zero_wire_angles, "estimated")[0:3, 3]
        d_theor_list = np.zeros((len(dataset)))
        for i, angles in enumerate(dataset):
            d_theor_list[i] = calculate_estimated_distance(model, angles, zero_pos)
        
        diff = d_theor_list - real_distances
        error = np.sum(diff**2)
        
        return error

    #print(f"Optimizing param {model.error_list[param_index]}...")

    x0 = np.zeros(len(param_letter_list))
    for i in range(len(param_num_list)):
            x0[i] = model.nominal_dh[param_num_list[i]][param_letter_list[i]]


    res = minimize(loss_function, x0=x0, 
                   method='Nelder-Mead', tol=1e-10)
    return res

def calculate_estimated_distance(model: hayati_model.HayatiModel, angles: Union[np.ndarray, list], zero_pos: Union[np.ndarray, list]):
    [x, y, z] = model.get_transition_matrix(angles, "estimated")[0:3, 3]
    #[x_0, y_0, z_0] = model.get_transition_matrix(model.zero_wire_angles, "estimated")[0:3, 3]
    distance = sqrt((x - zero_pos[0])**2 + (y - zero_pos[1])**2 + (z - zero_pos[2])**2)
    return distance

def check_results(model: hayati_model.HayatiModel):
    poses = read_dataset(model.test_dataset_file, model.fieldnames_options["random_poses"])
    
    nominal_data = np.array([model.get_transition_matrix(pose, "nominal")[0:3, 3] for pose in poses])
    real_data = np.array([model.get_transition_matrix(pose, "real")[0:3, 3] for pose in poses])
    estimated_data = np.array([model.get_transition_matrix(pose, "estimated")[0:3, 3] for pose in poses])

    nominal_diff = (real_data - nominal_data) * 1000 # mm
    estimated_diff = (real_data - estimated_data) * 1000 # mm
    nominal_dist = np.linalg.norm(nominal_diff, axis=1).reshape(len(poses), 1)
    estimated_dist = np.linalg.norm(estimated_diff, axis=1).reshape(len(poses), 1)
    delta = nominal_dist - estimated_dist

    nom_params_errors = model.real_dh - model.nominal_dh
    est_params_errors = model.real_dh - model.estimated_dh
    params_errors = (nom_params_errors - est_params_errors)

    printable_res = np.concatenate((poses/DEG, nominal_dist, estimated_dist, delta), axis=1)
    avg_error = np.sum(delta) / len(delta)
    avg_nom_error = np.sum(nominal_dist)/len(nominal_dist)
    avg_est_error = np.sum(estimated_dist) / len (estimated_dist)
    median = np.array([0, 0, 0, 0, 0, 0, avg_nom_error, avg_est_error, avg_error]).reshape(1, 9)
    printable_res = np.append(printable_res, median, axis=0)
    #write_dataset(printable_res, model.results_file, model.fieldnames_options["test_result"], tolerance=".3f")
    return avg_nom_error, avg_est_error

def _calibrate_batch(model: hayati_model.HayatiModel, tries_count, generate):
    copy_model = copy.deepcopy(model)
    param_errors = np.zeros(len(model.error_list))

    local_nominal_error = 0.0
    local_estimated_error = 0.0
    for _ in range(tries_count):
        if generate == "true":
            new_dh = generate_real_dh(copy_model)
            copy_model.change_real_dh(new_dh)
        dataset = measure_all_distances(copy_model)
        param_errors += execute_calibration(copy_model, dataset=dataset)
        #param_errors += execute_calibration_group(copy_model, [[0, 6]], dataset=dataset)
        new_nom, new_est = check_results(copy_model)
        local_nominal_error += new_nom
        local_estimated_error += new_est
        copy_model.reset_estimated_dh()
    
    param_errors /= tries_count
    local_nominal_error /= tries_count
    local_estimated_error /= tries_count
    return local_nominal_error, local_estimated_error, param_errors

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
    params_error = np.array([x[2] for x in results])

    nominal_error_median = sum(nominal_error)/len(nominal_error)
    estimated_error_median = sum(estimated_error)/len(estimated_error)
    params_error_median = np.sum(params_error, axis=0)/len(params_error)

    return nominal_error_median, estimated_error_median, params_error_median

def main(args):
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    model = hayati_model.HayatiModel(config)

    nominal_error, estimated_error, params_error = make_many_calibration_attempts(model, args.num_of_tries, genarate=args.generate)
    print(f"Nominal error: {nominal_error:.6f}\nAfter calibration: {estimated_error:.6f}\nDifference: {(nominal_error - estimated_error):.6f}")
    print(params_error)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Name of .json configuration file. Default: ARM95.json", default="ARM95_new.json")
    parser.add_argument("-g", "--generate", help="Generate new real DH. Default: 'false'", default="true")
    parser.add_argument("-n", "--num_of_tries", help="How many tries to make. Default: 1", type=int, default=1)
    args = parser.parse_args()
    main(args)
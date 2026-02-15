import numpy as np
from math import cos, sin, pi, sqrt, atan2, asin, log10, acos, copysign
from typing import Union
from scipy.optimize import minimize_scalar, minimize
import argparse
import json
import time
import hayati_model
from random import uniform
from dataset_generation_absolute import generate_real_dh
from dataset_generation_wire import make_calibration_dataset, measure_all_distances, FIELDNAMES_OPTIONS, ERRORS_LIST
from csv_routines import read_dataset, write_dataset
from multiprocessing import Pool, cpu_count
import copy

DEG = np.pi / 180

def execute_calibration(model: hayati_model.HayatiModel, dataset=[]):
    if len(dataset) == 0:
        dataset = read_dataset(model.calibration_dataset, FIELDNAMES_OPTIONS["wire_samples"]) 
        dataset[:,0:6] = dataset[:,0:6] * DEG
    for i in range (len(ERRORS_LIST)):
        single_param_dataset = dataset[dataset[:, 6] == i]
        if(len(single_param_dataset) == 0):
            continue

        optimizing_param = calibrate_single_param(model, single_param_dataset)        
        
        param_num, param_letter_num = model.params_from_letters(ERRORS_LIST[i])
        # nominal_value = model.nominal_dh[param_num][param_letter_num]
        # real_value = model.real_dh[param_num][param_letter_num]
        final_value = optimizing_param.x[0]
        # error = final_value - real_value
        # start_error = nominal_value - real_value      

        # if (real_value != 0):         
        #     start_error_rel = start_error/real_value 
        #     end_error_rel = error/real_value 
        # else:
        #     start_error_rel = 1.0
        #     end_error_rel = 1.0

        # print(f"Nominal value: {nominal_value:.6f}")      
        # print(f"Found value: {final_value:.6f}")
        # print(f"Real value: {real_value:.6f}")

        # print(f"Start error: {start_error:.6f}; {start_error_rel* 100:.2f}%")
        # print(f"End error: {error:.6f}; {end_error_rel * 100:.4}%")

        # print("------------------------------------")

        model.estimated_dh[param_num][param_letter_num] = final_value 
        #print(final_value)       

def calibrate_single_param(model: hayati_model.HayatiModel, dataset: Union[np.ndarray, list], delta=10/1000):
    param_index = int(dataset[0,6])
    param_num, param_letter_num = model.params_from_letters(ERRORS_LIST[param_index])
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

    #print(f"Optimizing param {ERRORS_LIST[param_index]}...")

    res = minimize(loss_function, x0=model.nominal_dh[param_num][param_letter_num], 
                   method='Nelder-Mead', tol=1e-10)
    return res

def calculate_estimated_distance(model: hayati_model.HayatiModel, angles: Union[np.ndarray, list], zero_pos: Union[np.ndarray, list]):
    [x, y, z] = model.get_transition_matrix(angles, "estimated")[0:3, 3]
    #[x_0, y_0, z_0] = model.get_transition_matrix(model.zero_wire_angles, "estimated")[0:3, 3]
    distance = sqrt((x - zero_pos[0])**2 + (y - zero_pos[1])**2 + (z - zero_pos[2])**2)
    return distance

def check_results(model: hayati_model.HayatiModel):
    poses = read_dataset(model.test_dataset_file, FIELDNAMES_OPTIONS["random_poses"])
    
    nominal_data = np.array([model.get_transition_matrix(pose, "nominal")[0:3, 3] for pose in poses])
    real_data = np.array([model.get_transition_matrix(pose, "real")[0:3, 3] for pose in poses])
    estimated_data = np.array([model.get_transition_matrix(pose, "estimated")[0:3, 3] for pose in poses])

    nominal_diff = (real_data - nominal_data) * 1000 # mm
    estimated_diff = (real_data - estimated_data) * 1000 # mm
    nominal_dist = np.linalg.norm(nominal_diff, axis=1).reshape(len(poses), 1)
    estimated_dist = np.linalg.norm(estimated_diff, axis=1).reshape(len(poses), 1)
    delta = nominal_dist - estimated_dist

    printable_res = np.concatenate((poses/DEG, nominal_dist, estimated_dist, delta), axis=1)
    avg_error = np.sum(delta) / len(delta)
    avg_nom_error = np.sum(nominal_dist)/len(nominal_dist)
    avg_est_error = np.sum(estimated_dist) / len (estimated_dist)
    median = np.array([0, 0, 0, 0, 0, 0, avg_nom_error, avg_est_error, avg_error]).reshape(1, 9)
    printable_res = np.append(printable_res, median, axis=0)
    write_dataset(printable_res, model.results_file, FIELDNAMES_OPTIONS["test_result"], tolerance=".3f")
    return avg_nom_error, avg_est_error

def _calibrate_batch(model: hayati_model.HayatiModel, tries_count, generate):
    copy_model = copy.deepcopy(model)

    local_nominal_error = 0.0
    local_estimated_error = 0.0
    for _ in range(tries_count):
        if generate == "True":
            new_dh = generate_real_dh(copy_model, model.angle_delta, model.distance_delta)
            copy_model.change_real_dh(new_dh)
        dataset = measure_all_distances(copy_model)
        execute_calibration(copy_model, dataset=dataset)
        new_nom, new_est = check_results(copy_model)
        local_nominal_error += new_nom
        local_estimated_error += new_est
        copy_model.reset_estimated_dh()
    
    local_nominal_error /= tries_count
    local_estimated_error /= tries_count
    return local_nominal_error, local_estimated_error

def make_many_calibration_attempts(model: hayati_model.HayatiModel, tries_num: int, genarate="False"):
    num_cores = cpu_count()
    chunk_size = max(1, (tries_num // num_cores))

    tasks = []
    for i in range(num_cores):
        current_tries = chunk_size
        if current_tries > 0:
            tasks.append((model, chunk_size, genarate))        

    with Pool(processes=num_cores) as pool:
        results = pool.starmap(_calibrate_batch, tasks)

    nominal_error = [x[0] for x in results]
    estimated_error = [x[1] for x in results]

    nominal_error_median = sum(nominal_error)/len(nominal_error)
    estimated_error_median = sum(estimated_error)/len(estimated_error)


    # for local_nominal_error, local_estimated_error in results:
    #     nominal_error += local_nominal_error
    #     estimated_error += local_estimated_error

    # nominal_error /= len(results)
    # estimated_error /= len(results)

    # for _ in range(tries_num):
    #     if genarate == "True":
    #         new_dh = generate_real_dh(model, angle_delta=0.5*DEG, len_delta=0.5/1000)
    #         model.change_real_dh(new_dh)
    #     make_calibration_dataset(model)
    #     execute_calibration(model)
    #     new_nom, new_est = check_results(model)
    #     nominal_error += new_nom
    #     estimated_error += new_est
    #     model.reset_estimated_dh()
    # nominal_error /= tries_num
    # estimated_error /= tries_num
    return nominal_error_median, estimated_error_median

def main(args):
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    model = hayati_model.HayatiModel(config)

    nominal_error, estimated_error = make_many_calibration_attempts(model, args.num_of_tries, genarate=args.generate)
    print(f"Nominal error: {nominal_error}\nAfter calibration: {estimated_error}\nDifference: {nominal_error - estimated_error}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Name of .json configuration file. Default: ARM95.json", default="ARM95.json")
    parser.add_argument("-g", "--generate", help="Generate new real DH. Default: 'False'", default="False")
    parser.add_argument("-n", "--num_of_tries", help="How many tries to make. Default: 1", type=int, default=1)
    args = parser.parse_args()
    main(args)
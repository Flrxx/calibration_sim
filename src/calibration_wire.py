import numpy as np
from math import cos, sin, pi, sqrt, atan2, asin, log10, acos, copysign
from typing import Union
from scipy.optimize import minimize_scalar, minimize, Bounds
import argparse
import json
import time
import hayati_model
from math_routines import extract_zyx_euler
from random import uniform
from dataset_generation_absolute import generate_real_dh
from dataset_generation_wire import measure_all_distances, calculate_distance
from csv_routines import read_dataset, write_dataset
from multiprocessing import Pool, cpu_count
import copy
from math_routines import DEG


def execute_calibration(model: hayati_model.HayatiModel, dataset=[]):
    is_real = 0
    if len(dataset) == 0:
        is_real = 1
        dataset = read_dataset(model.calibration_dataset, model.fieldnames_options["wire_samples"]) 
        mask = dataset[:, 6] >= 0
        dataset = dataset[mask]
        dataset[:, 0:6] *= DEG
        dataset[:, 7] /= 1000
    param_errors = np.zeros(len(model.error_list))
    params_values = np.zeros(shape=(len(model.error_list), 2))

    for j in range(1):
        for i in range (len(model.error_list)):
            single_param_dataset = dataset[dataset[:, 6] == i]
            # if i == 7:
            #     print(single_param_dataset)
            if(len(single_param_dataset) == 0):
                continue
            param_num, param_letter_num = model.params_from_letters(model.error_list[i])        

            optimizing_param = calibrate_single_param(model, single_param_dataset)     
            
            final_value = optimizing_param.x[0]
            # if i == 7:
            #     print(final_value)
            if not is_real:
                model.estimated_dh[param_num][param_letter_num] = final_value 

                nom_params = model.get_readable_params("nominal")
                est_params = model.get_readable_params("estimated")
                real_params = model.get_readable_params("real")

                nominal_value = nom_params[param_num][param_letter_num] #model.nominal_dh[param_num][param_letter_num]
                real_value = real_params[param_num][param_letter_num] #model.real_dh[param_num][param_letter_num]
                final_value = est_params[param_num][param_letter_num]
                
                final_error = final_value - real_value
                start_error = nominal_value - real_value  

                if (real_value != 0):         
                    start_error_rel = start_error/real_value
                    final_error_rel = final_error/real_value
                    # print(f"start {i}: {start_error_rel}")
                    # print(f"end {i}: {final_error_rel}")

                else:
                    start_error_rel = 100
                    final_error_rel = 100

                if start_error_rel != 0:
                    param_errors[i] = (abs(start_error) - abs(final_error))# / abs(start_error_rel) * 100 #%
                    #print(param_errors)
                else:
                    param_errors[i] = 0
                
            else:
                model.estimated_dh[param_num][param_letter_num] = final_value 


    if not is_real:
        # print(nom_params[:, :4])
        # print()
        # print(est_params[:, :4])
        # print()
        # print(real_params[:, :4])

        return [param_num, param_letter_num], param_errors 
    else:
        return True

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
                   method='SLSQP', tol=1e-16)#, bounds=bounds)
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
    

    # nom_params_errors = model.real_dh - model.nominal_dh
    # est_params_errors = model.real_dh - model.estimated_dh

    #params_errors = (nom_params_errors - est_params_errors)

    avg_nom_error = np.sum(nominal_dist)/len(nominal_dist)
    avg_est_error = np.sum(estimated_dist) / len (estimated_dist)
    avg_est_error_prev = np.sum(estimated_dist_prev) / len (estimated_dist_prev)
    
    delta = nominal_dist - estimated_dist
    printable_res = np.concatenate((poses/DEG, nominal_dist, estimated_dist, delta), axis=1)
    avg_error = np.sum(delta) / len(delta)
    median = np.array([0, 0, 0, 0, 0, 0, avg_nom_error, avg_est_error, avg_error]).reshape(1, 9)
    printable_res = np.append(printable_res, median, axis=0)
    write_dataset(printable_res, "results/ARM95/test_result.csv", ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'd_nom', 'd_est', 'delta'],
                     tolerance=".3f")
    return avg_nom_error, avg_est_error, avg_est_error_prev

def _calibrate_batch(model: hayati_model.HayatiModel, tries_count, generate):
    copy_model = copy.deepcopy(model)
    local_param_errors = np.zeros(len(model.error_list))

    local_nominal_error = 0.0
    local_estimated_error = 0.0
    local_estimated_error_prev = 0.0
    for _ in range(tries_count):
        if generate:
            new_dh = generate_real_dh(copy_model)
            copy_model.change_real_dh(new_dh)
        dataset = measure_all_distances(copy_model)

        last_indexes, param_errors_new  = execute_calibration(copy_model, dataset=dataset) 
        #print(param_errors_new)
        local_param_errors += param_errors_new

        # new_nom, new_est, new_esp_prev = check_results(copy_model, last_indexes)
        # local_nominal_error += new_nom
        # local_estimated_error += new_est
        # local_estimated_error_prev += new_esp_prev
        
        
        copy_model.reset_estimated_dh()
    
    #print(local_param_errors)
    local_param_errors /= (tries_count)
    #print(local_param_errors)
    local_nominal_error /= tries_count
    local_estimated_error /= tries_count
    local_estimated_error_prev /= tries_count
    return local_nominal_error, local_estimated_error, local_estimated_error_prev, local_param_errors

def make_many_calibration_attempts(model: hayati_model.HayatiModel, tries_num: int, genarate="false"):
    num_cores = cpu_count()
    chunk_size = max(1, (tries_num // num_cores))

    tasks = []
    for _ in range(min(num_cores, tries_num)):
        current_tries = chunk_size
        if current_tries > 0:
            tasks.append((model, chunk_size, genarate))        

    with Pool(processes=num_cores) as pool:
        results = pool.starmap(_calibrate_batch, tasks)

    nominal_error = [x[0] for x in results]
    estimated_error = [x[1] for x in results]
    estimated_error_prev = [x[2] for x in results]
    params_error = np.array([x[3] for x in results])
    #print(params_error)

    nominal_error_median = sum(nominal_error)/len(nominal_error)
    estimated_error_median = sum(estimated_error)/len(estimated_error)
    estimated_error_prev_median = sum(estimated_error_prev)/len(estimated_error_prev)
    params_error_median = np.sum(params_error, axis=0)/len(params_error)

    print(f"Nominal error: {nominal_error_median:.6f}\nAfter calibration: {estimated_error_median:.6f}\n\nDifference: {(nominal_error_median - estimated_error_median):.6f}")
    print(f"Previous difference: {(nominal_error_median - estimated_error_prev_median):.6f}\n\nContribution of new: {(estimated_error_prev_median - estimated_error_median):.6f}")
    print(params_error_median)

def write_results(model: hayati_model.HayatiModel):
    tfs = model.get_all_transition_matrixes([0, 0, 0, 0, 0, 0], 'estimated')[1:-1]
    mcx_params = []
    joint_offsets = []
    for index, tf in enumerate(tfs):
        offset = tf[:3, 3]
        rotation = extract_zyx_euler(tf[:3, :3])
        mcx_params.append([offset[0], offset[1], offset[2], 0, rotation[1], rotation[2]])
        joint_offsets.append(model.estimated_dh[index][3] - model.nominal_dh[index][3])

    res_dict = {"estimated_dh": model.estimated_dh.tolist(), "mcx_params": mcx_params, "offsets": joint_offsets}
    with open(model.results_file, 'w') as file:
        json_str = json.dumps(res_dict)
        file.write(json_str.replace("]], ", "]],\n"))

    nom_params = model.get_readable_params("nominal")
    est_params = model.get_readable_params("estimated")
    print(nom_params[:, :4])
    print()
    print(est_params[:, :4])

def validate_wire(model: hayati_model.HayatiModel):
    validation_dataset = read_dataset("results/ARM95/validation_results.csv", model.fieldnames_options["test_result"])
    validation_dataset[:, :6] *= DEG
    validation_dataset[:, 6:] /= 1000

    res = np.zeros(shape=(validation_dataset.shape)) 
    for i, line in enumerate(validation_dataset[:-1]):
        if line[8] <= 0:
            continue
        distance_theoretical = np.linalg.norm(calculate_distance(model, line[0:6], "nominal"))
        distance_estimated = np.linalg.norm(calculate_distance(model, line[0:6], "estimated"))
        
        res[i] = np.concatenate([line[0:6], [distance_theoretical, distance_estimated, 
                                             line[8], (abs(line[8] - distance_theoretical) - abs(line[8] - distance_estimated))]])
    res[-1, 6:] = np.array([np.median(res[:, 6]), np.median(res[:, 7]), np.median(res[:, 8]), np.median(res[:, 9])])
    return res

def write_validation_dataset(model: hayati_model.HayatiModel):
    validation_dataset = validate_wire(model)
    
    validation_dataset[:, :6] /= DEG
    validation_dataset[:, 6:] *= 1000
    write_dataset(validation_dataset, "results/ARM95/validation_results.csv", model.fieldnames_options["test_result"], tolerance = ".3f")
    print("Done")

def main(args):
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    model = hayati_model.HayatiModel(config)
    if args.is_real:
        execute_calibration(model)
        write_results(model)
        write_validation_dataset(model)
    else:
        make_many_calibration_attempts(model, args.num_of_tries, genarate=args.generate)  # ,params_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Name of .json configuration file. Default: src/config/ARM95_calibration.json", default="src/config/ARM95_calibration.json")
    #parser.add_argument("-g", "--generate", help="Generate new real DH. Default: 'true'", default="true")
    parser.add_argument("--generate", action='store_true')
    parser.add_argument("--is_real", action='store_true')
    parser.add_argument("-n", "--num_of_tries", help="How many tries to make. Default: 1", type=int, default=1)
    args = parser.parse_args()
    main(args)
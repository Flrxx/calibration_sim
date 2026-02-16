import csv
import argparse
import json
import os
import numpy as np
from math import sqrt, acos
from random import uniform
from math_routines import extract_zyx_euler
import hayati_model
from typing import Union
from scipy.optimize import minimize
from csv_routines import write_dataset, add_line_csv, read_dataset
from multiprocessing import Pool, cpu_count
import copy
from math_routines import DEG

np.set_printoptions(precision=3, suppress=True, formatter={'all': lambda x: f'{x:0.3f}'})

#ERRORS_LIST = ['theta_6', 'd_6', 'theta_5', 'theta_4', 'a_3', 'd_4', 'a_2', 'theta_3', 'theta_2', 'a_1']

def calculate_jac_DH_params(model: hayati_model.HayatiModel, angles: Union[list, np.ndarray], eps = 1e-6):
    jac_DH = np.zeros((3, len(model.error_list)))
    initial_coords = model.get_transition_matrix(angles, "nominal")[0:3, 3]
    for i, e in enumerate(model.error_list):        
        model.vary_nominal_dh(e, eps)                                     #change DH
        new_coords = model.get_transition_matrix(angles, "nominal")[0:3, 3]
        for j in range(3):
            jac_DH[j, i] = (new_coords[j] - initial_coords[j]) / eps
        model.vary_nominal_dh(e, -eps)                                    #reset DH
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
    wire_direction = calculate_wire_direction(model, angles)
    contribution = np.dot(wire_direction.T, jac_DH - model.jac_DH_0)
    return contribution.flatten()

def calculate_optimal_position(model: hayati_model.HayatiModel, i_target: int, angles_start:Union[list, np.ndarray],
                                bounds:Union[list, np.ndarray], wire_limits:Union[list, np.ndarray], angle_limit: float,
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

    if i_target < len(model.error_list) - 1:
        def future_limits(angles):
            past_values = calculate_contribution(model, angles)[i_target + 1:]
            upper = future_eps - past_values
            lower = past_values + future_eps
            return np.concatenate([upper, lower])
        cons.append({'type': 'ineq', 'fun': future_limits})

    def z_direction_limit(angles):    # so z_new face forward
        z_dir = model.get_transition_matrix(angles, "nominal")[0:3, 2]
        scalar_z = np.vdot(z_dir, -model.zero_wire_direction)
        alpha_z = acos(scalar_z)
        angle_cons_z = np.array([alpha_z, angle_limit - alpha_z]) # so angle would fit

        wire = (model.get_transition_matrix(angles, "nominal")[0:3, 3] - model.zero_offset_nominal)
        scalar_wire = np.vdot(-(wire / np.linalg.norm(wire)), z_dir)
        alpha_wire = acos(scalar_wire)
        angle_cons_wire = np.array([alpha_wire, angle_limit - alpha_wire]) # so angle would fit 

        return np.concatenate(([scalar_z], [scalar_wire], angle_cons_z, angle_cons_wire)) #z_dir[0]    
    cons.append({'type': 'ineq', 'fun': z_direction_limit})

    def wire_limits_cons(angles):
        wire = (model.get_transition_matrix(angles, "nominal")[0:3, 3] - model.zero_offset_nominal)
        wire_len = np.linalg.norm(wire) 
        len_cons = np.array([wire_len - wire_limits[0], wire_limits[1] - wire_len]) # so wire len would fit

        scalar = np.vdot(wire / wire_len, model.zero_wire_direction)  # > 0 so wire stretch forward        
        alpha = acos(scalar)
        angle_cons = np.array([alpha, angle_limit - alpha]) # so angle would fit

        return np.concatenate(([scalar], angle_cons, len_cons))
    
    cons.append({'type': 'ineq', 'fun': wire_limits_cons})

    res = minimize(objective, angles_start, method='SLSQP', 
        bounds=bounds, constraints=cons, tol=1e-6, options={'ftol': 1e-6})
    if res.success:
        return(res.x)
    else:
        return([None for _ in range(6)])

def _generate_batch(tries_count, model, i_target, user_joint_mask, bounds, wire_limits, angle_limit, prev_eps, future_eps, angle_diff):
    np.random.seed()
    
    local_samples = []
    local_contributions = []

    active_joint_indices = np.where(user_joint_mask)[0]
    num_active = len(active_joint_indices)

    if num_active > 0:
        active_bounds = np.array(model.bounds)[active_joint_indices]
        
        all_random_angles = np.random.uniform(
            low=active_bounds[:, 0], 
            high=active_bounds[:, 1], 
            size=(tries_count, num_active)
        )
    
    angles = model.zero_wire_angles.copy()

    for i in range(tries_count):
        if num_active > 0:
            angles[active_joint_indices] = all_random_angles[i]
        
        new_sample = calculate_optimal_position(model, i_target, angles, bounds, wire_limits, angle_limit, prev_eps, future_eps)
        
        if new_sample[0] is not None:
            single_contribution = calculate_contribution(model, new_sample)
            
            similar_index = get_similar_index(local_samples, new_sample, angle_diff * DEG)
            
            if similar_index is None:
                local_samples.append(new_sample.copy())
                local_contributions.append(single_contribution.copy())
            elif abs(single_contribution[i_target]) > abs(local_contributions[similar_index][i_target]):
                local_samples[similar_index] = new_sample.copy()
                local_contributions[similar_index] = single_contribution.copy()

    return local_samples, local_contributions

def generate_single_layer(model: hayati_model.HayatiModel, i_target, user_joint_mask, tries_num, wire_limits, angle_limit, prev_eps, future_eps, angle_diff, erase_previous):
    num_cores = cpu_count()
    #chunk_size = tries_num // num_cores
    chunk_size = max(1, (tries_num // num_cores))

    angles = model.zero_wire_angles.copy()
    bounds = np.array([[angles[i], angles[i]] for i in range(6)])
    bounds = bounds * abs(user_joint_mask - 1) + model.bounds * user_joint_mask

    tasks = []
    for i in range(num_cores):
        current_tries = chunk_size
        if current_tries > 0:
            copy_model = copy.deepcopy(model)
            tasks.append((
                current_tries, copy_model, i_target, user_joint_mask, bounds, 
                wire_limits, angle_limit,
                prev_eps, future_eps, angle_diff
            ))        

    with Pool(processes=num_cores) as pool:
        results = pool.starmap(_generate_batch, tasks)
    
    if erase_previous == "false":
        previous_res = read_dataset(model.generation_output, model.fieldnames_options["wire_contributions"])
        final_samples = (previous_res[:,0:6] * DEG)
        final_contributions = previous_res[:,7:]
    elif erase_previous == "true":
        final_samples = np.array([]).reshape(0, 6)
        final_contributions = np.array([]).reshape(0, len(model.error_list))
    else:
        raise ValueError("Wrong erase_previous")

    for batch_samples, batch_contributions in results:
        for new_sample, single_contribution in zip(batch_samples, batch_contributions):
            
            similar_index = get_similar_index(final_samples, new_sample, angle_diff * DEG)
            
            if similar_index is None:
                final_samples = np.append(final_samples, new_sample.reshape(1, 6), axis = 0)
                final_contributions = np.append(final_contributions, single_contribution.reshape(1, len(model.error_list)), axis = 0)
            elif abs(single_contribution[i_target]) > abs(final_contributions[similar_index][i_target]):
                final_samples[similar_index] = new_sample
                final_contributions[similar_index] = single_contribution

    return {"samples": final_samples, "contributions": final_contributions}

def get_similar_index(samples, new_sample: np.ndarray, delta):     
    if len(samples) == 0:
        return None
    
    #samples_array = np.array(samples)
    diffs = np.abs(samples - new_sample)
    
    is_similar_mask = np.all(diffs <= delta, axis=1)
    if not np.any(is_similar_mask):
        return None
    
    return np.argmax(is_similar_mask)

def generate_samples(model: hayati_model.HayatiModel, i_target: int, user_joint_mask, 
                     tries_num, wire_limits, angle_limit,
                     prev_eps, future_eps, angle_diff, erase_previous):
    res = generate_single_layer(model, i_target, user_joint_mask, tries_num, wire_limits, angle_limit, prev_eps, future_eps, angle_diff, erase_previous)
    printable_res = [np.concatenate((s / DEG, [i_target], c.flatten())).tolist() for s, c in zip(res["samples"], res["contributions"])]
    printable_res.sort(key = lambda x: -abs(x[7 + i_target]))
    write_dataset(printable_res, model.generation_output, model.fieldnames_options["wire_contributions"], tolerance = ".3f")
    print(f"Found {len(printable_res)} poses")

def measure_real_distance(model: hayati_model.HayatiModel, angles: Union[list, np.ndarray]):
    [x, y, z] = model.get_transition_matrix(angles, "real")[0:3, 3]
    distance = sqrt((x - model.zero_offset_real[0])**2 + (y - model.zero_offset_real[1])**2 + (z - model.zero_offset_real[2])**2)
    distance *= 1 + uniform(-1, 1) * model.encoder_abs_tolerance                            # fit tolerance
    if model.encoder_resolution != 0:
        distance = round(distance / model.encoder_resolution) * model.encoder_resolution    # fit resolution
    return distance

def measure_all_distances(model: hayati_model.HayatiModel, dataset=[]):
    #if len(dataset) == 0:
    dataset = read_dataset(model.contribution_dataset, model.fieldnames_options['wire_contributions'])[:, 0:8]
    dataset[:, 0:6] *= DEG
    for sample in dataset:
        sample[7] = measure_real_distance(model, sample[0:6])
    return dataset

def make_calibration_dataset(model: hayati_model.HayatiModel):
    model.jac_DH_0 = calculate_jac_DH_params(model, model.zero_wire_angles)
    dataset = measure_all_distances(model)
    dataset[:, 0:6] /= DEG
    write_dataset(dataset, model.calibration_dataset, model.fieldnames_options["wire_samples"], tolerance=".6f") 

def main(args):
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    model = hayati_model.HayatiModel(config)
    model.jac_DH_0 = calculate_jac_DH_params(model, model.zero_wire_angles)

    user_joint_mask = np.array([1, 1, 1, 1, 1, 1]).reshape(6, 1)
    # i_target = 10
    # max_tries = 1000
    # prev_eps = 10
    # future_eps = 0.0004 
    # angle_diff = 20   

    # angle_limit = 85
     
    generate_samples(model, args.target_index, user_joint_mask, args.tries_num, 
                        model.wire_limits, model.angle_limit, 
                        args.prev_eps, args.future_eps, args.angle_diff, args.erase_previous) 

    # elif mode == "check":
    #     print("nothing for now")
    #     angles = np.array([0 * DEG, 36.1 * DEG, 96.8 * DEG, -90 * DEG, -41.976 * DEG, -81.238 * DEG])
    #     contribution = calculate_contribution(model, angles)
    #     add_line_csv(np.concatenate((angles / DEG, [0], contribution.flatten())).tolist(), model.contribution_dataset, model.fieldnames_options['wire_contributions'], ".3f")
        
    # elif mode == "collect_dist":
    #     make_calibration_dataset(model, model.contribution_dataset, model.calibration_dataset, model.encoder_abs_tolerance, model.encoder_resolution)

    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Name of .json configuration file. Default: ARM95.json", default="ARM95.json")
    #parser.add_argument("-m", "--mode", help="Selected mode: 'generation', 'check' or 'collect_dist'. Default: 'generation'", default="generation")
    parser.add_argument("-e", "--erase_previous", help="Erase previous generation result. Default: 'false'", default="false")
    parser.add_argument("-i", "--target_index", help="Index of target. Default: 0", type=int, default=0)
    parser.add_argument("-t", "--tries_num", help="Number of tries. Default: 10", type=int, default=10)
    parser.add_argument("-a", "--angle_diff", help="Angle difference between poses. Default: 10", type=float, default=10)
    parser.add_argument("-p", "--prev_eps", help="Number of tries. Default: 0.005", type=float, default=0.005)
    parser.add_argument("-f", "--future_eps", help="Number of tries. Default: 0.0004", type=float, default=0.0004)
    args = parser.parse_args()
    main(args)
        

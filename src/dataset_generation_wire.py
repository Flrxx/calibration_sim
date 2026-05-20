import csv
import argparse
import json
import os
import numpy as np
from math import sqrt, acos
from random import uniform, random
from math_routines import extract_zyx_euler
import hayati_model
from typing import Union
from scipy.optimize import minimize
from csv_routines import write_dataset, add_line_csv, read_dataset
from multiprocessing import Pool, cpu_count
import copy
from math_routines import DEG

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

def calculate_distance(model: hayati_model.HayatiModel, angles: Union[list, np.ndarray], type: str):
    if type not in ["estimated", "nominal", "real"]:
        print("Wrong type")
        return(None)
    point_offset = model.get_transition_matrix(angles, type)[0:3, 3]
    wire_vector = (np.array([point_offset[0] - model.zero_offset_nominal[0],
                                point_offset[1] - model.zero_offset_nominal[1],
                                point_offset[2] - model.zero_offset_nominal[2]])).reshape(3, 1)
    return wire_vector

def calculate_wire_direction(model: hayati_model.HayatiModel, angles: Union[list, np.ndarray]):
    wire_vector = calculate_distance(model, angles, "nominal")
    # point_offset = model.get_transition_matrix(angles, "nominal")[0:3, 3]
    # wire_vector = (np.array([point_offset[0] - model.zero_offset_nominal[0],
    #                             point_offset[1] - model.zero_offset_nominal[1],
    #                             point_offset[2] - model.zero_offset_nominal[2]])).reshape(3, 1)
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
    contribution[0, model.angle_idexes] *= DEG * 1000 # mm/deg
    return contribution.flatten()

def calculate_optimal_position(model: hayati_model.HayatiModel, i_target: int, angles_start:Union[list, np.ndarray],
                                bounds:Union[list, np.ndarray], prev_eps: float, future_eps: float, border: int, border_eps: float):
    
    def objective(angles):
        output = calculate_contribution(model, angles)
        
        #return -(output[i_target]**2   -   1 * np.max(output[0:i_target]**2, initial=0.0001) - 5 * np.max(output[i_target + 1:]**2, initial=0.0001))  #/ max(len(output[i_target + 1:]), 1)
        return -(abs(output[i_target]) /( sqrt(np.sum(output[i_target + 1:]**2)) + 0.01  + sqrt(np.sum(output[0:i_target]**2/50))    ) )

    cons = []
    if border == -1:
        border = i_target + 1

    def past_and_future_limits(angles):
        contribution = calculate_contribution(model, angles)
        if i_target > 0:
            past_values = contribution[:i_target]
            past_upper = prev_eps - past_values
            past_lower = past_values + prev_eps
        else:
            past_upper = [0]
            past_lower = [0]
        if border > 0 and border != i_target + 1:            
            border_values = contribution[i_target + 1:border] # +!1!!!
            border_upper = border_eps - border_values
            border_lower = border_values + border_eps
        else:
            border_upper = [0]
            border_lower = [0]

        if i_target < len(model.error_list) - 1:
            future_values = contribution[border:] # +!1!!!
            future_upper = future_eps - future_values
            future_lower = future_values + future_eps
        else:
            future_upper = [0]
            future_lower = [0]
        
        return np.concatenate([past_upper, past_lower, border_upper, border_lower, future_upper, future_lower])        
    cons.append({'type': 'ineq', 'fun': past_and_future_limits})

    # if border < len(model.error_list) - 1:
    #     def future_limits(angles):
    #         upper = future_eps - past_values
    #         lower = past_values + future_eps
    #         return np.concatenate([upper, lower])
    #     cons.append({'type': 'ineq', 'fun': future_limits})

    def value_floor(angles):
        contribution = calculate_contribution(model, angles)
        return abs(contribution[i_target]) - max(abs(contribution[i_target + 1:]))
    #cons.append({'type': 'ineq', 'fun': value_floor})

    def wire_limits(angles):    
        # z limits
        matrix = model.get_transition_matrix(angles, "nominal")
        z_dir = matrix[0:3, 2]
        scalar_z = np.vdot(z_dir, -model.zero_wire_direction)
        
        #np.clip(scalar_z, -1.0, 1.0)
        #alpha_z = acos(scalar_z)
        #angle_cons_z = np.array([alpha_z, model.angle_limit_z - alpha_z]) # so angle would fit

        wire = matrix[0:3, 3] - model.zero_offset_nominal
        wire_len = np.linalg.norm(wire) 
        scalar_wire = np.vdot(-(wire / wire_len), z_dir)
        np.clip(scalar_wire, -1, 1)
        alpha_z = acos(scalar_wire)
        angle_cons_z = np.array([alpha_z, model.angle_limit_z - alpha_z]) # so angle would fit
        
        # wire limits        
        len_cons = np.array([wire_len - model.wire_limits[0], model.wire_limits[1] - wire_len]) # so wire len would fit

        scalar_wire = np.vdot(wire / wire_len, model.zero_wire_direction)  # > 0 so wire stretch forward        
        alpha = acos(scalar_wire)
        angle_cons_wire = np.array([alpha, model.angle_limit_wire - alpha]) # so angle would fit

        return np.concatenate(([scalar_z], [scalar_wire], angle_cons_z,  [scalar_wire], angle_cons_wire, len_cons)) #angle_cons_wire,

    cons.append({'type': 'ineq', 'fun': wire_limits})

    res = minimize(objective, angles_start, method='SLSQP', 
        bounds=bounds, constraints=cons, tol=1e-6, options={'ftol': 1e-6})
    if res.success:
        return (res.x, objective(res.x))
    else:
        return([None for _ in range(6)], None)

def _generate_batch(tries_count, model: hayati_model.HayatiModel, i_target, user_joint_mask, bounds, prev_eps, future_eps, angle_diff, len_diff, border: int, border_eps: float):
    np.random.seed()
    
    local_samples = []
    local_contributions = []
    local_objective_values = []
    local_coords = []

    num_active = len(user_joint_mask)

    if num_active > 0:
        active_bounds = model.bounds[user_joint_mask]
        
        all_random_angles = np.random.uniform(
            low=active_bounds[:, 0], 
            high=active_bounds[:, 1], 
            size=(tries_count, num_active)
        )
    
    angles = model.zero_wire_angles.copy()

    for i in range(tries_count):
        if num_active > 0:
            angles[user_joint_mask] = all_random_angles[i]
        
        new_sample, objective_value = calculate_optimal_position(model, i_target, angles, bounds, prev_eps, future_eps, border, border_eps)

        if new_sample[0] is not None:
            single_contribution = calculate_contribution(model, new_sample)
            single_coords = model.get_transition_matrix(new_sample, "nominal")[0:3, 3]
            similar_index = get_similar_index(new_sample, local_samples, angle_diff * DEG, single_coords, local_coords, len_diff / 1000)
            
            if similar_index is None:
                local_samples.append(new_sample.copy())
                local_contributions.append(single_contribution.copy())
                local_objective_values.append(objective_value)
                local_coords.append(single_coords)
            # elif abs(single_contribution[i_target]) > abs(local_contributions[similar_index][i_target]):
            #     local_samples[similar_index] = new_sample.copy()
            #     local_contributions[similar_index] = single_contribution.copy()
            
            elif objective_value < local_objective_values[similar_index]: #abs()
                local_samples[similar_index] = new_sample.copy()
                local_contributions[similar_index] = single_contribution.copy()
                local_objective_values[similar_index] = objective_value
                local_coords[similar_index] = single_coords

    return local_samples, local_contributions, local_objective_values, local_coords

def generate_single_layer(model: hayati_model.HayatiModel, i_target, user_joint_mask, tries_count, prev_eps, future_eps, angle_diff, len_diff, erase_previous, border: int, border_eps: float):
    num_cores = cpu_count()
    #chunk_size = tries_count // num_cores
    chunk_size = max(1, (tries_count // num_cores))

    angles = model.zero_wire_angles.copy()
    bounds = np.array([[angles[i], angles[i]] for i in range(6)])

    num_active = len(user_joint_mask)

    if num_active > 0:
        bounds[user_joint_mask] = model.bounds[user_joint_mask]

    #old_bounds = bounds * abs(user_joint_mask - 1) + model.bounds * user_joint_mask

    tasks = []
    for i in range(min(num_cores, tries_count)):
        current_tries = chunk_size
        if current_tries > 0:
            copy_model = copy.deepcopy(model)
            tasks.append((
                current_tries, copy_model, i_target, user_joint_mask, bounds, 
                prev_eps, future_eps, angle_diff, len_diff, border, border_eps
            ))        

    with Pool(processes=num_cores) as pool:
        results = pool.starmap(_generate_batch, tasks)
    
    lenght = 0
    for res in results:
        lenght += len(res[0])
        
    print(f"Found {lenght} poses")
    

    final_samples = np.array([]).reshape(0, 6)
    final_contributions = np.array([]).reshape(0, len(model.error_list))
    final_objective_values = np.array([]).reshape(0, 1)
    final_coords = np.array([]).reshape(0, 3)

    if erase_previous == "false":
        previous_res = read_dataset(model.generation_output, model.fieldnames_options["wire_contributions"])
        # final_samples = (previous_res[:,0:6] * DEG)
        # final_contributions = previous_res[:,8:]
        # final_objective_values = previous_res[:, 7]
        # final_coords = np.zeros(shape = (len(final_samples), 3))
        # for i in range(len(final_samples)):
        #     final_coords[i] = model.get_transition_matrix(final_samples[i], "nominal")[0:3, 3]

        poses = [np.asarray(row) for row in previous_res[:,0:6] * DEG]
        contributions = [np.asarray(row) for row in previous_res[:,8:]]
        objecticve_values = [np.asarray(row) for row in previous_res[:, 7]]
        coords = [model.get_transition_matrix(pose, "nominal")[0:3, 3] for pose in poses]

        results.append([poses, contributions, objecticve_values, coords])
        # results[0] += (previous_res[:,0:6] * DEG).tolist()
        # results[1] += (previous_res[:,8:]).tolist()
        # results[2] += 


    elif erase_previous == "true":
        pass
        # final_samples = np.array([]).reshape(0, 6)
        # final_contributions = np.array([]).reshape(0, len(model.error_list))
        # final_objective_values = np.array([]).reshape(0, 1)
        # final_coords = np.array([]).reshape(0, 3)
    else:
        raise ValueError("Wrong erase_previous")
    
    for batch_samples, batch_contributions, batch_objective_value, batch_coords in results:
        for new_sample, single_contribution, single_objective_value, single_coords in zip(batch_samples, batch_contributions, batch_objective_value, batch_coords):
            
            similar_index = get_similar_index(new_sample, final_samples, angle_diff * DEG, single_coords, final_coords, len_diff / 1000)
            
            if similar_index is None:
                final_samples = np.append(final_samples, new_sample.reshape(1, 6), axis = 0)
                final_contributions = np.append(final_contributions, single_contribution.reshape(1, len(model.error_list)), axis = 0)
                final_objective_values = np.append(final_objective_values, single_objective_value)
                final_coords = np.append(final_coords, single_coords.reshape(1, 3), axis=0)
            # elif abs(single_contribution[i_target]) > abs(final_contributions[similar_index][i_target]):
            #     final_samples[similar_index] = new_sample
            #     final_contributions[similar_index] = single_contribution
            elif single_objective_value < final_objective_values[similar_index]: #abs
                final_samples[similar_index] = new_sample
                final_contributions[similar_index] = single_contribution
                final_objective_values[similar_index] = single_objective_value
                final_coords[similar_index] = single_coords

    return {"samples": final_samples, "contributions": final_contributions, "objective_values": final_objective_values}

def get_similar_index(new_sample, samples, angle_delta, new_sample_coords, samples_coords, len_delta):     
    if len(samples) == 0:
        return None
    
    diffs_angle = np.abs(samples - new_sample)
    is_similar_mask_angle = np.all(diffs_angle <= angle_delta, axis=1)
    
    dist_tcp = np.linalg.norm(samples_coords - new_sample_coords, axis=1)
    is_similar_mask_len = dist_tcp <= len_delta
    
    combined_mask = is_similar_mask_angle | is_similar_mask_len
    
    if np.any(combined_mask):
        return np.argmax(combined_mask)
    
    return None

def generate_samples(model: hayati_model.HayatiModel, i_target: int, user_joint_mask, 
                     tries_count, prev_eps, future_eps, angle_diff, len_diff, erase_previous, border: int, border_eps: float):
    res = generate_single_layer(model, i_target, user_joint_mask, tries_count, prev_eps, future_eps, angle_diff, len_diff, erase_previous, border, border_eps)
    printable_res = [np.concatenate((s / DEG, [i_target, o], c.flatten())).tolist() for s, c, o in zip(res["samples"], res["contributions"], res["objective_values"])]
    printable_res.sort(key = lambda x: x[7 ]) # + i_target
    write_dataset(printable_res, model.generation_output, model.fieldnames_options["wire_contributions"], tolerance = ".3f")
    print(f"Left {len(printable_res)} poses")

def measure_real_distance(model: hayati_model.HayatiModel, angles: Union[list, np.ndarray]):
    [x, y, z] = model.get_transition_matrix(angles, "real")[0:3, 3]
    distance = sqrt((x - model.zero_offset_real[0])**2 + (y - model.zero_offset_real[1])**2 + (z - model.zero_offset_real[2])**2)
    distance *= 1 + uniform(-1, 1) * model.encoder_abs_tolerance                            # fit tolerance
    if model.encoder_resolution != 0:
        distance = round(distance / model.encoder_resolution) * model.encoder_resolution    # fit resolution
   
    # point_offset = model.get_transition_matrix(angles, "nominal")[0:3, 3]
    # wire_vector = (np.array([point_offset[0] - model.zero_offset_nominal[0],
    #                             point_offset[1] - model.zero_offset_nominal[1],
    #                             point_offset[2] - model.zero_offset_nominal[2]])).reshape(3, 1)
    # wire_norm = np.linalg.norm(wire_vector)
    # print(f"nominal: {(wire_norm * 1000):.3f} real: {(distance * 1000):.3f}")

    return distance

def measure_all_distances(model: hayati_model.HayatiModel, dataset=[]):
    if len(dataset) == 0:
        dataset = read_dataset(model.contribution_dataset, model.fieldnames_options['wire_contributions'])[:, 0:8]
        dataset[:, 0:6] *= DEG
    mask = dataset[:, 6] >= 0
    dataset = dataset[mask]
    for sample in dataset:
        sample[7] = measure_real_distance(model, sample[0:6])
    return dataset

def make_calibration_dataset(model: hayati_model.HayatiModel):
    model.jac_DH_0 = calculate_jac_DH_params(model, model.zero_wire_angles)
    dataset = measure_all_distances(model)
    dataset[:, 0:6] *= DEG
    write_dataset(dataset, model.calibration_dataset, model.fieldnames_options["wire_samples"], tolerance=".6f") 

def vary_joints(model: hayati_model.HayatiModel, user_joint_mask: Union[list, np.ndarray], tries_count: int):

    angles = model.zero_wire_angles.copy()
    bounds = np.array([[angles[i], angles[i]] for i in range(6)])

    num_active = len(user_joint_mask)

    if num_active > 0:
        bounds[user_joint_mask] = model.bounds[user_joint_mask]
        num_active = len(user_joint_mask)

        active_bounds = model.bounds[user_joint_mask]
        
        all_random_angles = np.random.uniform(
            low=active_bounds[:, 0], 
            high=active_bounds[:, 1], 
            size=(tries_count, num_active)
        )
        angles = model.zero_wire_angles.copy()

        for i in range(tries_count):
            # random_angles = np.random.uniform(
            #     low=active_bounds[:, 0], 
            #     high=active_bounds[:, 1]
            # )
            angles[user_joint_mask] = all_random_angles[i]
            # if np.any(wire_limits(angles) <= 0):
            #     continue
            contribution = calculate_contribution(model, angles)
            add_line_csv(np.concatenate((angles / DEG, [0], contribution.flatten())).tolist(), "datasets/ARM95/wire/test.csv", model.fieldnames_options['wire_contributions'], ".3f")

def correct_poses(model: hayati_model.HayatiModel):
    dataset = read_dataset(model.contribution_dataset, model.fieldnames_options['wire_contributions'])
    dataset[:, 0:6] *= DEG
    # mask = dataset[:, 6] >= 0
    # dataset = dataset[mask]
    new_samples = np.zeros(shape=(dataset.shape))
    
    user_joint_mask = [0, 1, 2, 3, 4, 5]
    angles = model.zero_wire_angles.copy()
    bounds = np.array([[angles[i], angles[i]] for i in range(6)])
    num_active = len(user_joint_mask)

    if num_active > 0:
        bounds[user_joint_mask] = model.bounds[user_joint_mask]

    for i, pose in enumerate(dataset):
        target_index = int(pose[6])
        if target_index < 0:
            new_samples[i, :] = pose
            continue
        new_pose = calculate_optimal_position(model, target_index, pose[0:6], bounds, np.max(pose[7: 7 + target_index], initial=0.0001), np.max(pose[target_index + 7 + 1:], initial=0.0001), -1, 1)
        
        if new_pose[0] != None:
            delta = 15 * DEG
            diffs = np.abs(new_pose - pose[0:6])    
            if np.any(diffs >= delta):
                new_pose = pose[0:6]
        else: 
            new_pose = pose[0:6]

        new_samples[i, :6] = new_pose
        new_samples[i, 6] = target_index
        new_samples[i, 7:] = calculate_contribution(model, new_samples[i, :6])
    
    new_samples[:, :6] /= DEG
    write_dataset(new_samples, model.contribution_dataset, model.fieldnames_options["wire_contributions"], tolerance=".3f") 
    print("Done")

def get_dataset_for_validation(model: hayati_model.HayatiModel, tries_num: int, angle_limit, wire_limits, angle_diff):
    with open(model.results_file, 'r') as config_file:
        config = json.load(config_file)
    model.estimated_dh = copy.deepcopy(config["estimated_dh"])
    i = 0
    printable_res = np.zeros(shape=(tries_num, len(model.fieldnames_options["test_result"])))
    bounds = copy.deepcopy(model.bounds)
    bounds[0, :] = np.array([-45 * DEG, 45 * DEG])
    bounds[1, :] = np.array([-40 * DEG, 40 * DEG])
    bounds[4, :] = np.array([60 * DEG, 120 * DEG])

    while i < tries_num:
        random_angles = np.array([model.joint_limits_general_l[axis] + random() * (model.joint_limits_general_h[axis] - model.joint_limits_general_l[axis]) for axis in range(6)], dtype='float')
        angles = model.satisfies_wire_limits(random_angles, bounds, angle_limit, wire_limits)
        if angles[0] == None:
            continue

        similar_index = get_similar_index(printable_res[:, :6], angles, angle_diff * DEG)
        if similar_index is not None:
            continue

        random_point = model.get_transition_matrix(angles, "nominal")
        if model.satisfies_cartesian_limits(random_point):
            distance_theoretical = np.linalg.norm(calculate_distance(model, angles, "nominal"))
            distance_estimated = np.linalg.norm(calculate_distance(model, angles, "estimated"))
            printable_res[i] = np.concatenate([angles, [distance_theoretical, distance_estimated, -1, -1]])
            i += 1
    printable_res[:, :6] /= DEG
    printable_res[:, 6:] *= 1000
    write_dataset(printable_res, "datasets/ARM95/validation_dataset.csv", model.fieldnames_options["test_result"], tolerance=".3f")

#def auto_collect_layer

def main(args):
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    model = hayati_model.HayatiModel(config)
    model.jac_DH_0 = calculate_jac_DH_params(model, model.zero_wire_angles)

    if args.mode == 'generation':
        generate_samples(model, args.target_index, np.array(args.user_joint_mask), args.tries_count, 
                            args.prev_eps, args.future_eps, args.angle_diff, args.len_diff, args.erase_previous, args.border, args.border_eps) 
    elif args.mode == 'check':
        #angles = np.array(args.position) * DEG
        vary_joints(model, np.array(args.user_joint_mask), args.tries_count) #args.user_joint_mask)
        #angles = np.array([14.95 * DEG, 23.83 * DEG, 139.02 * DEG, -72.84 * DEG, 90.00 * DEG, 14.95 * DEG])
        # contribution = calculate_contribution(model, angles)
        # add_line_csv(np.concatenate((angles / DEG, [0], contribution.flatten())).tolist(), "datasets/ARM95/wire/test.csv", model.fieldnames_options['wire_contributions'], ".3f")
    elif args.mode == 'correct':
        correct_poses(model)
    elif args.mode == 'validation':
        get_dataset_for_validation(model, args.tries_count, 15 * DEG, [100 / 1000, 600 / 1000], 15*DEG)
    else:
        print("Wrong mode")

    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Name of .json configuration file. Default: ARM95.json", default="src/config/ARM95_calibration.json")
    parser.add_argument("-m", "--mode", help="Selected mode: 'generation' or 'check'. Default: 'generation'", default="generation")
    parser.add_argument("-mask", "--user_joint_mask", help="", nargs='+', type=int, default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("-e", "--erase_previous", help="Erase previous generation result. Default: 'false'", default="false")
    parser.add_argument("-i", "--target_index", help="Index of target. Default: 0", type=int, default=1)
    parser.add_argument("-n", "--tries_count", help="Number of tries. Default: 10", type=int, default=0)
    parser.add_argument("-a", "--angle_diff", help="Angle difference between poses. Default: 5", type=float, default=7)
    parser.add_argument("-l", "--len_diff", help="Lenght difference between poses, mm. Default: 2", type=float, default=50)
    parser.add_argument("-p", "--prev_eps", help="Epsilon for previous params. Default: 0.005", type=float, default=100)
    parser.add_argument("-f", "--future_eps", help="Epsilon for future params. Default: 0.0004", type=float, default=100)
    parser.add_argument("-b", "--border", help="", type=int, default=-1)
    parser.add_argument("-b_e", "--border_eps", help="", type=float, default=0.01)

    args = parser.parse_args()
    main(args)
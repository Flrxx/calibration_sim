import csv
import argparse
import json
import os
import numpy as np
from math import sqrt, acos
from random import uniform, random, sample
from math_routines import extract_zyx_euler
import hayati_model
from typing import Union
from scipy.optimize import minimize
from csv_routines import write_dataset, add_line_csv, read_dataset
from multiprocessing import Pool, cpu_count
import copy
from math_routines import DEG
import warnings
from scipy.linalg import svd

def calculate_derivatives(model: hayati_model.HayatiModel, angles: Union[list, np.ndarray], eps=1e-5):
    """
    Вычисляет первую и вторую производные длины троса по каждому параметру.
    """
    first_deriv = np.zeros(len(model.error_list))
    second_deriv = np.zeros(len(model.error_list))
    
    # Центральная точка
    zero_pos = model.get_transition_matrix(angles, "nominal")[0:3, 3]
    L_center = np.linalg.norm(zero_pos - model.zero_offset_nominal)
    
    for i, p_name in enumerate(model.error_list):
        # Шаг вперед (+eps)
        model.vary_nominal_dh(p_name, eps)
        pos_plus = model.get_transition_matrix(angles, "nominal")[0:3, 3]
        L_plus = np.linalg.norm(pos_plus - model.zero_offset_nominal)
        model.vary_nominal_dh(p_name, -eps) # Возврат
        
        # Шаг назад (-eps)
        model.vary_nominal_dh(p_name, -eps)
        pos_minus = model.get_transition_matrix(angles, "nominal")[0:3, 3]
        L_minus = np.linalg.norm(pos_minus - model.zero_offset_nominal)
        model.vary_nominal_dh(p_name, eps) # Возврат
        
        # Первая производная (Градиент) - метод центральных разностей
        first_deriv[i] = (L_plus - L_minus) / (2 * eps)
        
        # Вторая производная (Кривизна / Гессиан)
        second_deriv[i] = (L_plus - 2 * L_center + L_minus) / (eps**2)
        
    return first_deriv, second_deriv

def evaluate_snr_curvature_penalty(model, angles, numerical_target_index, nonlinearity_penalty=0.5, drift_weight=0.5):
    """
    Оценка SNR с учетом штрафа за нелинейность (стабильность градиента) 
    и штрафа за "дрейф" (удаление от базовой комфортной позы).
    
    :param drift_weight: Чем выше значение (например, 1.0 или 2.0), тем сильнее
                         оптимизатор будет "привязан" к начальным координатам.
    """
    first_deriv, second_deriv = calculate_derivatives(model, angles)
    
    # 1. Полезный сигнал (Первая производная)
    target_signal = abs(first_deriv[numerical_target_index])
    
    # 2. Нестабильность целевого параметра (Вторая производная)
    target_curvature = abs(second_deriv[numerical_target_index])
    
    # 3. Кинематический шум
    noise_uncalib = np.sum(first_deriv[numerical_target_index + 1:]**2)
    noise_calib = np.sum((first_deriv[0:numerical_target_index]**2) / 50.0)
    total_noise = np.sqrt(noise_uncalib) + 0.01 + np.sqrt(noise_calib)
    
    # 4. Штраф за дрейф (L2-регуляризация)
    # Вычисляем расстояние от текущей позы до базовой в радианах.
    # Это "стягивает" все точки обратно к начальному положению (например, к 14 град для q1).
    angles_arr = np.array(angles)
    zero_arr = np.array(model.zero_wire_angles)
    drift_distance = np.linalg.norm(angles_arr - zero_arr)
    
    # НОВЫЙ SNR: Штрафуем за кривизну И за сильное удаление от нулевой позы
    robust_snr = target_signal / (total_noise + nonlinearity_penalty * target_curvature + drift_weight * drift_distance)
    
    return robust_snr

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

def calculate_optimal_position(model, numerical_target_index: int, angles_start: Union[list, np.ndarray],
                               bounds: Union[list, np.ndarray], prev_eps: float, future_eps: float,
                               limit_flange_deg: float = 75.0):
    
    def objective(angles):
        output = calculate_contribution(model, angles)

        # 1. Проверка пользовательских динамических коллизий
        if model.violates_dynamic_limits(angles):
            return 1e6
            
        matrix = model.get_transition_matrix(angles, "nominal")
        wire = matrix[0:3, 3] - model.zero_offset_nominal
        wire_len_mm = np.linalg.norm(wire) * 1000.0 # Предполагаем, что модель в метрах
        
        scalar_wire = np.vdot(wire / np.linalg.norm(wire), model.zero_wire_direction)
        alpha_deg = acos(np.clip(scalar_wire, -1.0, 1.0)) / DEG
        
        if wire_len_mm < 150.0 and alpha_deg > 45.0:
            return 1e6 # Жесткий штраф 

        # 3. Расчет SNR (Твоя формула)
        snr = (abs(output[numerical_target_index]) / 
              (sqrt(np.sum(output[numerical_target_index + 1:]**2)) + 0.01 + 
               sqrt(np.sum(output[0:numerical_target_index]**2 / 50))))
               
        return -snr

    def wire_limits(angles):
        """
        SLSQP ожидает, что все значения массива constraints будут >= 0 для валидной позы.
        """
        matrix = model.get_transition_matrix(angles, "nominal")
        z_dir = matrix[0:3, 2]
        wire = matrix[0:3, 3] - model.zero_offset_nominal
        wire_len = np.linalg.norm(wire) 
        
        if wire_len < 1e-6:
            return np.array([-1.0]) # Fail-safe для предотвращения деления на ноль
            
        wire_norm = wire / wire_len
        
        # 1. Лимит угла ФЛАНЦА (Адаптивный, передан в функцию)
        scalar_z = np.vdot(-wire_norm, z_dir)
        scalar_z = np.clip(scalar_z, -1.0, 1.0) # ИСПРАВЛЕН БАГ: переприсваивание
        alpha_z = acos(scalar_z)
        angle_cons_z = (limit_flange_deg * DEG) - alpha_z # Должно быть >= 0
        
        # 2. ГЛОБАЛЬНЫЙ лимит угла ДАТЧИКА (Защита корпуса 75 градусов)
        scalar_wire = np.vdot(wire_norm, model.zero_wire_direction)
        scalar_wire = np.clip(scalar_wire, -1.0, 1.0) # ИСПРАВЛЕН БАГ
        alpha_wire = acos(scalar_wire)
        angle_cons_wire = (75.0 * DEG) - alpha_wire # Должно быть >= 0
        
        # 3. Лимиты длины троса
        len_min_cons = wire_len - model.wire_limits[0] # Должно быть >= 0
        len_max_cons = model.wire_limits[1] - wire_len # Должно быть >= 0

        # Возвращаем плоский массив со всеми условиями
        return np.array([angle_cons_z, angle_cons_wire, len_min_cons, len_max_cons])

    # Оформляем ограничения для scipy
    cons = [{'type': 'ineq', 'fun': wire_limits}]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Values in x were outside bounds")
        res = minimize(
            objective, 
            angles_start, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=cons, 
            tol=1e-6, 
            options={'ftol': 1e-6, 'maxiter': 50}
        )
        
    if res.success:
        return (res.x, res.fun) # res.fun уже содержит значение objective(res.x)
    else:
        return ([None for _ in range(6)], None)

def _generate_batch(initial_angles, model: hayati_model.HayatiModel, i_target, start_joint, bounds, prev_eps, future_eps, angle_diff, len_diff):
    np.random.seed()
    
    local_samples = []
    local_contributions = []
    local_objective_values = []
    local_coords = []

    # num_active = len(start_joint)

    # if num_active > 0:
    #     active_bounds = model.bounds[start_joint]
        
    #     all_random_angles = np.random.uniform(
    #         low=active_bounds[:, 0], 
    #         high=active_bounds[:, 1], 
    #         size=(tries_count, num_active)
    #     )
    
    # angles = model.zero_wire_angles.copy()
    numerical_target_index = model.error_list.index(i_target)

    for single_pose in initial_angles:
        # if num_active > 0:
        #     angles[start_joint] = all_random_angles[i]

        new_sample, objective_value = calculate_optimal_position(model, numerical_target_index, single_pose, bounds, prev_eps, future_eps)

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

def generate_single_layer(model: hayati_model.HayatiModel, i_target, start_joint, tries_count, prev_eps, future_eps, angle_diff, len_diff, erase_previous):
    num_cores = cpu_count()
    #chunk_size = tries_count // num_cores
    chunk_size = max(1, (tries_count // num_cores))

    angles = model.zero_wire_angles.copy()
    bounds = np.array([[angles[i], angles[i]] for i in range(6)])

    #start_joint = 0
    for j in range(start_joint, 6):
        bounds[j] = model.bounds[j]

    # initial_angles = np.zeros(shape=(tries_count, len(model.zero_wire_angles)))
    # i = 0
    # attempts = 0
    # while attempts < 10000 and i < tries_count:
    #     attempts += 1        
    #     q_rand = np.copy(model.zero_wire_angles)
    #     for j in range(start_joint, 6):
    #         q_rand[j] = np.random.uniform(low=model.joint_limits_general_l[j], high=model.joint_limits_general_h[j])
            
    #     if model.is_pose_valid(q_rand):
    #         initial_angles[i] = q_rand
    #         i += 1

    # if i < tries_count:
    #     print(f"Warning: only {i} valid poses generated out of {tries_count} requested.")

    initial_angles = generate_initial_poses(model, i_target, start_joint, tries_count)
    print(f"Found {len(initial_angles)} initial poses")

    tasks = []
    for i in range(min(num_cores, tries_count)):
        current_tries = chunk_size
        if current_tries > 0:
            copy_model = copy.deepcopy(model)
            tasks.append((
                initial_angles[i * chunk_size: min((i+1)*chunk_size, tries_count)].copy(), copy_model, i_target, start_joint, bounds, 
                prev_eps, future_eps, angle_diff, len_diff
            ))        

    with Pool(processes=num_cores) as pool:
        results = pool.starmap(_generate_batch, tasks)

    # scores = []
    # for task in tasks:
    #     # *task распаковывает кортеж аргументов в функцию
    #     scores.append(_generate_batch(*task))
    
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

def generate_samples(model: hayati_model.HayatiModel, i_target: int, start_joint, 
                     tries_count, prev_eps, future_eps, angle_diff, len_diff, erase_previous):
    res = generate_single_layer(model, i_target, start_joint, tries_count, prev_eps, future_eps, angle_diff, len_diff, erase_previous)
    printable_res = [(s / DEG).tolist() + [i_target, o] + c.flatten().tolist() for s, c, o in zip(res["samples"], res["contributions"], res["objective_values"])]
    printable_res.sort(key = lambda x: x[7 ]) # + i_target
    write_dataset(printable_res, model.generation_output, model.fieldnames_options["wire_contributions"], tolerance = 3)
    print(f"Left {len(printable_res)} poses")

def measure_real_distance(model, angles: Union[list, np.ndarray]):
    [x, y, z] = model.get_transition_matrix(angles, "real")[0:3, 3]
    distance = sqrt((x - model.zero_offset_real[0])**2 + (y - model.zero_offset_real[1])**2 + (z - model.zero_offset_real[2])**2)
    
    # 1. Защита для одиночных вызовов: 
    # Если функция вызвана напрямую (вне measure_all_distances), генерируем масштабную ошибку
    if not hasattr(model, 'encoder_linear_scale'):
        model.encoder_linear_scale = uniform(-1, 1) * model.encoder_abs_tolerance
        
    # 2. Применяем зафиксированную систематическую (линейную) ошибку
    distance *= 1 + model.encoder_linear_scale 
    
    # 3. Применяем разрешение энкодера (ступенчатость)
    if model.encoder_resolution != 0:
        distance = round(distance / model.encoder_resolution) * model.encoder_resolution    
        
    return distance

def measure_all_distances(model, dataset=[]):
    # 1. Принудительно генерируем НОВУЮ систематическую ошибку для этой "виртуальной сессии"
    model.encoder_linear_scale = uniform(-1, 1) * model.encoder_abs_tolerance

    if len(dataset) == 0:
        dataset = read_dataset(model.contribution_dataset, model.fieldnames_options['wire_contributions'])[:, 0:8]
        dataset[:, 0:6] *= DEG
        
    mask = np.array([isinstance(val, str) for val in dataset[:, 6]])
    dataset = dataset[mask]
    
    # 2. Измеряем все точки (measure_real_distance использует уже сгенерированный encoder_linear_scale)
    for sample in dataset:
        sample[7] = measure_real_distance(model, sample[0:6])
        
    # 3. Сбрасываем ошибку после обработки всего датасета.
    # Следующий вызов этой функции (например, в цикле ГА) создаст совершенно новую погрешность.
    if hasattr(model, 'encoder_linear_scale'):
        delattr(model, 'encoder_linear_scale')
        
    return dataset

def make_calibration_dataset(model: hayati_model.HayatiModel, dataset=[]):
    model.jac_DH_0 = calculate_jac_DH_params(model, model.zero_wire_angles)
    dataset = measure_all_distances(model, dataset)
    dataset[:, 0:6] *= DEG
    write_dataset(dataset, model.calibration_dataset, model.fieldnames_options["wire_samples"], tolerance=6) 

def generate_initial_poses(model: hayati_model.HayatiModel, target_param, start_idx, num_candidates):
    valid_poses = []
    signals = []
    attempts = 0
    target_idx = model.error_list.index(target_param)
    
    while len(valid_poses) < num_candidates * 2 and attempts < 30000:
        attempts += 1
        q_rand = np.copy(model.zero_wire_angles)
        
        # # Генерируем точки строго в пределах заданных лимитов
        # for j in range(start_idx, 6):
        #     q_rand[j] = np.random.uniform(low=model.joint_limits_general_l[j], high=model.joint_limits_general_h[j])
            
        for j in range(start_idx, 6):
            dyn_low, dyn_high = model.get_dynamic_limits(q_rand, j, model.joint_limits_general_l[j], model.joint_limits_general_h[j])
            
            # Защита от нелогичных правил (если min > max)
            if dyn_low > dyn_high:
                dyn_low, dyn_high = dyn_high, dyn_low
                
            q_rand[j] = np.random.uniform(low=dyn_low, high=dyn_high)

        if model.is_pose_valid(q_rand):
            contrib = calculate_contribution(model, q_rand)
            signal = abs(contrib[target_idx])
            
            if signal > 0.05:
                valid_poses.append(q_rand)
                signals.append(signal)
                
    if not valid_poses:
        return []
        
    sorted_indices = np.argsort(signals)[::-1]
    best_poses = [valid_poses[i] for i in sorted_indices[:num_candidates]]
    return best_poses

def correct_poses(model: hayati_model.HayatiModel):
    dataset = read_dataset(model.contribution_dataset, model.fieldnames_options['wire_contributions'])
    dataset[:, 0:6] *= DEG
    # mask = dataset[:, 6] >= 0
    # dataset = dataset[mask]
    new_samples = np.zeros(shape=(dataset.shape))
    
    start_joint = [0, 1, 2, 3, 4, 5]
    angles = model.zero_wire_angles.copy()
    bounds = np.array([[angles[i], angles[i]] for i in range(6)])
    num_active = len(start_joint)

    if num_active > 0:
        bounds[start_joint] = model.bounds[start_joint]

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
    write_dataset(new_samples, model.contribution_dataset, model.fieldnames_options["wire_contributions"], tolerance=3) 
    print("Done")

def generate_validation_dataset(model: hayati_model.HayatiModel, num_points: int = 50, num_bins: int = 5):
    """
    Генерирует случайные валидные позы методом Монте-Карло 
    с равномерным распределением (стратификацией) по длине троса.
    """
    print(f"Запуск Монте-Карло: поиск {num_points} валидационных точек...")
    print(f"Включена балансировка распределения длины троса (Корзин: {num_bins})")
    
    valid_poses = []
    attempts = 0
    
    low_lim = model.joint_limits_general_l
    high_lim = model.joint_limits_general_h
    
    # 1. Настройка корзин для длин троса (в метрах)
    l_min = model.wire_limits[0]
    l_max = model.wire_limits[1]
    bin_edges = np.linspace(l_min, l_max, num_bins + 1)
    bins = [[] for _ in range(num_bins)]
    
    # Рассчитываем квоты для каждой корзины (поровну, остаток раскидываем)
    points_per_bin = num_points // num_bins
    remainder = num_points % num_bins
    bin_targets = [points_per_bin + (1 if i < remainder else 0) for i in range(num_bins)]
    
    strict_mode = True # Пока True - жестко следим за наполнением корзин
    
    while len(valid_poses) < num_points and attempts < 1500000:
        attempts += 1
        
        # Векторизованная генерация случайных углов (быстрее цикла)
        q_rand = np.random.uniform(low=low_lim, high=high_lim)
            
        if model.is_pose_valid(q_rand) and not model.violates_dynamic_limits(q_rand):
            
            if strict_mode:
                # В строгом режиме проверяем длину троса
                matrix = model.get_transition_matrix(q_rand, "nominal")
                wire = matrix[0:3, 3] - model.zero_offset_nominal
                wire_len = np.linalg.norm(wire)
                
                # Ищем нужную корзину для этой длины
                bin_idx = -1
                for i in range(num_bins):
                    if bin_edges[i] <= wire_len <= bin_edges[i+1]:
                        bin_idx = i
                        break
                        
                # Если корзина еще не полная — забираем точку
                if bin_idx != -1 and len(bins[bin_idx]) < bin_targets[bin_idx]:
                    bins[bin_idx].append(q_rand)
                    valid_poses.append(q_rand)
            else:
                # Если включился добор, берем любые валидные точки
                valid_poses.append(q_rand)
                
        if attempts % 50000 == 0:
            if strict_mode:
                status = [len(b) for b in bins]
                print(f"  Попыток: {attempts}, найдено: {len(valid_poses)}/{num_points}. Корзины: {status}")
            else:
                print(f"  Попыток: {attempts}, найдено: {len(valid_poses)}/{num_points} (Свободный добор)")
                
        # 2. ПРЕДОХРАНИТЕЛЬ: Если за 500 000 попыток не смогли заполнить корзины,
        # значит какие-то длины (например, самые короткие) физически недостижимы.
        if strict_mode and attempts > 500000:
            print("\n⚠️ Внимание: Не удается заполнить некоторые корзины.")
            print("Вероятно, экстремальные длины физически недостижимы. Включаю свободный добор...")
            strict_mode = False
            
    print(f"Готово! Найдено {len(valid_poses)} точек за {attempts} попыток.")
    
    return valid_poses

def export_validation_csv(model: hayati_model.HayatiModel, poses: list, output_filepath: str):
    """
    Рассчитывает номинальную длину, формирует шаблон и записывает в CSV.
    """
    # Заголовки таблицы
    headers = [
        "q1", "q2", "q3", "q4", "q5", "q6", 
        "d_nom", "d_real", "d_est", "diff"
    ]
    
    rows = []
    sum_d_nom = 0.0
    
    for angles in poses:
        # Считаем номинальную дистанцию
        matrix = model.get_transition_matrix(angles, "nominal")
        wire = matrix[0:3, 3] - model.zero_offset_nominal
        
        d_nom_mm = np.linalg.norm(wire) * 1000.0 
        
        sum_d_nom += d_nom_mm
        
        # Углы переводим в градусы для удобства
        q_deg = angles / DEG
        
        # Формируем строку: углы, номинальная длина, и -1 для пустых колонок
        row = [f"{q:.3f}" for q in q_deg] + [f"{d_nom_mm:.3f}", "-1", "-1", "-1"]
        rows.append(row)
        
    # Рассчитываем среднее только для номинальной длины
    avg_d_nom = sum_d_nom / len(poses) if poses else 0.0
    
    # Динамический номер последней строки для формул Excel
    end_row = len(poses) + 1
    
    # Строка средних значений в конце (-1 вместо пустых ячеек для углов)
    avg_row = [
        "-1", "-1", "-1", "-1", "-1", "-1", 
        f"{avg_d_nom:.3f}", 
        f"-1", 
        f"-1", 
        f"-1"
    ]
    rows.append(avg_row)

    rows.sort(key=lambda x: float(x[6]))
    
    write_dataset(rows, output_filepath, headers, tolerance=3)
        
    print(f"Валидационный датасет успешно сохранен в: {output_filepath}")

def is_pose_safe(model: hayati_model.HayatiModel, angles: np.ndarray) -> bool:
    """
    Проверка на аппаратную безопасность: защита от 'Зоны смерти' и лимитов троса.
    """
    if hasattr(model, 'violates_dynamic_limits') and model.violates_dynamic_limits(angles):
        return False

    matrix = model.get_transition_matrix(angles, "nominal")
    wire = matrix[0:3, 3] - model.zero_offset_nominal
    wire_len = np.linalg.norm(wire) 
    
    if wire_len < model.wire_limits[0] or wire_len > model.wire_limits[1]:
        return False

    wire_norm = wire / wire_len
    scalar_wire_forward = np.vdot(wire_norm, model.zero_wire_direction)
    if scalar_wire_forward <= 0:
        return False
        
    alpha_deg = acos(np.clip(scalar_wire_forward, -1.0, 1.0)) / DEG
    
    wire_len_mm = wire_len * 1000.0
    if wire_len_mm < 150.0 and alpha_deg > 45.0:
        return False # Коротко и криво -> Аппаратный шум датчика
        
    limit_wire_deg = model.angle_limit_wire if model.angle_limit_wire > 3.15 else model.angle_limit_wire / DEG
    if alpha_deg > limit_wire_deg: 
        return False
        
    return True

def calculate_group_jacobian(model: hayati_model.HayatiModel, angles: np.ndarray, target_params: list):
    """
    Считает вектор влияния (строку Якобиана) только для указанной группы параметров.
    """
    full_contrib = calculate_contribution(model, angles)
    indices = [model.error_list.index(p) for p in target_params]
    return full_contrib[indices]

def worker_generate_scored_points(args):
    """
    Один поток берет на себя генерацию и оценку части (chunk) общего пула точек.
    Мы передаем dict 'config', чтобы каждый процесс создал свой независимый объект 
    HayatiModel. Это защищает от багов состояния (shared state) при расчете Якобианов.
    """
    model, wrist_params, base_params, points_to_find = args
    
    # Каждый поток создает свою собственную модель
    copy_model = copy.deepcopy(model)
    copy_model.jac_DH_0 = calculate_jac_DH_params(copy_model, copy_model.zero_wire_angles)
    
    local_scored_pool = []
    low_lim = model.joint_limits_general_l
    high_lim = model.joint_limits_general_h
    attempts = 0
    
    # Ищем, пока не наберем нужную квоту (или не превысим лимит попыток)
    while len(local_scored_pool) < points_to_find and attempts < points_to_find * 100:
        attempts += 1
        q_rand = np.random.uniform(low=low_lim, high=high_lim)
        
        # ЗАМОРОЗКА БАЗЫ
        q_rand[0:3] = copy_model.zero_wire_angles[0:3]
        
        if is_pose_safe(copy_model, q_rand):
            # Точка физически безопасна. Сразу считаем для нее градиенты!
            J_wrist = calculate_group_jacobian(copy_model, q_rand, wrist_params)
            J_base = calculate_group_jacobian(copy_model, q_rand, base_params)
            
            signal = np.linalg.norm(J_wrist)
            noise = np.linalg.norm(J_base)
            
            # Считаем изоляцию от базы
            block_snr = signal / (noise + 1e-6)
            
            local_scored_pool.append((block_snr, q_rand, J_wrist))
            
    return local_scored_pool, attempts

def generate_wrist_dataset(model: hayati_model.HayatiModel, wrist_params: list, base_params: list, 
                                    num_points_needed=50, pool_size=10000, elite_pool_size=1000, search_iterations=5000):
    
    num_cores = cpu_count()
    print(f"\n[ЭТАП 1 & 2] Параллельная генерация и оценка (CPU Количеств: {num_cores})...")
    print(f"Цель: {pool_size} безопасных точек с оценкой градиентов.")
    
    points_per_core = pool_size // num_cores
    
    tasks = [(model, wrist_params, base_params, points_per_core) for _ in range(num_cores)]
    
    scored_pool = []
    total_attempts = 0
    
    # Запускаем параллельный процесс
    with Pool(processes=num_cores) as pool:
        results = pool.map(worker_generate_scored_points, tasks)
        
    # Собираем результаты из всех потоков
    for local_pool, attempts in results:
        scored_pool.extend(local_pool)
        total_attempts += attempts
        
    print(f"[+] Успешно сгенерировано и оценено {len(scored_pool)} точек (Попыток: {total_attempts}).")
    
    # Сортируем и оставляем только максимальный Block SNR
    scored_pool.sort(key=lambda x: x[0], reverse=True)
    elite_pool = scored_pool[:elite_pool_size]
    
    if len(elite_pool) == 0:
        print("🔴 КРИТИЧЕСКАЯ ОШИБКА: Не найдено ни одной безопасной точки!")
        return []
        
    avg_snr = np.mean([x[0] for x in elite_pool])
    print(f"[+] Оставлено {len(elite_pool)} элитных точек. Средний Block SNR: {avg_snr:.2f}")

    # --- СТУПЕНЬ 3: D-ОПТИМАЛЬНОСТЬ (CONDITION NUMBER) ---
    # Этот шаг выполняется в один поток, так как SVD для малых матриц происходит мгновенно
    print(f"\n[ЭТАП 3] Поиск оптимального подмножества из {num_points_needed} точек (SVD)...")
    
    elite_jacobians = np.array([x[2] for x in elite_pool])
    elite_qs = [x[1] for x in elite_pool]

    best_cond_num = float('inf')
    best_subset_indices = None
    
    for i in range(search_iterations):
        # Если элитный пул меньше, чем нужно точек - берем сколько есть
        sample_size = min(num_points_needed, len(elite_pool))
        subset_indices = sample(range(len(elite_pool)), sample_size)
        J_subset = elite_jacobians[subset_indices, :]
        
        _, s, _ = svd(J_subset)
        if s[-1] < 1e-9: 
            continue
            
        cond_num = s[0] / s[-1]
        
        if cond_num < best_cond_num:
            best_cond_num = cond_num
            best_subset_indices = subset_indices
            if i % 1000 == 0 or i < 100:
                print(f"   -> Итерация {i:<5}: Новое лучшее число обусловленности = {best_cond_num:.2f}")

    if best_subset_indices is None:
        print("🔴 КРИТИЧЕСКАЯ ОШИБКА: Якобиан полностью вырожден.")
        return []

    print(f"\n🟢 УСПЕХ! Финальное число обусловленности блока кисти: {best_cond_num:.2f}")
    if best_cond_num < 100:
        print("   -> Отличный результат! Параметры кисти разделятся идеально.")
    elif best_cond_num < 500:
        print("   -> Хороший результат. Модель должна стабильно сойтись.")
    else:
        print("   -> ⚠️ Внимание: число обусловленности велико. Ожидается кросс-корреляция.")

    final_dataset = []
    for idx in best_subset_indices:
        q_opt = elite_qs[idx]
        row = (q_opt / DEG).tolist() + ["WRIST_BLOCK", float(best_cond_num)]
        final_dataset.append(row)
        
    return final_dataset

def main(args):
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    model = hayati_model.HayatiModel(config)
    model.jac_DH_0 = calculate_jac_DH_params(model, model.zero_wire_angles)

    if args.mode == 'generation':
        generate_samples(model, args.target_index, args.start_joint - 1, args.tries_count, 
                            args.prev_eps, args.future_eps, args.angle_diff, args.len_diff, args.erase_previous) 

    elif args.mode == 'validation':
        validation_poses = generate_validation_dataset(model, num_points=args.tries_count, num_bins=args.num_bins)
        export_validation_csv(model, validation_poses, "datasets/ARM95/wire/validation_dataset.csv")
        #get_dataset_for_validation(model, args.tries_count, 15 * DEG, [100 / 1000, 600 / 1000], 15*DEG)
    elif args.mode == 'correct':
        correct_poses(model)
    elif args.mode == 'wrist_generation':
        dataset = generate_wrist_dataset(model, model.wrist_params, list(set(model.calibration_mask) - set(model.wrist_params)))
        write_dataset(dataset, model.wrist_dataset, ["q1", "q2", "q3", "q4", "q5", "q6", "type", "block_condition_number"], tolerance=3)
    else:
        print("Wrong mode")

    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Name of .json configuration file. Default: ARM95.json", default="src/config/ARM95_calibration.json")
    parser.add_argument("-m", "--mode", help="Selected mode: 'generation' or 'check'. Default: 'generation'", default="generation")
    parser.add_argument("-s_j", "--start_joint", help="1-6", type=int, default=1)
    parser.add_argument("-e", "--erase_previous", help="Erase previous generation result. Default: 'false'", default="true")
    parser.add_argument("-i", "--target_index", help="", type=str, default="theta_6")
    parser.add_argument("-n", "--tries_count", help="Number of tries. Default: 10", type=int, default=1000)
    parser.add_argument("-b", "--num_bins", help="Number of bins for validation dataset. Default: 15", type=int, default=15)
    parser.add_argument("-a", "--angle_diff", help="Angle difference between poses. Default: 5", type=float, default=5)
    parser.add_argument("-l", "--len_diff", help="Lenght difference between poses, mm. Default: 2", type=float, default=5)
    parser.add_argument("-p", "--prev_eps", help="Epsilon for previous params. Default: 0.005", type=float, default=100)
    parser.add_argument("-f", "--future_eps", help="Epsilon for future params. Default: 0.0004", type=float, default=100)

    args = parser.parse_args()
    main(args)
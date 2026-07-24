import numpy as np
from math import cos, sin, pi, sqrt, atan2, asin, log10, acos, copysign
from typing import Union
from pytest import param
from scipy.optimize import minimize_scalar, minimize, Bounds, least_squares
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import argparse
import random
import json
import time
import hayati_model
from math_routines import extract_zyx_euler
from random import uniform
from dataset_generation_absolute import generate_real_dh
from dataset_generation_wire import measure_all_distances, calculate_distance, calculate_contribution
from csv_routines import read_dataset, write_dataset
from multiprocessing import Pool, cpu_count
import copy
from math_routines import DEG
from graph import plot_side_by_side_hist, plot_dh_comparison, plot_all_6_axes, plot_calibration_errors_from_data
import itertools

EXTRA_POINTS = ["LINEAR", "JOINT", "JOINT_CONTINUE", "END"]    # -2, -1, -3

def step_by_step_base_calibration(model: hayati_model.HayatiModel, dataset=[], is_real=0, passes=1, polish=False):
    calibration_mask = np.zeros(len(dataset), dtype=bool)
    for i, pose in enumerate(dataset):
        if pose[6] in model.calibration_mask:
            calibration_mask[i] = True
    filtered_dataset = dataset[calibration_mask]
    
    for _ in range(passes):
        for i in range (len(model.base_params)):
            single_param_dataset = filtered_dataset[filtered_dataset[:, 6] == model.base_params[i]]

            param_num, param_letter_num = model.params_from_letters(model.base_params[i])
            if(len(single_param_dataset) == 0):
                continue        

            optimizing_param = calibrate_single_param(model, single_param_dataset)     
            
            final_value = optimizing_param.x[0]
            model.estimated_dh[param_num][param_letter_num] = final_value 

    # Глобальная полировка перенесена сюда
    if polish:
        run_global_polish(model, filtered_dataset[:, :8], limit_mm=5, limit_deg=5)

    return True

def calibrate_single_param(model: hayati_model.HayatiModel, dataset: Union[np.ndarray, list]): 
    param_index = int(model.base_params.index(dataset[0, 6]))
    param_num, param_letter_num = model.params_from_letters(model.base_params[param_index])
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

    #print(f"Optimizing param {model.base_params[param_index]}...")
    if (param_index in model.angle_idexes):
        bounds = Bounds(model.nominal_dh[param_num][param_letter_num] - 2 * DEG, 
                        model.nominal_dh[param_num][param_letter_num] + 2 * DEG) # [-2 * DEG, 2 * DEG]
    else:
        bounds = Bounds(model.nominal_dh[param_num][param_letter_num] - 2/1000, 
                        model.nominal_dh[param_num][param_letter_num] + 2/1000)

    res = minimize(loss_function, x0=model.nominal_dh[param_num][param_letter_num], 
                   method='SLSQP', tol=1e-16)#, bounds=bounds)
    
    # res = least_squares(
    #     loss_function, 
    #     x0=model.nominal_dh[param_num][param_letter_num], 
    #     method='lm', 
    #     x_scale='jac', 
    #     ftol=1e-10, 
    #     xtol=1e-10,
    #     verbose=0
    # )
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
    
    nominal_angles = extract_zyx_euler(np.array([model.get_transition_matrix(pose, "nominal")[0:3, 0:3] for pose in poses]))
    real_angles = extract_zyx_euler(np.array([model.get_transition_matrix(pose, "real")[0:3, 0:3] for pose in poses]))
    estimated_angles = extract_zyx_euler(np.array([model.get_transition_matrix(pose, "estimated")[0:3, 0:3] for pose in poses]))
    
    nominal_diff_for_print = np.concatenate((np.abs(real_data - nominal_data) * 1000, np.abs(real_angles - nominal_angles)/DEG), axis=1)
    estimated_diff_for_print = np.concatenate((np.abs(real_data - estimated_data) * 1000, np.abs(real_angles - estimated_angles)/DEG), axis=1)

    swap_data = model.estimated_dh[last_indexes[0]][last_indexes[1]]
    model.estimated_dh[last_indexes[0]][last_indexes[1]] = model.nominal_dh[last_indexes[0]][last_indexes[1]]
    estimated_data_prev = np.array([model.get_transition_matrix(pose, "estimated")[0:3, 3] for pose in poses])
    model.estimated_dh[last_indexes[0]][last_indexes[1]] = swap_data

    nominal_diff = np.abs(real_data - nominal_data) * 1000 # mm
    estimated_diff = np.abs(real_data - estimated_data) * 1000 # mm
    estimated_diff_prev = np.abs(real_data - estimated_data_prev) * 1000 # mm

    nominal_dist = np.linalg.norm(nominal_diff, axis=1).reshape(len(poses), 1)
    estimated_dist = np.linalg.norm(estimated_diff, axis=1).reshape(len(poses), 1)
    estimated_dist_prev = np.linalg.norm(estimated_diff_prev, axis=1).reshape(len(poses), 1)
    
    #delta_coords = (np.abs(real_data - nominal_data) - np.abs(real_data - estimated_data)) * 1000
    
    # nom_params_errors = model.real_dh - model.nominal_dh
    # est_params_errors = model.real_dh - model.estimated_dh

    #params_errors = (nom_params_errors - est_params_errors)

    avg_nom_error = np.sum(nominal_dist)/len(nominal_dist)
    avg_est_error = np.sum(estimated_dist) / len (estimated_dist)
    avg_est_error_prev = np.sum(estimated_dist_prev) / len (estimated_dist_prev)
    
    delta = nominal_dist - estimated_dist
    printable_res = np.concatenate((poses/DEG, nominal_dist, estimated_dist, delta, nominal_diff_for_print, estimated_diff_for_print), axis=1)
    
    avg_error = np.sum(delta) / len(delta)
    median = np.concatenate( [np.array([0, 0, 0, 0, 0, 0, avg_nom_error, avg_est_error, avg_error]).reshape(1, 9), np.mean(nominal_diff_for_print, axis=0).reshape(1, 6), np.mean(estimated_diff_for_print, axis=0).reshape(1, 6)], axis=1)
    printable_res = np.append(printable_res, median, axis=0)
    write_dataset(printable_res, "results/ARM95/test_result.csv", ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'd_nom', 'd_est', 'delta', 
                                                                   'x_nom', 'y_nom', 'z_nom', 'alpha_nom', 'beta_nom', 'gamma_nom',
                                                                   'x_est', 'y_est', 'z_est', 'alpha_est', 'beta_est', 'gamma_est'],
                     tolerance=3)
    return avg_nom_error, avg_est_error, avg_est_error_prev

def write_results(model: hayati_model.HayatiModel):
    # Получаем ВСЕ матрицы (включая базовую, чтобы было на что инвертировать первую)
    all_tfs = model.get_all_transition_matrixes([0, 0, 0, 0, 0, 0], 'estimated')
    
    # Наш рабочий срез для суставов (как у тебя и было)
    tfs = all_tfs[1:-1]
    
    mcx_params = []
    joint_offsets = []
    
    for index, tf in enumerate(tfs):
        # Предыдущая глобальная матрица в цепочке находится в all_tfs[index]
        T_prev = all_tfs[index]
        
        # Вычисляем ОТНОСИТЕЛЬНУЮ матрицу перехода между соседними звеньями
        tf_relative = np.linalg.inv(T_prev) @ tf
        
        # Вытаскиваем локальные смещения и повороты
        offset = tf_relative[:3, 3]
        rotation = extract_zyx_euler(tf_relative[:3, :3])
        
        # Записываем уже адекватные локальные параметры
        mcx_params.append([offset[0], offset[1], offset[2], 0, rotation[1], rotation[2]])
        joint_offsets.append(model.estimated_dh[index][3] - model.nominal_dh[index][3])

    res_dict = {"estimated_dh": model.estimated_dh.tolist(), "mcx_params": mcx_params, "offsets": joint_offsets}
    
    with open(model.results_file, 'w') as file:
        json_str = json.dumps(res_dict)
        file.write(json_str.replace("]], ", "]],\n"))

    nom_params = model.get_readable_params("nominal")
    est_params = model.get_readable_params("estimated")
    #print(nom_params[:, :4])
    #print()
    print(est_params[:, :4])

def validate_wire(model: hayati_model.HayatiModel, validation_dataset, is_real):
    res = np.zeros(shape=(validation_dataset.shape)) 
    for i, line in enumerate(validation_dataset[:-1]):
        if line[8] <= 0:
            continue
        distance_theoretical = np.linalg.norm(calculate_distance(model, line[0:6], "nominal"))
        distance_estimated = np.linalg.norm(calculate_distance(model, line[0:6], "estimated"))
        
        
        if is_real:
            res[i] = np.concatenate([line[0:6], [distance_theoretical, distance_estimated, 
                                                line[8], (abs(line[8] - distance_theoretical) - abs(line[8] - distance_estimated))]])
        
        else:    
            distance_real = np.linalg.norm(calculate_distance(model, line[0:6], "real"))
            res[i] = np.concatenate([line[0:6], [distance_theoretical, distance_estimated, 
                                                distance_real, (abs(distance_real - distance_theoretical) - abs(distance_real - distance_estimated))]])


    res[-1, 6:] = np.array([np.mean(abs(res[:-1, 6] - res[:-1, 8])), np.mean(abs(res[:-1, 7] - res[:-1, 8])), np.mean(res[:-1, 8]), np.mean(res[:-1, 9])])
    res[-1, :6] = np.array([-1, -1, -1, -1, -1, -1]) * DEG
    #print((res[:-1, 7] - res[:-1, 8])* 1000)
    return res

def write_validation_dataset(model: hayati_model.HayatiModel, is_real=False, source_path=""):
    if source_path == "":
        if is_real:
            source_path = "results/ARM95/validation_results_real.csv"
        else:
            source_path = "results/ARM95/validation_results_sim.csv"

    #source_path = "results/ARM95/validation_results_calib.csv"

    validation_dataset = read_dataset(source_path, model.fieldnames_options["test_result"])
    validation_dataset[:, :6] *= DEG
    validation_dataset[:, 6:] /= 1000

    output_dataset = validate_wire(model, validation_dataset, is_real)
    output_dataset[:, :6] /= DEG
    output_dataset[:, 6:] *= 1000

    write_dataset(output_dataset, source_path, model.fieldnames_options["test_result"], tolerance = 3)
    #print("Done")

def _calibrate_list_of_dh(model, i_target, dataset, dh_list, passes, polish):
    copy_model = copy.deepcopy(model)
    height = copy_model.estimated_dh.shape[0]

    num_iters = int(len(dh_list) / height)
    # Возвращаем полную матрицу ошибок для вычисления медианы в главном потоке
    errors_matrix = np.zeros((num_iters, len(copy_model.error_list)))
    
    for i in range(num_iters):
        copy_model.reset_estimated_dh()
        copy_model.change_real_dh(dh_list[i * height: (i+1) * height, :])
        current_dataset = measure_all_distances(copy_model, copy.deepcopy(dataset))
        
        step_by_step_base_calibration(copy_model, dataset=current_dataset, passes=passes, polish=polish)
        
        est_params = np.array(copy_model.get_readable_params("estimated"))
        real_params = np.array(copy_model.get_readable_params("real"))
        nominal_params = np.array(copy_model.get_readable_params("nominal"))
        
        for j, param_name in enumerate(copy_model.error_list):
            p_num, p_let = copy_model.params_from_letters(param_name)
            final_error = est_params[p_num][p_let] - real_params[p_num][p_let]
            start_error = nominal_params[p_num][p_let] - real_params[p_num][p_let]
            errors_matrix[i, j] = abs(start_error) - abs(final_error)

    return errors_matrix

def _rate_batch_worker(model, dataset_eval, i_target, num_tries_in_batch, dh_array, passes, polish):
    copy_model = copy.deepcopy(model)
    new_error_list = []
    height = copy_model.estimated_dh.shape[0]
    
    p_num, p_let = copy_model.params_from_letters(copy_model.error_list[i_target])
    
    for i in range(num_tries_in_batch):
        copy_model.reset_estimated_dh()
        copy_model.change_real_dh(dh_array[i * height: (i+1) * height, :])
        
        dataset_curr = measure_all_distances(copy_model, copy.deepcopy(dataset_eval))
        
        step_by_step_base_calibration(copy_model, dataset=dataset_curr, passes=passes, polish=polish)
        
        est_params = np.array(copy_model.get_readable_params("estimated"))
        real_params = np.array(copy_model.get_readable_params("real"))
        nominal_params = np.array(copy_model.get_readable_params("nominal"))

        final_error = est_params[p_num][p_let] - real_params[p_num][p_let]
        start_error = nominal_params[p_num][p_let] - real_params[p_num][p_let]
        
        new_error_list.append(abs(start_error) - abs(final_error))
            
    return new_error_list

def rate_poses_contributions(model, target_param, num_tries, floor, passes, polish):
    dataset_orig = read_dataset(model.contribution_dataset, model.fieldnames_options["wire_contributions"])
    dataset_orig[:, 0:6] *= DEG
    
    mask = np.array([val not in EXTRA_POINTS for val in dataset_orig[:, 6]])
    dataset_orig = dataset_orig[mask]

    i_target = model.error_list.index(target_param)
    valid_params = model.error_list[:i_target + 1]
    time_mask = np.array([val in valid_params for val in dataset_orig[:, 6]])
    dataset_orig = dataset_orig[time_mask]

    target_mask = dataset_orig[:, 6] == target_param
    target_poses = dataset_orig[target_mask]
    other_poses = dataset_orig[~target_mask]
    
    num_cores = cpu_count()
    chunk_size = num_tries // num_cores
    remainder = num_tries % num_cores
    
    is_changed = 1
    loop_completed = 1
    
    print(f"Starting analysis of {len(target_poses)} poses across {num_tries} tries...")

    while is_changed == 1:
        if loop_completed: 
            is_changed = 0
            height = model.estimated_dh.shape[0]
            
            dh_array = np.zeros(shape=(num_tries * height, model.estimated_dh.shape[1]))
            for i in range(num_tries):
                dh_array[i * height: (i+1) * height, :] = generate_real_dh(model)
                
            dataset_full = np.vstack([target_poses, other_poses]) if len(target_poses) > 0 else other_poses

            tasks = []
            start_idx = 0
            for i in range(num_cores):
                current_batch_size = chunk_size + (1 if i


< remainder else 0)
                if current_batch_size > 0:
                    end_idx = start_idx + current_batch_size
                    tasks.append((model, i_target, dataset_full,
                                  dh_array[start_idx * height : end_idx * height, :], passes, polish))
                    start_idx = end_idx
                    
            with Pool(processes=num_cores) as pool1:
                results = pool1.starmap(_calibrate_list_of_dh, tasks)

            # Собираем полную матрицу (1000 x num_params) и берем МЕДИАНУ
            all_initial_errors = np.vstack(results)
            initial_errors_list = np.median(all_initial_errors, axis=0)
            
            initial_error = initial_errors_list[i_target] if len(target_poses) > 0 else 0.0
            print(f"Initial error {initial_error:.3f}")

        with Pool(processes=num_cores) as pool:
            loop_completed = 0
            for pose_num in range(len(target_poses) - 1, -1, -1):
                temp_target = np.delete(target_poses, pose_num, axis=0)
                dataset_eval = np.vstack([temp_target, other_poses]) if len(temp_target) > 0 else other_poses

                tasks = []
                start_idx = 0
                for i in range(num_cores):
                    current_batch_size = chunk_size + (1 if i < remainder else 0)
                    if current_batch_size > 0:
                        end_idx = start_idx + current_batch_size
                        tasks.append((model, dataset_eval, i_target, current_batch_size,
                                      dh_array[start_idx * height : end_idx * height, :], passes, polish))
                        start_idx = end_idx

                results = pool.starmap(_rate_batch_worker, tasks)
                new_error_vector = np.concatenate(results)
                # Берем строгую МЕДИАНУ
                new_median = np.median(new_error_vector) 

                print(f"New value {new_median:.3f}")
                
                if (new_median - initial_error) > floor:
                    target_poses = temp_target
                    is_changed = 1
                    initial_error = new_median
                    print(f"Deleted pose index {pose_num}")
                else:
                    print(f"Saved pose index {pose_num}")
            else:
                loop_completed = 1
            
    target_poses[:, 0:6] /= DEG
    print(f"Found {len(target_poses)} poses with positive contribution")
    print(f"Final error {initial_error:.3f}")
    
    write_dataset(
        target_poses.tolist() if hasattr(target_poses, 'tolist') else target_poses, 
        "datasets/ARM95/wire/generation/positive_contributions.csv", 
        model.fieldnames_options["wire_contributions"], 
        tolerance=3
    )
    
    return target_poses

def cluster_and_visualize_poses(model: hayati_model.HayatiModel, target_param, n_clusters=5):
    """
    Разбивает массив точек на кластеры, строит PCA-проекцию и выводит 
    исходный номер точки в датасете при наведении мыши.
    """
    # 1. Чтение данных
    raw_data = read_dataset(model.contribution_dataset, model.fieldnames_options["wire_contributions"])
    # if not raw_data:
    #     print("Датасет пуст.")
    #     return None, None
        
    dataset_orig = np.array(raw_data, dtype=object)
    
    # Создаем массив исходных индексов (номера строк изначального датасета)
    original_indices = np.arange(len(dataset_orig))
    
    # Фильтрация битых строк
    mask = np.array([isinstance(val, str) for val in dataset_orig[:, 6]])
    dataset_orig = dataset_orig[mask]
    original_indices = original_indices[mask]
    
    # Фильтрация по целевому параметру
    poses_mask = dataset_orig[:, 6] == target_param
    target_poses = dataset_orig[poses_mask]
    final_original_indices = original_indices[poses_mask]
    
    if len(target_poses) == 0:
        print(f"Нет точек для параметра {target_param}.")
        return None, None

    print(f"Найдено {len(target_poses)} точек для {target_param}. Выполняю кластеризацию...")

    # Извлекаем только 6 углов суставов
    angles = target_poses[:, 0:6].astype(float)
    
    # 2. Нормализация (чтобы все суставы имели одинаковый вес для K-Means)
    scaler = StandardScaler()
    angles_scaled = scaler.fit_transform(angles)
    
    # 3. Кластеризация K-Means
    kmeans = KMeans(n_clusters=min(n_clusters, len(angles)), random_state=42, n_init=10)
    labels = kmeans.fit_predict(angles_scaled)
    
    # 4. Сжатие 6D пространства в 2D для визуализации (PCA)
    pca = PCA(n_components=2)
    angles_2d = pca.fit_transform(angles_scaled)
    
    # --- ВИЗУАЛИЗАЦИЯ ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # График 1: Карта точек
    scatter = ax1.scatter(angles_2d[:, 0], angles_2d[:, 1], c=labels, cmap='tab10', s=60, alpha=0.8, edgecolors='k')
    ax1.set_title(f'PCA: Проекция множества точек ({target_param})', fontsize=12)
    ax1.set_xlabel('Главная компонента 1')
    ax1.set_ylabel('Главная компонента 2')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Легенда
    legend1 = ax1.legend(*scatter.legend_elements(), title="Кластеры")
    ax1.add_artist(legend1)

    # ИНТЕРАКТИВНОСТЬ: Всплывающие подсказки
    try:
        import mplcursors
        # hover=True означает появление при наведении
        cursor = mplcursors.cursor(scatter, hover=True)
        
        @cursor.connect("add")
        def on_add(sel):
            idx = sel.index
            orig_idx = final_original_indices[idx]
            cluster_id = labels[idx]
            # Форматируем текст подсказки
            sel.annotation.set_text(f"Строка в CSV: {orig_idx + 2}\nКластер: {cluster_id}")
            sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)
    except ImportError:
        print("⚠️ Установите библиотеку 'mplcursors' (pip install mplcursors) для работы всплывающих подсказок.")

    # График 2: Профили кластеров
    centers_real = scaler.inverse_transform(kmeans.cluster_centers_)
    
    for i, center in enumerate(centers_real):
        ax2.plot(range(1, 7), center, marker='o', linewidth=2, label=f'Кластер {i}')
        
    ax2.set_title('Усредненные позы для каждого кластера', fontsize=12)
    ax2.set_xlabel('Номер сустава (1-6)')
    ax2.set_ylabel('Угол (в радианах или градусах, как в датасете)')
    ax2.set_xticks(range(1, 7))
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.show()
    
    return target_poses, labels

def calculate_estimated_distance(model, angles, zero_pos):
    """Вычисляет предсказанное вытягивание троса (в метрах)"""
    matrix = model.get_transition_matrix(angles, "estimated")
    pos = matrix[0:3, 3]
    distance = sqrt((pos[0] - zero_pos[0])**2 + (pos[1] - zero_pos[1])**2 + (pos[2] - zero_pos[2])**2)
    return distance

def run_global_polish(model: hayati_model.HayatiModel, dataset: np.ndarray, limit_mm=0.5, limit_deg=0.3):
    # Фильтруем битые строки и переводим углы в радианы
    #mask = np.array([isinstance(val, str) for val in dataset[:, 6]])
    #dataset = dataset[mask]
    #dataset[:, 0:6] = dataset[:, 0:6].astype(float) * DEG
    
    # Расстояния в датасете обычно в мм. Переводим в метры для внутренних расчетов
    #real_distances_m = dataset[:, 7].astype(float) / 1000.0
    poses_rad = dataset[:, 0:6].astype(float)
    real_distances_m = dataset[:, 7].astype(float)
    
    # print(f"Количество калибровочных точек: {len(poses_rad)}")
    # print(f"Количество полируемых параметров: {len(params_to_tune)}")

    params_to_tune = model.calibration_mask
    
    x0 = []
    lower_bounds = []
    upper_bounds = []
    param_indices = []
    
    # Настраиваем границы (Bounds): +- 0.5 мм и +- 0.3 градуса
    LIMIT_MM = limit_mm / 1000.0 # 0.5 мм в метрах
    LIMIT_RAD = limit_deg * DEG   # 0.3 градуса в радианах
    
    for param_name in params_to_tune:
        p_num, p_let = model.params_from_letters(param_name)
        current_val = model.estimated_dh[p_num][p_let]
        
        x0.append(current_val)
        param_indices.append((p_num, p_let))
        
        # Определяем тип параметра для правильных границ
        if 'alpha' in param_name or 'theta' in param_name or 'beta' in param_name:
            lower_bounds.append(current_val - LIMIT_RAD)
            upper_bounds.append(current_val + LIMIT_RAD)
        else:
            lower_bounds.append(current_val - LIMIT_MM)
            upper_bounds.append(current_val + LIMIT_MM)
            
    x0 = np.array(x0)

    def objective_function(x):
        # 1. Обновляем модель
        for i, (p_num, p_let) in enumerate(param_indices):
            model.estimated_dh[p_num][p_let] = x[i]
            
        # 2. КРИТИЧНО: Пересчитываем нулевую позицию фланца!
        zero_pos = model.get_transition_matrix(model.zero_wire_angles, "estimated")[0:3, 3]
        
        # 3. Считаем невязки
        residuals = np.zeros(len(poses_rad))
        for i, angles in enumerate(poses_rad):
            d_theor = calculate_estimated_distance(model, angles, zero_pos)
            residuals[i] = d_theor - real_distances_m[i]
            
        return residuals

    #print("Запуск оптимизатора Trust Region Reflective (trf) с жесткими границами...")
    res = least_squares(
        objective_function, 
        x0=x0, 
        bounds=(lower_bounds, upper_bounds),
        method='trf', # Метод 'trf' поддерживает границы (bounds), в отличие от 'lm'
        x_scale='jac', 
        ftol=1e-8, 
        xtol=1e-8,
        verbose=0
    )
    
    # Фиксируем финальные отполированные значения
    for i, (p_num, p_let) in enumerate(param_indices):
        model.estimated_dh[p_num][p_let] = res.x[i]
        
    final_rmse_mm = np.sqrt(np.mean(res.fun**2)) * 1000.0
    #print(f"Полировка завершена успешно! Финальная RMSE на калибровке: {final_rmse_mm:.4f} мм")

def wrist_block_calibration(model: hayati_model.HayatiModel, dataset: Union[np.ndarray, list]): 
    """
    SLSQP оптимизация блока параметров кисти одновременно.
    """
    # Автоматически выбираем из error_list только параметры 4, 5, 6 звеньев
    wrist_params = model.wrist_params
    
    param_indices = []
    x0 = []
    bounds_min = []
    bounds_max = []
    
    # Формируем стартовые точки и границы
    for param_name in wrist_params:
        p_num, p_let = model.params_from_letters(param_name)
        param_indices.append((p_num, p_let))
        
        nominal_val = model.nominal_dh[p_num][p_let]
        x0.append(nominal_val)
        
        p_index = model.error_list.index(param_name)
        if p_index in model.angle_idexes:
            bounds_min.append(nominal_val - 2 * DEG)
            bounds_max.append(nominal_val + 2 * DEG)
        else:
            bounds_min.append(nominal_val - 2/1000)
            bounds_max.append(nominal_val + 2/1000)
            
    real_distances = dataset[:, 7]

    def loss_function(optimizing_params):
        # 1. Применяем параметры кисти к модели
        for i, (p_num, p_let) in enumerate(param_indices):
            model.estimated_dh[p_num][p_let] = optimizing_params[i]

        # 2. Вычисляем нулевую точку
        zero_pos = model.get_transition_matrix(model.zero_wire_angles, "estimated")[0:3, 3]
        
        # 3. Вычисляем ошибку
        d_theor_list = np.zeros(len(dataset))
        for i, row in enumerate(dataset):
            angles = row[0:6]
            d_theor_list[i] = calculate_estimated_distance(model, angles, zero_pos)
        
        diff = d_theor_list - real_distances
        error = np.sum(diff**2)
        
        return error

    bounds = Bounds(bounds_min, bounds_max)

    def objective_function(x):
        # 1. Обновляем модель текущими параметрами оптимизатора
        for i, (p_num, p_let) in enumerate(param_indices):
            model.estimated_dh[p_num][p_let] = x[i]

        # 2. Пересчитываем нулевую позицию датчика с новыми параметрами
        zero_pos = model.get_transition_matrix(model.zero_wire_angles, "estimated")[0:3, 3]
        
        # 3. Вычисляем невязки для всего датасета
        residuals = np.zeros(len(dataset))
        for i, angles in enumerate(dataset[:, 0:6]):
            d_theor = calculate_estimated_distance(model, angles, zero_pos)
            residuals[i] = d_theor - real_distances[i]
            
        return residuals

    # Левенберг-Марквардт с автоматическим масштабированием (jac)
    res = least_squares(
        objective_function, 
        x0=x0, 
        method='lm', 
        x_scale='jac', 
        ftol=1e-8, 
        xtol=1e-8,
        verbose=0
    )

    #res = minimize(loss_function, x0=x0, method='SLSQP', tol=1e-16, bounds=bounds)
    
    # Окончательно фиксируем найденные значения
    for i, (p_num, p_let) in enumerate(param_indices):
        model.estimated_dh[p_num][p_let] = res.x[i]
    return res

def _calibrate_batch(model: hayati_model.HayatiModel, base_dataset: np.ndarray, 
                            tries_count: int, generate: bool, passes: int, polish: bool):
    copy_model = copy.deepcopy(model)
    
    param_count = len(copy_model.error_list)
    local_param_errors = np.zeros(param_count)
    local_nom_param_errors = np.zeros(param_count)
    local_est_param_errors = np.zeros(param_count)

    local_nominal_error = 0
    local_estimated_error = 0
    local_estimated_error_prev = 0

    total_nominal_error = 0.0
    total_estimated_error = 0.0
    total_estimated_error_prev = 0.0

    for _ in range(tries_count):
        copy_model.reset_estimated_dh()
        
        if generate:
            new_dh = generate_real_dh(copy_model)
            copy_model.change_real_dh(new_dh)
            
        curr_base_dataset = measure_all_distances(copy_model, copy.deepcopy(base_dataset))
        
        # === УНИФИЦИРОВАННЫЙ ВЫЗОВ ===
        # Вся логика passes и polish теперь спрятана внутри функции, как и в других местах
        step_by_step_base_calibration(copy_model, dataset=curr_base_dataset, is_real=0, passes=passes, polish=polish) 
        last_indexes = [0, 0]
        
        new_nom, new_est, new_esp_prev = check_results(copy_model, last_indexes)
        local_nominal_error += new_nom
        local_estimated_error += new_est
        local_estimated_error_prev += new_esp_prev       
        
        total_nominal_error += new_nom
        total_estimated_error += new_est
        total_estimated_error_prev += new_esp_prev
        
        nom_params = copy_model.get_readable_params("nominal")
        real_params = copy_model.get_readable_params("real")
        est_params = copy_model.get_readable_params("estimated")
        
        for i in range(param_count):
            param_num, param_letter_num = copy_model.params_from_letters(copy_model.error_list[i])
            
            nominal_value = nom_params[param_num][param_letter_num]
            real_value = real_params[param_num][param_letter_num]
            final_value = est_params[param_num][param_letter_num]
            
            start_error = nominal_value - real_value
            final_error = final_value - real_value

            if real_value != 0:         
                start_error_rel = start_error / real_value
                final_error_rel = final_error / real_value
            else:
                start_error_rel = 100
                final_error_rel = 100

            local_nom_param_errors[i] += start_error
            local_est_param_errors[i] += final_error
            
            if start_error_rel != 0:
                local_param_errors[i] += (abs(start_error) - abs(final_error))
                
    return (
        total_nominal_error / tries_count,
        total_estimated_error / tries_count,
        total_estimated_error_prev / tries_count,
        local_param_errors / tries_count,
        local_nom_param_errors / tries_count,
        local_est_param_errors / tries_count
    )

def make_many_calibration_attempts(model: hayati_model.HayatiModel, tries_num: int, generate: bool, passes: int, polish: bool, draw: bool = False):
    num_cores = cpu_count()
    chunk_size = max(1, (tries_num // num_cores))

    print("Загрузка датасетов...")
    # 1. Загружаем датасет для кисти
    # wrist_dataset = read_dataset(model.wrist_dataset, ["q1", "q2", "q3", "q4", "q5", "q6", "type", "block_condition_number"])
    # wrist_dataset = np.array(wrist_dataset, dtype=object)
    # wrist_dataset[:, 0:6] = wrist_dataset[:, 0:6].astype(float) * DEG
    # mask = np.array([val not in EXTRA_POINTS for val in wrist_dataset[:, 6]])
    # wrist_dataset = wrist_dataset[mask])
    full_dataset = read_dataset(model.contribution_dataset, model.fieldnames_options["wire_contributions"])
    full_dataset = np.array(full_dataset, dtype=object)
    full_dataset[:, 0:6] = full_dataset[:, 0:6].astype(float) * DEG
    full_dataset = measure_all_distances(model, full_dataset)
    mask = np.array([val not in EXTRA_POINTS for val in full_dataset[:, 6]])
    full_dataset = full_dataset[mask]

    wrist_mask = np.array([True if elem[6] == "WRIST_BLOCK" else False for elem in full_dataset])
    #wrist_dataset = full_dataset[wrist_mask]

    base_mask = np.array([not elem for elem in wrist_mask])
    base_dataset = full_dataset[base_mask]

    tasks = []
    for _ in range(min(num_cores, tries_num)):
        if chunk_size > 0:
            tasks.append((model, base_dataset, chunk_size, generate, passes, polish)) #wrist_dataset   
  
    print(f"Используется циклов (passes): {passes}")
    
    with Pool(processes=num_cores) as pool:
        results = pool.starmap(_calibrate_batch, tasks)

    # Распаковка результатов
    nominal_error = [x[0] for x in results]
    estimated_error = [x[1] for x in results]
    estimated_error_prev = [x[2] for x in results]
    params_error = np.array([x[3] for x in results])

    nominal_error_median = sum(nominal_error) / len(nominal_error)
    estimated_error_median = sum(estimated_error) / len(estimated_error)
    estimated_error_prev_median = sum(estimated_error_prev) / len(estimated_error_prev)
    params_error_median = np.sum(params_error, axis=0) / len(params_error)

    print("\n" + "="*50)
    print(" РЕЗУЛЬТАТЫ ГИБРИДНОЙ КАЛИБРОВКИ (в МИЛЛИМЕТРАХ)")
    print("="*50)
    print(f"Nominal error (до калибровки):     {nominal_error_median:.4f} мм")
    print(f"After calibration (после):         {estimated_error_median:.4f} мм")
    print(f"Difference (Улучшение троса):      {(nominal_error_median - estimated_error_median):.4f} мм")
    print("-" * 50)

    print("\n--- Улучшение по каждому параметру ---")
    for i, value in enumerate(params_error_median):
        print(f"{model.error_list[i]:>8}: {value:+.4f}")

    mask_positive = params_error_median > 0
    mask_negative = params_error_median < 0

    norm_pos = np.linalg.norm(params_error_median[mask_positive]) if np.any(mask_positive) else 0.0
    norm_neg = np.linalg.norm(params_error_median[mask_negative]) if np.any(mask_negative) else 0.0
    
    score = (norm_pos - norm_neg) * 100
    print(f"\nИтоговый Score (Эффективность параметров): {score:.3f}")

    if draw:
        plot_side_by_side_hist(data_before, data_after)
        plot_dh_comparison(np.abs(params_nom_error_median), np.abs(params_est_error_median))
        plot_all_6_axes(data[:, 9:15], data[:, 15:])

def main(args):      
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    model = hayati_model.HayatiModel(config)
    if args.mode == "calibration":
        if args.is_real:
            dataset = read_dataset(model.calibration_dataset, model.fieldnames_options["wire_samples"]) 
            dataset[:, 0:6] *= DEG
            dataset[:, 7] /= 1000
            step_by_step_base_calibration(model, dataset=dataset)
            write_results(model)
            write_validation_dataset(model, args.is_real)
            write_validation_dataset(model, args.is_real, source_path="results/ARM95/validation_results_calib.csv")
            data = read_dataset("results/ARM95/validation_results_real.csv", ['q1','q2','q3','q4','q5','q6','d_nom','d_est','d_encoder','diff'])
            data = data[:-1, :]
            if args.draw:
                plot_calibration_errors_from_data(data)
        else:
            make_many_calibration_attempts(model, args.num_of_tries, args.generate,  args.passes, args.polish, args.draw)  # ,params_error
            
    elif args.mode == "rate":
        rate_poses_contributions(model, args.target_param, args.num_of_tries, args.floor, args.passes, args.polish)
    elif args.mode == "cluster":
        cluster_and_visualize_poses(model, args.target_param, n_clusters=args.num_of_tries)

    elif args.mode == "test":
        validation_data = read_dataset("results/ARM95/validation_results.csv", ['q1','q2','q3','q4','q5','q6','d_nom','d_est','d_encoder','diff'])
        calibration_data = read_dataset("datasets/ARM95/wire/calibration_dataset_full_29.05.csv", ['q1','q2','q3','q4','q5','q6','index','d'])
        
        validation_data[:, :6] *= DEG
        validation_data[:, 6:] /= 1000

        calibration_data[:, 0:6] *= DEG
        calibration_data[:, 7] /= 1000
        
        combination, score = find_best_combination(model, calibration_data, validation_data)
        print(f"Best combination: {combination}\n Score: {score:.3f}")
    
    #print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Name of .json configuration file. Default: src/config/ARM95_calibration.json", default="src/config/ARM95_calibration.json")
    parser.add_argument("-m", "--mode", help="calibration or rate", default="calibration")
    parser.add_argument("--generate", action='store_true')
    parser.add_argument("--is_real", action='store_true')
    parser.add_argument("--draw", action='store_true')
    parser.add_argument("-n", "--num_of_tries", help="How many tries to make. Default: 1", type=int, default=1)
    parser.add_argument("-i", "--i_target", help="", type=int, default=10)
    parser.add_argument("-f", "--floor", help="", type=float, default=0.001)
    parser.add_argument("-t", "--target_param", help="", type=str, default="a_6")
    parser.add_argument("-p", "--passes", help="", type=int, default=1)
    parser.add_argument("--polish", action='store_true')

    args = parser.parse_args()
    main(args)
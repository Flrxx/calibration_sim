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
from dataset_generation_wire import calculate_wire_direction

def get_expected_deltas(model: hayati_model.HayatiModel):
    """
    Создает вектор ожидаемых амплитуд ошибок для нормализации вектора влияния.
    Углы в градусах, длины в миллиметрах.
    """
    deltas = np.zeros(len(model.error_list))
    for i, param_name in enumerate(model.error_list):
        if 'alpha' in param_name or 'theta' in param_name or 'beta' in param_name:
            deltas[i] = model.theta_delta / DEG  # Ожидаемая ошибка в градусах
        elif 'a_' in param_name or 'd_' in param_name:
            deltas[i] = model.a_delta * 1000 # Ожидаемая ошибка в мм
    return deltas

def calculate_expected_influence_mm(model, angles):
    base_contribution = calculate_contribution(model, angles)
    deltas = get_expected_deltas(model)
    return base_contribution * deltas

def rate_pose_for_parameter(model, angles, target_param, already_calibrated_params):
    influence_mm = calculate_expected_influence_mm(model, angles)
    target_idx = model.error_list.index(target_param)
    
    target_signal = abs(influence_mm[target_idx])
    if target_signal < 0.1:
        return 0.0

    noise_sq = 0.0
    for i, p_name in enumerate(model.error_list):
        if i == target_idx:
            continue
            
        influence_sq = influence_mm[i]**2
        if p_name in already_calibrated_params:
            noise_sq += influence_sq * 0.1 
        else:
            noise_sq += influence_sq * 1.0
            
    total_noise = sqrt(noise_sq)
    snr = target_signal / (total_noise + 1e-6)
    return snr

def is_pose_valid(model, angles):
    matrix = model.get_transition_matrix(angles, "nominal")
    z_dir = matrix[0:3, 2]
    wire = matrix[0:3, 3] - model.zero_offset_nominal
    wire_len = np.linalg.norm(wire) 
    
    if wire_len < model.wire_limits[0] or wire_len > model.wire_limits[1]:
        return False

    wire_norm = wire / wire_len
    scalar_wire_forward = np.vdot(wire_norm, model.zero_wire_direction)
    if scalar_wire_forward <= 0:
        return False
        
    alpha = acos(np.clip(scalar_wire_forward, -1.0, 1.0))
    if alpha > model.angle_limit_wire:
        return False

    scalar_z = np.vdot(-wire_norm, z_dir)
    alpha_z = acos(np.clip(scalar_z, -1.0, 1.0))
    if alpha_z > model.angle_limit_z:
        return False
        
    return True

def find_best_poses_monte_carlo(model, target_param, already_calibrated_params, num_candidates=10000, top_k=50, min_distance_rad=0.5):
    valid_poses = []
    
    for _ in range(num_candidates):
        q_rand = np.random.uniform(low=-np.pi, high=np.pi, size=6) 
        if not is_pose_valid(model, q_rand):
            continue
            
        snr = rate_pose_for_parameter(model, q_rand, target_param, already_calibrated_params)
        if snr > 0.5: 
            valid_poses.append((snr, q_rand))
            
    valid_poses.sort(key=lambda x: x[0], reverse=True)
    
    best_poses = []
    for snr, pose in valid_poses:
        if not best_poses:
            best_poses.append((snr, pose))
            continue
            
        is_far_enough = True
        for _, selected_pose in best_poses:
            distance = np.linalg.norm(pose - selected_pose)
            if distance < min_distance_rad:
                is_far_enough = False
                break 
                
        if is_far_enough:
            best_poses.append((snr, pose))
            
        if len(best_poses) >= top_k:
            break
            
    return best_poses # Возвращаем список кортежей: [(snr, angles), ...]

# ==========================================
# НОВАЯ ФУНКЦИЯ ДЛЯ ГЕНЕРАЦИИ И ЗАПИСИ В CSV
# ==========================================
def generate_samples_monte_carlo(model, target_param, already_calibrated_params, 
                                 num_candidates=10000, top_k=50, min_distance_rad=0.5):
    """
    Генерирует лучшие позы для параметра и сохраняет их в CSV формат.
    Является заменой старой функции generate_samples.
    """
    # 1. Получаем лучшие позы с их оценками SNR
    best_poses_with_snr = find_best_poses_monte_carlo(
        model, target_param, already_calibrated_params, 
        num_candidates, top_k, min_distance_rad
    )
    
    printable_res = []
    for snr, angles in best_poses_with_snr:
        # 2. Вычисляем вектор влияния для записи в датасет
        contribution = calculate_contribution(model, angles)
        
        # 3. Формируем строку датасета. 
        # Используем списки Python, чтобы строка (target_param) не превратила числа в строки
        row = (angles / DEG).tolist() + [target_param, snr] + contribution.flatten().tolist()
        printable_res.append(row)
        
    # 4. Сортируем результаты по колонке obj (индекс 7). 
    # ВАЖНО: Так как теперь это SNR, мы ищем максимальное значение (reverse=True)
    printable_res.sort(key=lambda x: x[7], reverse=True)
    
    # 5. Записываем в файл
    # ПРИМЕЧАНИЕ: Если ты вызываешь эту функцию в цикле для каждого параметра, 
    # убедись, что метод write_dataset дописывает файл ('a'), а не перезаписывает его ('w').
    # write_dataset(printable_res, model.generation_output, model.fieldnames_options["wire_contributions"], tolerance=3)
    
    print(f"Left {len(printable_res)} poses for {target_param}")
    return printable_res

# --- ОРИГИНАЛЬНАЯ ФУНКЦИЯ ДЛЯ ВЫЗОВА В calculate_expected_influence_mm ---
def calculate_contribution(model: hayati_model.HayatiModel, angles: Union[list, np.ndarray]):
    jac_DH = calculate_jac_DH_params(model, angles)
    wire_direction = calculate_wire_direction(model, angles)
    contribution = np.dot(wire_direction.T, jac_DH - model.jac_DH_0)
    contribution[0, model.angle_idexes] *= DEG * 1000 # mm/deg
    return contribution.flatten()

# def calculate_contribution(model: hayati_model.HayatiModel, angles: Union[list, np.ndarray]):
#     jac_DH = calculate_jac_DH_params(model, angles)
#     wire_direction = calculate_wire_direction(model, angles)
    
#     # Чистая математическая проекция Якобиана на линию троса
#     base_contribution = np.dot(wire_direction.T, jac_DH - model.jac_DH_0).flatten()
    
#     # Формируем вектор реальных амплитуд (ожидаемых ошибок)
#     deltas = np.zeros(len(model.error_list))
    
#     for i, param_name in enumerate(model.error_list):
#         if param_name.startswith('a_'):
#             deltas[i] = model.a_delta * 1000
#         elif param_name.startswith('d_'):
#             deltas[i] = model.d_delta * 1000
#         elif param_name.startswith('alpha_'):
#             deltas[i] = model.alpha_delta / DEG * DEG * 1000
#         elif param_name.startswith('theta_'):
#             deltas[i] = model.theta_delta / DEG * DEG * 1000
#         elif param_name.startswith('beta_'):
#             deltas[i] = model.beta_delta / DEG * DEG * 1000
#         else:
#             deltas[i] = 1.0
            
#     # Перемножаем градиент на допуск (мм/мм * мм ИЛИ м/рад * рад * 1000)
#     # Итог: физическое влияние каждого параметра в миллиметрах
#     return base_contribution * deltas

def calculate_jac_DH_params(model, angles: Union[list, np.ndarray], eps = 1e-6):
    jac_DH = np.zeros((3, len(model.error_list)))
    initial_coords = model.get_transition_matrix(angles, "nominal")[0:3, 3]
    for i, e in enumerate(model.error_list):        
        model.vary_nominal_dh(e, eps)                                     
        new_coords = model.get_transition_matrix(angles, "nominal")[0:3, 3]
        for j in range(3):
            jac_DH[j, i] = (new_coords[j] - initial_coords[j]) / eps
        model.vary_nominal_dh(e, -eps)                                    
    return jac_DH

def generate_wrist_block_dataset(model, num_poses=50, min_distance_rad=0.3):
    """
    Генерирует датасет исключительно для Блока Кисти (суставы 4, 5, 6).
    Суставы 1, 2, 3 жестко заморожены в zero_wire_angles.
    """
    print("\n--- Сбор данных: Блок Кисти (суставы 4, 5, 6) ---")
    valid_poses = []
    attempts = 0
    all_results = []
    
    while len(valid_poses) < num_poses and attempts < 100000:
        attempts += 1
        
        # Начинаем с замороженной нулевой позы
        q_rand = np.copy(model.zero_wire_angles)
        
        # Размораживаем 4, 5 и 6 суставы (индексы 3, 4, 5)
        for j in range(3, 6):
            q_rand[j] = np.random.uniform(low=-np.pi, high=np.pi)
            
        # Проверяем физические лимиты (strict=True, чтобы угол не превышал 75 град)
        if not is_pose_valid(model, q_rand):
            continue
            
        # Защита от слишком близких/похожих точек
        is_far_enough = True
        for selected_pose in valid_poses:
            if np.linalg.norm(q_rand - selected_pose) < min_distance_rad:
                is_far_enough = False
                break
                
        if is_far_enough:
            valid_poses.append(q_rand)
            
    print(f"Найдено {len(valid_poses)}/{num_poses} разнообразных поз за {attempts} попыток.")

    # Формируем структуру данных (для реального робота вместо 0.0 нужно будет подставить реальную длину)
    for angles in valid_poses:
        contribution = calculate_contribution(model, angles)
        row = (angles / DEG).tolist() + ["free_4_to_6", 0.0] + contribution.flatten().tolist()
        all_results.append(row)
        
    return all_results
    
def main(args):
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    model = hayati_model.HayatiModel(config)
    model.jac_DH_0 = calculate_jac_DH_params(model, model.zero_wire_angles)

    wrist_dataset = generate_wrist_block_dataset(model, num_poses=150)

    write_dataset(wrist_dataset, model.generation_output, model.fieldnames_options["wire_contributions"], tolerance=3)

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

    args = parser.parse_args()
    main(args)
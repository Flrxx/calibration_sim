import json
import argparse
import numpy as np
from math import sqrt, acos
import hayati_model
from typing import Union
from csv_routines import write_dataset
from dataset_generation_wire import calculate_wire_direction
from math_routines import DEG
from new_dataset_generation import calculate_contribution

# --- Базовые функции оценки (оставлены для совместимости) ---
def get_expected_deltas(model: hayati_model.HayatiModel):
    deltas = np.zeros(len(model.error_list))
    for i, param_name in enumerate(model.error_list):
        if 'alpha' in param_name or 'theta' in param_name or 'beta' in param_name:
            deltas[i] = model.theta_delta / DEG  
        elif 'a_' in param_name or 'd_' in param_name:
            deltas[i] = model.a_delta * 1000 
    return deltas

def calculate_expected_influence_mm(model, angles):
    base_contribution = calculate_contribution(model, angles)
    deltas = get_expected_deltas(model)
    return base_contribution * deltas

def rate_pose_for_parameter(model, angles, target_param, already_calibrated_params):
    influence_mm = calculate_expected_influence_mm(model, angles)
    target_idx = model.error_list.index(target_param)
    target_signal = abs(influence_mm[target_idx])
    if target_signal < 0.1: return 0.0

    noise_sq = 0.0
    for i, p_name in enumerate(model.error_list):
        if i == target_idx: continue
        influence_sq = influence_mm[i]**2
        if p_name in already_calibrated_params:
            noise_sq += influence_sq * 0.1 
        else:
            noise_sq += influence_sq * 1.0
            
    total_noise = sqrt(noise_sq)
    return target_signal / (total_noise + 1e-6)

# --- Физические ограничения ---
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

# # --- Расчет влияния ---
# def calculate_contribution(model, angles: Union[list, np.ndarray]):
#     jac_DH = calculate_jac_DH_params(model, angles)
#     wire_direction = calculate_wire_direction(model, angles)
#     contribution = np.dot(wire_direction.T, jac_DH - model.jac_DH_0)
#     return contribution.flatten()

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

# ==========================================
# СПЕЦИАЛЬНЫЙ ТЕСТОВЫЙ КОД (ПРОГРЕССИВНАЯ РАЗМОРОЗКА)
# ==========================================
def generate_progressive_test(model, num_poses_per_step=20, min_distance_rad=0.3):
    all_results = []
    
    # Идем от 5 (только 6-й сустав разморожен) до 0 (все суставы разморожены)
    for start_idx in range(5, -1, -1):
        label = f"free_{start_idx + 1}_to_6"
        print(f"\n--- Сбор данных: разморожены суставы с {start_idx + 1} по 6 ---")
        
        valid_poses = []
        attempts = 0
        
        # Генерируем позы, пока не наберем нужное количество
        while len(valid_poses) < num_poses_per_step and attempts < 100000:
            attempts += 1
            
            # Начинаем с замороженной нулевой позы
            q_rand = np.copy(model.zero_wire_angles)
            
            # Размораживаем разрешенные суставы
            for j in range(start_idx, 6):
                q_rand[j] = np.random.uniform(low=-np.pi, high=np.pi)
                
            # Проверяем физические лимиты
            if not is_pose_valid(model, q_rand):
                continue
                
            # Проверяем, чтобы позы не слипались в одну точку
            is_far_enough = True
            for selected_pose in valid_poses:
                if np.linalg.norm(q_rand - selected_pose) < min_distance_rad:
                    is_far_enough = False
                    break
                    
            if is_far_enough:
                valid_poses.append(q_rand)
                
        print(f"Найдено {len(valid_poses)}/{num_poses_per_step} разнообразных поз за {attempts} попыток.")
        
        # Считаем влияние и добавляем в датасет
        for angles in valid_poses:
            contribution = calculate_contribution(model, angles)
            # Записываем label вместо имени параметра, и 0.0 вместо SNR
            row = (angles / DEG).tolist() + [label, 0.0] + contribution.flatten().tolist()
            all_results.append(row)
            
    return all_results

def main(args):
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    model = hayati_model.HayatiModel(config)
    
    # Инициализация нулевого Якобиана обязательна для корректного расчета contribution!
    model.jac_DH_0 = calculate_jac_DH_params(model, model.zero_wire_angles)
    
    print("Запуск тестового скрипта прогрессивной разморозки суставов...")
    test_results = generate_progressive_test(model, num_poses_per_step=20)
    
    # Сохраняем тестовый датасет
    output_filename = model.generation_output
    write_dataset(test_results, output_filename, model.fieldnames_options["wire_contributions"], tolerance=3)
    print(f"\nТестовые данные успешно сохранены в {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Name of .json configuration file. Default: ARM95.json", default="src/config/ARM95_calibration.json")
    args = parser.parse_args()
    main(args)
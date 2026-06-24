import numpy as np
import json
import argparse
from math import sqrt, acos
from typing import Union
import hayati_model
from dataset_generation_wire import calculate_wire_direction
from math_routines import DEG

try:
    from new_dataset_generation import calculate_jac_DH_params
except ImportError:
    from pose_generation import calculate_jac_DH_params

# === 1. МАКРО-УРОВЕНЬ: Строгое кинематическое распределение ===
MACRO_BLOCKS = [
    {"name": "free_6_to_6", "start_idx": 5, "params": ["alpha_6"]},
    {"name": "free_5_to_6", "start_idx": 4, "params": ["a_6", "d_6", "theta_6", "a_5", "alpha_5", "theta_5"]},
    {"name": "free_4_to_6", "start_idx": 3, "params": ["d_5", "a_4", "alpha_4", "theta_4"]},
    {"name": "free_3_to_6", "start_idx": 2, "params": ["d_4", "a_3", "alpha_3", "theta_3"]},
    {"name": "free_2_to_6", "start_idx": 1, "params": ["beta_3", "a_2", "alpha_2", "theta_2"]},
    {"name": "free_1_to_6", "start_idx": 0, "params": ["beta_2", "a_1", "alpha_1"]}
]

def calculate_contribution(model: hayati_model.HayatiModel, angles: Union[list, np.ndarray]):
    jac_DH = calculate_jac_DH_params(model, angles)
    wire_direction = calculate_wire_direction(model, angles)
    contribution = np.dot(wire_direction.T, jac_DH - model.jac_DH_0)
    contribution[0, model.angle_idexes] *= DEG * 1000 # mm/deg
    return contribution.flatten()

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

# === Генератор тестовых точек (Монте-Карло) ===
def generate_block_poses(model, start_idx, num_poses=500):
    valid_poses = []
    attempts = 0
    
    while len(valid_poses) < num_poses and attempts < 50000:
        attempts += 1
        q_rand = np.copy(model.zero_wire_angles)
        for j in range(start_idx, 6):
            q_rand[j] = np.random.uniform(low=-np.pi, high=np.pi)
            
        if is_pose_valid(model, q_rand):
            valid_poses.append(q_rand)
            
    return valid_poses

# === МИКРО-УРОВЕНЬ: Анализ блока и фильтрация ===
def process_block(model, block_name, block_params, poses):
    if not poses:
        print(f"[{block_name}] 🔴 Ошибка: Не удалось сгенерировать точки.")
        return [], []
        
    if not block_params:
        return [], []
        
    print(f"\n=== БЛОК: {block_name} ({len(poses)} точек) ===")
    
    param_indices = [model.error_list.index(p) for p in block_params]
    J = np.zeros((len(poses), len(block_params)))
    
    for i, angles in enumerate(poses):
        contrib = calculate_contribution(model, angles)
        for j, global_idx in enumerate(param_indices):
            J[i, j] = contrib[global_idx]
            
    # 1. Оцениваем силу физического сигнала (норма градиентов по столбцам)
    norms = np.linalg.norm(J, axis=0)
    
    # 2. Сортируем параметры по убыванию сигнала
    sorted_indices = np.argsort(norms)[::-1]
    sorted_params = [block_params[i] for i in sorted_indices]
    sorted_norms = [norms[i] for i in sorted_indices]
    
    # 3. Рассчитываем матрицу корреляции
    norms_safe = np.copy(norms)
    norms_safe[norms_safe < 1e-12] = 1e-12
    J_norm = J / norms_safe
    corr_matrix = np.abs(np.dot(J_norm.T, J_norm))
    
    # 4. Сохраняем активные, переносим ненаблюдаемые
    active_params = []
    deferred_params = []
    
    print(f"{'Параметр':<10} | {'Сигнал (ед)':<12} | {'Статус'}")
    print("-" * 55)
    
    for i in range(len(sorted_params)):
        p1 = sorted_params[i]
        signal = sorted_norms[i]
        orig_idx1 = block_params.index(p1)
        
        # Если сигнал слишком слаб, переносим параметр на следующий слой разморозки
        if signal < 0.01:
            print(f"{p1:<10} | {signal:<12.4f} | ⏭ ПЕРЕНЕСЕН НА СЛЕД. СЛОЙ")
            deferred_params.append(p1)
            continue
            
        active_params.append(p1)
        print(f"{p1:<10} | {signal:<12.4f} | 🟢 ДОБАВЛЕН В ПОРЯДОК")
        
        # Проверяем все параметры с МЕНЬШИМ сигналом (те, что ниже в отсортированном списке)
        for j in range(i + 1, len(sorted_params)):
            p2 = sorted_params[j]
            if sorted_norms[j] < 0.01:
                continue # Пропускаем те, что уже перенесены
                
            orig_idx2 = block_params.index(p2)
            corr = corr_matrix[orig_idx1, orig_idx2]
            
            if corr > 0.85:
                print(f"           | {'':<12} | ⚠️ ВНИМАНИЕ: Связь {corr*100:.1f}% с {p2}")
                
    return active_params, deferred_params

def main(args):
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    model = hayati_model.HayatiModel(config)
    model.jac_DH_0 = calculate_jac_DH_params(model, model.zero_wire_angles)
    
    print("="*60)
    print(" АВТОМАТИЧЕСКОЕ ПОСТРОЕНИЕ ПОРЯДКА КАЛИБРОВКИ (OED)")
    print("="*60)
    
    final_calibration_order = []
    deferred_queue = []
    
    # Проходим по блокам от фланца к базе
    for block in MACRO_BLOCKS:
        poses = []
        search_idx = block["start_idx"]
        
        # Добавляем к текущему блоку те параметры, которые не удалось "увидеть" на прошлом шаге
        current_params = block["params"] + deferred_queue
        deferred_queue = [] # Очищаем очередь, так как мы их сейчас проверим
        
        if not current_params:
            continue
            
        # Механизм Fallback: если при текущем замороженном состоянии не находим точек, 
        # размораживаем следующий сустав базы и ищем снова.
        while search_idx >= 0:
            poses = generate_block_poses(model, search_idx, num_poses=500)
            if len(poses) >= 10:
                if search_idx != block["start_idx"]:
                    print(f"\n[!] Блок {block['name']}: Недостаточно точек. Разморожен сустав {search_idx+1} для поиска.")
                break
            print(f"\n[!] Блок {block['name']}: Не найдено точек при start_idx={search_idx}. Расширяем поиск...")
            search_idx -= 1
            
        # Анализируем блок и фильтруем параметры
        active, deferred = process_block(model, block["name"], current_params, poses)
        
        final_calibration_order.extend(active)
        deferred_queue.extend(deferred)
        
    print("\n" + "="*60)
    print(" ИТОГОВЫЙ ПОРЯДОК КАЛИБРОВКИ (ДЛЯ ВСТАВКИ В КОД)")
    print("="*60)
    print(json.dumps(final_calibration_order, indent=4))
    
    if deferred_queue:
        print("\n--- Ненаблюдаемые параметры (Оставлены номинальными) ---")
        print("Эти параметры не удалось 'увидеть' даже при полной разморозке базы:")
        print(deferred_queue)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Name of .json configuration file", default="src/config/ARM95_calibration.json")
    args = parser.parse_args()
    main(args)
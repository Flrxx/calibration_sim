import numpy as np
import json
import argparse
import csv
import hayati_model
from math_routines import DEG

# Пытаемся импортировать самую актуальную функцию расчета влияния
try:
    from dataset_generation_wire import calculate_contribution, calculate_jac_DH_params
except ImportError:
    from pose_generation import calculate_contribution, calculate_jac_DH_params

# Тот самый утвержденный порядок параметров
TARGET_ORDER = [
    "alpha_6", 
    "a_6", "theta_6", "d_6", "theta_5", "a_5", "alpha_5",  
    "d_5", "alpha_4", "a_4", "theta_4",
    "alpha_3", "a_3", "theta_3", "d_4",
    "beta_3", "alpha_2", "a_2", "theta_2",
    "beta_2", "alpha_1", "a_1",
    
    "d_1", "theta_1"
]

def load_dataset(csv_filepath):
    dataset = []
    with open(csv_filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row or row[0].upper() == "AVERAGE" or row[0] == "-1": 
                continue
            dataset.append(row)
    return np.array(dataset, dtype=object)

def analyze_step_by_step_observability(model: hayati_model.HayatiModel, dataset: np.ndarray, target_order: list):
    """
    Эмулирует пошаговую калибровку. На шаге K оценивает влияние параметров
    с 1 по K-1 как "известных", а параметров с K+1 до конца как "шум".
    """
    print("\n" + "="*70)
    print(" ХРОНОЛОГИЧЕСКИЙ АНАЛИЗ НАБЛЮДАЕМОСТИ (CASCADE OBSERVABILITY)")
    print("="*70)
    
    # Углы в радианы
    mask = np.array([isinstance(val, str) for val in dataset[:, 6]])
    dataset = dataset[mask]
    dataset[:, 0:6] = dataset[:, 0:6].astype(float) * DEG
    
    for step_idx, current_param in enumerate(target_order):
        # Параметры, которые ЕЩЕ НЕ откалиброваны (они создают шум для текущего)
        uncalibrated_params = target_order[step_idx + 1:]
        
        # Извлекаем точки только для текущего параметра
        step_mask = dataset[:, 6] == current_param
        poses = dataset[step_mask, 0:6].astype(float)
        
        print(f"\n[{step_idx + 1}/{len(target_order)}] ШАГ КАЛИБРОВКИ: '{current_param}' (Точек: {len(poses)})")
        print("-" * 70)
        
        if len(poses) == 0:
            print(f"🔴 ОШИБКА: В датасете нет точек для параметра {current_param}!")
            continue
            
        # Формируем подматрицу Якобиана ТОЛЬКО для текущего и еще неизвестных параметров
        current_subset = [current_param] + uncalibrated_params
        J = np.zeros((len(poses), len(current_subset)))
        
        for i, angles in enumerate(poses):
            full_contrib = calculate_contribution(model, angles)
            for j, p_name in enumerate(current_subset):
                global_idx = model.error_list.index(p_name)
                J[i, j] = full_contrib[global_idx]
                
        # 1. Сигнал текущего параметра (норма первого столбца)
        norms = np.linalg.norm(J, axis=0)
        target_signal = norms[0]
        
        if target_signal < 1e-6:
            print(f"🔴 КРИТИЧЕСКИЙ ПРОВАЛ: '{current_param}' имеет нулевой градиент (не влияет на трос)!")
            continue
            
        print(f"Полезный физический сигнал: {target_signal:.4f} ед.")
        
        # Если это последний параметр, шума от неизвестных больше нет
        if not uncalibrated_params:
            print("🟢 ИДЕАЛЬНО: Это последний параметр. Мешающих шумов больше нет.")
            continue
            
        # 2. Число обусловленности оставшейся подсистемы
        # Показывает, насколько "спутаны" оставшиеся параметры на этих конкретных точках
        _, s, _ = np.linalg.svd(J)
        cond_number = s[0] / s[-1] if s[-1] > 1e-12 else float('inf')
        print(f"Обусловленность подсистемы (Cond Num): {cond_number:.2f}")
        
        # 3. Анализ кросс-корреляции текущего параметра с неизвестными
        norms_safe = np.copy(norms)
        norms_safe[norms_safe < 1e-12] = 1e-12
        J_normalized = J / norms_safe
        
        corr_matrix = np.abs(np.dot(J_normalized.T, J_normalized))
        
        # Нас интересует только первая строка матрицы (связь текущего со всеми остальными)
        target_correlations = corr_matrix[0, 1:]
        
        critical_links = []
        medium_links = []
        
        for j, corr in enumerate(target_correlations):
            uncalib_name = uncalibrated_params[j]
            if corr > 0.85:
                critical_links.append((corr, uncalib_name))
            elif corr > 0.50:
                medium_links.append((corr, uncalib_name))
                
        if not critical_links and not medium_links:
            print(f"🟢 ЧИСТЫЙ ГРАДИЕНТ: '{current_param}' полностью изолирован от будущего шума.")
        else:
            if critical_links:
                print("🔴 ОПАСНОСТЬ ВПИТЫВАНИЯ ШУМА (>85%):")
                critical_links.sort(reverse=True)
                for corr, name in critical_links:
                    print(f"   ⚠️ Оптимизатор может спутать '{current_param}' с '{name}' (Связь {corr*100:.1f}%)")
                    
            if medium_links:
                print("🟠 УМЕРЕННЫЙ РИСК ДРЕЙФА (>50%):")
                medium_links.sort(reverse=True)
                for corr, name in medium_links:
                    print(f"   - с '{name}' (Связь {corr*100:.1f}%)")

def main(args):
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    model = hayati_model.HayatiModel(config)
    
    # Обязательно инициализируем нулевой Якобиан
    model.jac_DH_0 = calculate_jac_DH_params(model, model.zero_wire_angles)
    
    dataset = load_dataset(args.dataset)
    
    analyze_step_by_step_observability(model, dataset, TARGET_ORDER)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Хронологический анализ корреляций")
    parser.add_argument('--config', type=str, default="src/config/ARM95_calibration.json")
    parser.add_argument('--dataset', default="datasets/ARM95/wire/calibration/calibration_dataset.csv", help="Путь к CSV датасету")

    args = parser.parse_args()
    main(args)
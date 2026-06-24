import numpy as np
import json
import argparse
import hayati_model
from new_dataset_generation import calculate_contribution, calculate_jac_DH_params
from math_routines import DEG
from csv_routines import read_dataset

def analyze_step_by_step_correlation(model, dataset, all_params):
    """
    Анализирует датасет пошагово: для каждого целевого параметра (столбец 6)
    вычисляет его независимость от других параметров на выделенных для него точках.
    """
    # 1. Извлекаем порядок калибровки из датасета (сохраняя последовательность)
    calibration_steps = []
    for val in dataset[:, 6]:
        if val not in calibration_steps:
            calibration_steps.append(val)
            
    print("\n" + "="*60)
    print(f"=== ПОШАГОВЫЙ АНАЛИЗ ДАТАСЕТА ({len(calibration_steps)} ШАГОВ) ===")
    print("="*60)
    
    calibrated_so_far = set()
    
    for step_idx, step_param in enumerate(calibration_steps):
        # Выделяем точки, предназначенные только для текущего параметра
        step_mask = dataset[:, 6] == step_param
        poses = dataset[step_mask, 0:6].astype(float)
        
        print(f"\n[{step_idx + 1}/{len(calibration_steps)}] Анализ шага: '{step_param}' ({len(poses)} точек)")
        print("-" * 60)
        
        if step_param not in all_params:
            print(f"Ошибка: Параметр '{step_param}' не найден в модели!")
            continue
            
        target_idx = all_params.index(step_param)
        
        # Вычисляем матрицу Якобиана только для этих точек
        J = np.zeros((len(poses), len(all_params)))
        for i, angles in enumerate(poses):
            J[i, :] = calculate_contribution(model, angles)
            
        # Оцениваем силу влияния (норму градиентов)
        norms = np.linalg.norm(J, axis=0)
        target_norm = norms[target_idx]
        
        if target_norm < 1e-6:
            print(f"🔴 КРИТИЧЕСКИЙ ПРОВАЛ: Параметр '{step_param}' физически не влияет на трос в этих точках (Градиент ~ 0)!")
            calibrated_so_far.add(step_param)
            continue
            
        print(f"Сила полезного сигнала (Норма градиента): {target_norm:.4f}")
        
        # Нормализуем векторы для вычисления косинусного сходства
        norms_safe = np.copy(norms)
        norms_safe[norms_safe < 1e-12] = 1e-12
        J_normalized = J / norms_safe
        
        target_vector = J_normalized[:, target_idx]
        
        # Списки для опасных связей
        critical_uncalibrated = []
        medium_uncalibrated = []
        
        for j, p_name in enumerate(all_params):
            if j == target_idx:
                continue
                
            # Косинусное сходство (корреляция)
            corr = abs(np.dot(target_vector, J_normalized[:, j]))
            
            # Пропускаем параметры, которые не влияют на этих точках
            if norms[j] < 1e-6:
                continue
                
            # Нас волнуют только параметры, которые еще НЕ откалиброваны.
            # Ошибки уже откалиброванных параметров зафиксированы, они не сломают оптимизатор на этом шаге.
            if p_name not in calibrated_so_far:
                if corr > 0.85:
                    critical_uncalibrated.append((corr, p_name))
                elif corr > 0.50:
                    medium_uncalibrated.append((corr, p_name))
                    
        # Вывод результатов для текущего шага
        if not critical_uncalibrated and not medium_uncalibrated:
            print(f"🟢 ИДЕАЛЬНО: '{step_param}' полностью изолирован от неизвестных параметров!")
        else:
            if critical_uncalibrated:
                print("🔴 КРИТИЧЕСКАЯ СВЯЗЬ (>85%): Оптимизатор 'впитает' ошибку этих параметров!")
                critical_uncalibrated.sort(reverse=True)
                for corr, name in critical_uncalibrated:
                    print(f"   - с {name} (Сходство: {corr*100:.1f}%)")
                    
            if medium_uncalibrated:
                print("🟠 УМЕРЕННАЯ СВЯЗЬ (>50%): Возможна небольшая компенсаторная ошибка.")
                medium_uncalibrated.sort(reverse=True)
                for corr, name in medium_uncalibrated:
                    print(f"   - с {name} (Сходство: {corr*100:.1f}%)")
                    
        # Добавляем параметр в список "известных" для следующих шагов
        calibrated_so_far.add(step_param)


def main(args):
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    model = hayati_model.HayatiModel(config)
    
    # Инициализация базового Якобиана
    model.jac_DH_0 = calculate_jac_DH_params(model, model.zero_wire_angles)

    print(f"Загрузка датасета: {args.dataset}")
    # Читаем датасет, используя твой метод
    dataset = read_dataset(args.dataset, model.fieldnames_options["wire_contributions"])
    
    # Перевод углов в радианы
    dataset[:, 0:6] *= DEG
    
    # Фильтруем строки, оставляя только те, где в колонке 'index' лежит строковое имя параметра
    mask = np.array([isinstance(val, str) for val in dataset[:, 6]])
    dataset = dataset[mask]
    
    analyze_step_by_step_correlation(model, dataset, model.error_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Путь к конфигу", default="src/config/ARM95_calibration.json")
    parser.add_argument("-d", "--dataset", help="Путь к CSV датасету", required=True)
    args = parser.parse_args()
    main(args)
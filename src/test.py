import numpy as np
import json
import argparse
import copy
import itertools
from typing import List, Tuple
from multiprocessing import Pool, cpu_count

from calibration_wire import step_by_step_base_calibration, validate_wire
import hayati_model
from csv_routines import read_dataset
from math_routines import DEG

# Базовый набор параметров (твоя зафиксированная маска)
BASE_PARAMS = [

    "beta_2",
    "beta_3",
    "theta_2",
    "theta_3",

    "theta_6",

  
]

# Кандидаты на добавление
CANDIDATE_PARAMS = [
    "d_5",  "alpha_4", "alpha_2", "a_1",
        "alpha_1",
    "alpha_3",
    "alpha_5",
    "a_3",
    "a_4",
    "a_5",
    "d_6",     "theta_4",
    "theta_5"    
]

# Стадии прогона
STAGES = [
    (1, False),
    (2, False),
    (3, False)
]

def test_parameter_set(model: hayati_model.HayatiModel, train_dataset, val_dataset, current_params: List[str]) -> Tuple[float, bool]:
    """Прогоняет набор параметров и отслеживает устойчивость на валидационном датасете."""
    model.calibration_mask = current_params 
    # if hasattr(model, 'error_list'):
    #     model.error_list = current_params
        
    stage_improvements = []
    
    # Замеряем базовую ошибку (до калибровки), чтобы корректно считать diff (улучшение)
    model.reset_estimated_dh()
    # out_nom = validate_wire(model, val_dataset, 1)
    # #print(out_nom)
    # nom_error = out_nom[-1, -1] * 1000 # Ожидается, что последняя ячейка - средняя ошибка
    
    for passes, polish in STAGES:
        try:
            model.reset_estimated_dh()
            step_by_step_base_calibration(model, dataset=train_dataset, is_real=1, passes=passes, polish=polish)
            out_val = validate_wire(model, val_dataset, 1)
            
            calib_error = out_val[-1, -1] * 1000
            #improvement = nom_error - calib_error
            stage_improvements.append(calib_error)
        except Exception as e:
            model.reset_estimated_dh()
            print(f"      [Ошибка алгоритма]: {e}")
            return -999.0, False
            
    # Вывод промежуточных значений проходов для наглядности
    # stages_str = " | ".join([f"Цикл {i+1}: {imp:+.4f}" for i, imp in enumerate(stage_improvements)])
    # print(f"      -> {stages_str}")

    # Проверка устойчивости (Divergence Check) - ТЕПЕРЬ ДИНАМИЧЕСКАЯ
    if len(stage_improvements) < len(STAGES):
        return -999.0, False
        
    pass_1 = stage_improvements[0]
    final_imp = stage_improvements[-1]
    is_stable = True
    
    # Если улучшение сильно падает (на 0.15 мм) по сравнению с 1-м проходом или уходит в минус — это расходимость
    if final_imp < (pass_1 - 0.15) or final_imp < -0.5 or np.isnan(final_imp):
        is_stable = False
        
    return final_imp, is_stable


def _worker_evaluate_combo(args):
    """
    Функция-воркер для мультипроцессинга. Создает изолированную модель
    и прогоняет тест для переданной комбинации.
    """
    combo, config_dict, train_dataset, val_dataset, base_params = args
    
    # Изоляция памяти: создаем свою модель для каждого процесса
    local_model = hayati_model.HayatiModel(config_dict)
    test_params = list(base_params) + list(combo)
    
    improvement, is_stable = test_parameter_set(local_model, train_dataset, val_dataset, test_params)
    return combo, improvement, is_stable


def run_forward_selection(config_dict: dict, model: hayati_model.HayatiModel, train_path: str, val_path: str):
    print("=" * 70)
    print(" АЛГОРИТМ ПРЯМОГО ДОБАВЛЕНИЯ ПАРАМЕТРОВ (PARALLEL FORWARD SELECTION)")
    print("=" * 70)

    # 1. Загрузка датасетов
    train_dataset = read_dataset(train_path, model.fieldnames_options["wire_samples"])
    train_dataset[:, :6] *= DEG
    train_dataset[:, 7] /= 1000

    val_dataset = read_dataset(val_path, model.fieldnames_options["test_result"])
    val_dataset[:, :6] *= DEG
    val_dataset[:, 6:] /= 1000
    
    #val_dataset = val_dataset[:-1, :]  # Убираем последнюю строку с суммарными результатами

    print(f"[*] Точек для калибровки (Train): {len(train_dataset)}")
    print(f"[*] Точек для валидации (Test):   {len(val_dataset)}\n")

    # ==============================================================
    # ФАЗА ПРОВЕРКИ БАЗОВОГО НАБОРА
    # ==============================================================
    current_params = list(BASE_PARAMS)
    print(f"[*] Тестирование базового набора из {len(current_params)} параметров...")
    
    best_improvement, is_stable = test_parameter_set(model, train_dataset, val_dataset, current_params)
    if best_improvement == -999.0: 
        best_improvement = 0.0
        print("  [!] Базовый набор выдал ошибку. Стартовый diff = 0.0")
    elif not is_stable:
        print(f"  [!] ВНИМАНИЕ: Базовый набор нестабилен! (Его diff: {best_improvement:+.4f} мм)")
        # Даже если нестабилен, мы запоминаем его diff, чтобы кандидаты пытались его превзойти
    else:
        print(f"  [+] Утвержден базовый diff: {best_improvement:+.4f} мм\n")

    best_params = list(current_params)

    # ==============================================================
    # ФАЗА ПОЛНОГО ПЕРЕБОРА (МНОГОПОТОЧНАЯ)
    # ==============================================================
    total_combinations = (2 ** len(CANDIDATE_PARAMS)) - 1
    num_cores = cpu_count()
    
    print("-" * 70)
    print(f" НАЧАЛО ПАРАЛЛЕЛЬНОГО ПЕРЕБОРА НА {num_cores} ЯДРАХ ({total_combinations} ВАРИАНТОВ)")
    print("-" * 70)

    # Подготавливаем все возможные комбинации
    all_combos = []
    for r in range(1, len(CANDIDATE_PARAMS) + 1):
        for combo in itertools.combinations(CANDIDATE_PARAMS, r):
            all_combos.append(combo)

    # Упаковываем аргументы для воркеров
    worker_args = [(combo, config_dict, train_dataset, val_dataset, BASE_PARAMS) for combo in all_combos]

    completed = 0
    # Используем imap_unordered для моментального вывода результатов по мере их готовности
    with Pool(processes=num_cores) as pool:
        for combo, current_improvement, is_stable in pool.imap_unordered(_worker_evaluate_combo, worker_args):
            completed += 1

            diff_gain = current_improvement - best_improvement
            if is_stable and diff_gain > 0:
                
                print(f"[{completed}/{total_combinations}] [+] НОВЫЙ РЕКОРД! {list(combo)} | diff: {current_improvement:+.4f} (Прирост: +{diff_gain:.4f})")
                best_params = list(BASE_PARAMS) + list(combo)
                best_improvement = current_improvement
            else:
                reason = "Нестабилен" if not is_stable else f"({current_improvement:+.4f})"
                print(f"[{completed}/{total_combinations}] [-] Отклонен {list(combo)} | {reason}")
            
    print("=" * 70)
    print(" ФИНАЛЬНЫЙ РЕЗУЛЬТАТ ПОЛНОГО ПЕРЕБОРА")
    print("=" * 70)
    print(f"Итоговое максимальное улучшение троса: {best_improvement:+.4f} мм")
    print(f"Итоговый оптимальный список ({len(best_params)} параметров):")
    print(json.dumps(best_params, indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Укажи здесь пути по умолчанию к твоим новым датасетам
    parser.add_argument("-d", "--dataset", default="datasets/ARM95/wire/calibration/calibration_dataset_75_deg.csv")
    parser.add_argument("-v", "--validation", default="results/ARM95/validation_results_real.csv")
    parser.add_argument("-c", "--config", default="src/config/ARM95_calibration_75_deg.json")
    args = parser.parse_args()

    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
        
    model = hayati_model.HayatiModel(config)
    
    # Передаем config в функцию, чтобы размножать его в потоках
    run_forward_selection(config, model, args.dataset, args.validation)

# [
#     "beta_2",
#     "beta_3",

#     "theta_2",
#     "theta_3",
#     "theta_4",
#     "theta_5",
#     "theta_6",

#     "alpha_3",
#     "alpha_6",

#     "a_1",
#     "a_2",
#     "a_3",
#     "a_4",
#     "d_5",
#     "d_6"
# ]

# [
#     "beta_2",
#     "beta_3",

#     "theta_2",
#     "theta_3",
#     "theta_4",
#     "theta_5",
#     "theta_6",

#     "alpha_1",
#     "alpha_3",
#     "alpha_5",
#     "alpha_6",
#     "a_2",
#     "a_3",
#     "a_4",
#     "a_5",
#     "d_6"   
# ]

import numpy as np
import json
import argparse
import copy
import itertools
from typing import List, Tuple
from calibration_wire import step_by_step_base_calibration, validate_wire
import hayati_model
from csv_routines import read_dataset
from math_routines import DEG

# Базовый набор параметров (твоя зафиксированная маска из 15 параметров)
# BASE_PARAMS = [
#     "beta_3", "theta_2", "theta_3", "a_4", "alpha_3", "a_2", "beta_2", "alpha_5",
#     "d_6", "theta_6", "theta_5", "alpha_6", "a_3", "theta_4", "alpha_1"
# ]

# # Кандидаты на добавление (оставшиеся кинематические параметры, кроме невидимых d_1, theta_1)
# CANDIDATE_PARAMS = [
#     "a_6", "a_5", "d_5", "alpha_4", "alpha_2", "a_1"#, "d_4"
# ]

BASE_PARAMS = [
    "beta_3", "theta_2", "theta_3", "beta_2", "theta_6", "theta_5","theta_4"
]

# Кандидаты на добавление (оставшиеся кинематические параметры, кроме невидимых d_1, theta_1)
CANDIDATE_PARAMS = [
     "a_5", "d_5", "alpha_4", "alpha_2", "a_1", "a_2", "a_4", "alpha_3", "alpha_5",
    "d_6",  "alpha_6", "a_3", "alpha_1"# "d_4", "a_6"
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
    #stages_str = " | ".join([f"Цикл {i+1}: {imp:+.4f}" for i, imp in enumerate(stage_improvements)])
    #print(f"      -> {stages_str}")

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

def run_forward_selection(model: hayati_model.HayatiModel, train_path: str, val_path: str):
    print("=" * 70)
    print(" АЛГОРИТМ ПРЯМОГО ДОБАВЛЕНИЯ ПАРАМЕТРОВ (FORWARD SELECTION)")
    print("=" * 70)

    # 1. Загрузка датасетов
    train_dataset = read_dataset(train_path, model.fieldnames_options["wire_samples"])
    train_dataset[:, :6] *= DEG
    train_dataset[:, 7] /= 1000

    val_dataset = read_dataset(val_path, model.fieldnames_options["test_result"])
    
    val_dataset[:, :6] *= DEG
    val_dataset[:, 6:] /= 1000
    
    val_dataset = val_dataset[:-1, :]  # Убираем последнюю строку с суммарными результатами


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
    # ФАЗА ПОЛНОГО ПЕРЕБОРА (EXHAUSTIVE SEARCH)
    # ==============================================================
    print("-" * 70)
    print(f" НАЧАЛО ПОЛНОГО ПЕРЕБОРА КОМБИНАЦИЙ (ИЗ {len(CANDIDATE_PARAMS)} ПАРАМЕТРОВ)")
    print("-" * 70)

    total_combinations = (2 ** len(CANDIDATE_PARAMS)) - 1
    current_iteration = 0

    for r in range(1, len(CANDIDATE_PARAMS) + 1):
        for combo in itertools.combinations(CANDIDATE_PARAMS, r):
            current_iteration += 1
            test_params = list(BASE_PARAMS) + list(combo)
            
            print(f"[{current_iteration}/{total_combinations}] Пробуем добавить {list(combo)}...")
            current_improvement, is_stable = test_parameter_set(model, train_dataset, val_dataset, test_params)
            
            if is_stable and current_improvement > best_improvement:
                diff_gain = current_improvement - best_improvement
                print(f"  [+] НОВЫЙ РЕКОРД! Новый diff: {current_improvement:+.4f} (Прирост: +{diff_gain:.4f} мм)\n")
                best_params = list(test_params)
                best_improvement = current_improvement
            else:
                reason = "Нестабилен" if not is_stable else f"Ухудшил или не изменил diff ({current_improvement:+.4f})"
                print(f"  [-] Отклонен. Причина: {reason}")
            
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
    
    run_forward_selection(model, args.dataset, args.validation)
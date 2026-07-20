import numpy as np
import json
import argparse
import copy
from typing import List, Tuple
from calibration_wire import execute_calibration, validate_wire
import hayati_model
from csv_routines import read_dataset
from math_routines import DEG

# Все 20 доступных параметров (изначальный набор для старта)
ALL_PARAMS = [
    "a_3", "beta_3", "theta_2", "alpha_1", "a_1", "theta_3", "alpha_2", # Фаза 1 (Безопасные)
    "alpha_4", "d_5", "theta_4", "a_4",                                 # Фаза 2 (Средние)
    "alpha_3", "a_2", "beta_2",                                         # Фаза 3 (Пограничные)
    "alpha_5", "d_6", "a_5", "a_6", "theta_5", "alpha_6"                # Фаза 4 (Зона смерти)
]

# Порядок проверки на УДАЛЕНИЕ (от самых токсичных к самым безопасным)
PARAMS_TO_TEST_REMOVAL = [
    "alpha_6", "theta_5", "a_6", "a_5", "d_6", "alpha_5", # Сначала пытаемся выкинуть кисть
    "beta_2", "a_2", "alpha_3",                           # Затем пограничные
    "a_4", "theta_4", "d_5", "alpha_4",                   # Затем средние
    "a_1", "theta_2", "alpha_2", "theta_3",               # Безопасные (скорее всего их удаление ухудшит результат)
    "alpha_1", "beta_3", "a_3"
]

# Стадии прогона
STAGES = [
    (1, False),
    (2, False),
    (3, False),
    (3, True)
]

def test_parameter_set(model: hayati_model.HayatiModel, train_dataset, val_subset, current_params: List[str]) -> Tuple[float, bool]:
    """Прогоняет набор параметров и отслеживает устойчивость на переданной подвыборке."""
    model.calibration_mask = current_params 
    if hasattr(model, 'error_list'):
        model.error_list = current_params
        
    stage_diffs = []
    
    for passes, polish in STAGES:
        try:
            execute_calibration(model, dataset=train_dataset, is_real=1, passes=passes, polish=polish)
            output = validate_wire(model, val_subset, 1)
            
            current_diff = output[-1, -1] * 1000
            stage_diffs.append(current_diff)
            model.reset_estimated_dh()
        except Exception as e:
            model.reset_estimated_dh()
            return -999.0, False
            
    # Проверка устойчивости (Divergence Check)
    if len(stage_diffs) < 4:
        return -999.0, False
        
    pass_1, pass_3, polish_diff = stage_diffs[0], stage_diffs[2], stage_diffs[3]
    is_stable = True
    
    if pass_3 < (pass_1 - 0.15) or polish_diff < (pass_3 - 0.15) or polish_diff < -1.0 or np.isnan(polish_diff):
        is_stable = False
        
    return polish_diff, is_stable

def run_auto_selection_and_test(model: hayati_model.HayatiModel, train_path: str, val_path: str, split_ratio: float = 0.5):
    print("=" * 70)
    print(" УЛЬТИМАТИВНЫЙ ТЕСТ: АВТО-ОТБОР (50%) + СЛЕПОЕ ТЕСТИРОВАНИЕ (50%)")
    print("=" * 70)

    # 1. Загрузка датасетов
    train_dataset = read_dataset(train_path, model.fieldnames_options["wire_samples"])
    train_dataset[:, :6] *= DEG
    train_dataset[:, 7] /= 1000

    full_val_dataset = read_dataset(val_path, model.fieldnames_options["test_result"])
    full_val_dataset[:, :6] *= DEG
    full_val_dataset[:, 6:] /= 1000

    # 2. Случайное перемешивание и разделение (Shuffle & Split)
    np.random.seed(42) # Фиксируем seed для стабильного сплита
    indices = np.arange(len(full_val_dataset))
    np.random.shuffle(indices)
    
    split_idx = int(len(full_val_dataset) * split_ratio)
    val_subset = full_val_dataset[indices[:split_idx]]
    test_subset = full_val_dataset[indices[split_idx:]]
    
    print(f"\n[*] Датасет разделен ({split_ratio*100:.0f}/{100 - split_ratio*100:.0f}):")
    print(f"    - Часть для отбора параметров: {len(val_subset)} точек")
    print(f"    - Часть для слепого теста:     {len(test_subset)} точек\n")

    # ==============================================================
    # ФАЗА 1: ОБРАТНОЕ ИСКЛЮЧЕНИЕ НА 1-Й ПОЛОВИНЕ ДАТАСЕТА
    # ==============================================================
    print("-" * 70)
    print(" ФАЗА 1: ПОИСК ИДЕАЛЬНОГО НАБОРА ПАРАМЕТРОВ")
    print("-" * 70)
    
    current_params = list(ALL_PARAMS)
    best_diff, _ = test_parameter_set(model, train_dataset, val_subset, current_params)
    if best_diff == -999.0: best_diff = -5.0
        
    print(f"[*] Базовый diff со всеми {len(current_params)} параметрами: {best_diff:+.4f} мм\n")

    for param in PARAMS_TO_TEST_REMOVAL:
        if param not in current_params: continue
            
        test_params = list(current_params)
        test_params.remove(param)
        
        current_diff, is_stable = test_parameter_set(model, train_dataset, val_subset, test_params)
        
        if is_stable and current_diff > best_diff:
            improvement = current_diff - best_diff if best_diff > -100 else current_diff
            print(f"  [+] УДАЛЕН '{param:<8}' | Новый diff: {current_diff:+.4f} (Улучшение: +{improvement:.4f} мм)")
            current_params = test_params
            best_diff = current_diff
        else:
            reason = "Нестабилен" if not is_stable else f"Упал до {current_diff:+.4f}"
            print(f"  [-] ОСТАВЛЕН '{param:<8}' | Причина: {reason}")
            
    print(f"\n[+] Фаза 1 завершена. Найден идеальный набор из {len(current_params)} параметров.")

    # ==============================================================
    # ФАЗА 2: СЛЕПОЕ ТЕСТИРОВАНИЕ НА 2-Й ПОЛОВИНЕ ДАТАСЕТА
    # ==============================================================
    print("\n" + "-" * 70)
    print(" ФАЗА 2: СЛЕПОЕ ТЕСТИРОВАНИЕ НА ОТЛОЖЕННОЙ ВЫБОРКЕ")
    print("-" * 70)
    
    # Обучаем модель финально на найденных параметрах
    model.calibration_mask = current_params                 
    #if hasattr(model, 'error_list'):
        #model.error_list = current_params
    model.reset_estimated_dh() 
    print("[*] Обучение модели (3 цикла + Полировка)...")
    # for passes in [1, 2, 3]:
    #     execute_calibration(model, dataset=train_dataset, is_real=1, passes=passes, polish=False)
    execute_calibration(model, dataset=train_dataset, is_real=1, passes=3, polish=True)
       
    # Тестируем на 1-й половине (Псевдо-валидация)
    out_val = validate_wire(model, val_subset, 1)
    val_diff = out_val[-1, -1] * 1000
    
    # Тестируем на 2-й половине (Слепой тест)
    out_test = validate_wire(model, test_subset, 1)
    test_diff = out_test[-1, -1] * 1000

    print("\n" + "=" * 70)
    print(" ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ (СРАВНЕНИЕ)")
    print("=" * 70)
    print(f"{'Метрика':<30} | {'Отбор (50%)':<12} | {'Слепой Тест (50%)'}")
    print("-" * 70)
    print(f"{'Улучшение точности (diff)':<30} | {val_diff:>+8.4f} мм | {test_diff:>+8.4f} мм")
    print("=" * 70)
    
    difference = abs(val_diff - test_diff)
    print("\n[ВЕРДИКТ АЛГОРИТМА]:")
    if difference < 0.2 and test_diff > 0:
        print("🟢 ИДЕАЛЬНОЕ ОБОБЩЕНИЕ! Результат на слепом тесте подтверждает выбор параметров.")
        print("Модель надежна и не переобучилась.")
    elif test_diff < 0:
        print("🔴 ПЕРЕОБУЧЕНИЕ! Выбранный набор параметров работает только для 1-й половины.")
    else:
        print("🟠 ПРИЕМЛЕМО. Модель в плюсе на тесте, но есть расхождение между половинами.")
        
    print("\n=== ИТОГОВЫЙ СПИСОК ПАРАМЕТРОВ ДЛЯ КОНФИГА ===")
    print(json.dumps(current_params, indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="datasets/ARM95/wire/calibration/calibration_dataset.csv")
    parser.add_argument("-v", "--validation", default="results/ARM95/validation_results_real.csv")
    parser.add_argument("-c", "--config", default="src/config/ARM95_calibration.json")
    parser.add_argument("--split", type=float, default=0.5, help="Доля точек для отбора параметров (по умолчанию 0.5)")
    args = parser.parse_args()

    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    model = hayati_model.HayatiModel(config)
    
    run_auto_selection_and_test(model, args.dataset, args.validation, args.split)


# # 1.

# ======================================================================
#  ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ (СРАВНЕНИЕ)
# ======================================================================
# Метрика                        | Отбор (50%)  | Слепой Тест (50%)
# ----------------------------------------------------------------------
# Улучшение точности (diff)      |  +0.8763 мм |  +1.2594 мм
# ======================================================================
#     [
#     "beta_3",
#     "theta_2",
#     "theta_3",
#     "a_4",
#     "alpha_3",
#     "a_2",
#     "beta_2",
#     "alpha_5",
#     "d_6",
#     "a_6",
#     "theta_5",
#     "alpha_6"
# ]




# 2.
# ======================================================================
#  ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ (СРАВНЕНИЕ)
# ======================================================================
# Метрика                        | Отбор (50%)  | Слепой Тест (50%)
# ----------------------------------------------------------------------
# Улучшение точности (diff)      |  +0.9516 мм |  +0.2789 мм
# ======================================================================
# [
#     "beta_3",
#     "theta_2",
#     "alpha_1",
#     "a_1",
#     "theta_3",
#     "alpha_2",
#     "alpha_4",
#     "d_5",
#     "theta_4",
#     "a_2",
#     "beta_2",
#     "alpha_5",
#     "d_6",
#     "a_6",
#     "theta_5",
#     "alpha_6"
# ]

# 3.
# ======================================================================
#  ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ (СРАВНЕНИЕ)
# ======================================================================
# Метрика                        | Отбор (50%)  | Слепой Тест (50%)
# ----------------------------------------------------------------------
# Улучшение точности (diff)      |  +0.3426 мм |  +1.1262 мм
# ======================================================================
# [
#     "a_3",
#     "beta_3",
#     "theta_2",
#     "alpha_1",
#     "a_1",
#     "theta_3",
#     "alpha_2",
#     "a_4",
#     "alpha_3",
#     "a_2",
#     "beta_2",
#     "alpha_5",
#     "d_6",
#     "a_5",
#     "a_6",
#     "theta_5",
#     "alpha_6"
# ]
import numpy as np
from math import cos, sin, pi, sqrt, atan2, asin, log10, acos, copysign
from typing import Union
from scipy.optimize import minimize_scalar, minimize, Bounds, least_squares
import argparse
import json
import time
import hayati_model
from math_routines import extract_zyx_euler
from random import uniform
from dataset_generation_absolute import generate_real_dh
from dataset_generation_wire import measure_all_distances, calculate_distance
from csv_routines import read_dataset, write_dataset
from multiprocessing import Pool, cpu_count
import copy
from math_routines import DEG
from graph import plot_side_by_side_hist, plot_dh_comparison, plot_all_6_axes, plot_calibration_errors_from_data
import itertools
from calibration_wire import calculate_estimated_distance, write_results, write_validation_dataset


def calibrate_wrist_block(model, dataset):
    """
    Изолированно калибрует Блок Кисти (11 параметров для 4, 5 и 6 звеньев).
    dataset: массив, где первые 6 столбцов - углы, 7-й столбец (индекс 7) - реальная длина.
    """
    #print("\n=== Старт оптимизации: Блок Кисти (4, 5, 6 звенья) ===")
    
    # Строгий список 11 параметров из нашей таблицы для free_4_to_6
    target_params = [
        "alpha_6", "a_6", "d_6", "theta_6",
        "a_5", "alpha_5", "theta_5", "d_5",
        "a_4", "alpha_4", "theta_4"
    ]
    
    real_distances = dataset[:, 7] 
    
    # Извлекаем начальные значения из модели
    x0 = []
    param_indices = []
    for param_name in target_params:
        p_num, p_let = model.params_from_letters(param_name)
        x0.append(model.estimated_dh[p_num][p_let])
        param_indices.append((p_num, p_let))
        
    x0 = np.array(x0)
    
    def objective_function(x):
        # 1. Обновляем модель текущими предположениями оптимизатора
        for i, (p_num, p_let) in enumerate(param_indices):
            model.estimated_dh[p_num][p_let] = x[i]
            
        # 2. КРИТИЧЕСКИЙ ШАГ: Пересчитываем zero_pos внутри цикла! 
        # (изменение параметров кисти физически меняет нулевую точку фланца у датчика)
        zero_pos = model.get_transition_matrix(model.zero_wire_angles, "estimated")[0:3, 3]
        
        # 3. Вычисляем невязки для всех точек датасета
        residuals = np.zeros(len(dataset))
        for i, angles in enumerate(dataset[:, 0:6]):
            d_theor = calculate_estimated_distance(model, angles, zero_pos)
            residuals[i] = d_theor - real_distances[i]
            
        return residuals

    # Алгоритм Левенберга-Маркварда без жестких границ (ищем истинную дельту кисти)
    res = least_squares(
        objective_function, 
        x0=x0, 
        method='lm', 
        x_scale='jac', 
        ftol=1e-8, 
        xtol=1e-8,
        verbose=0
    )
    
    # Сохраняем откалиброванные значения обратно в модель
    for i, (p_num, p_let) in enumerate(param_indices):
        model.estimated_dh[p_num][p_let] = res.x[i]
        
    final_rmse = np.sqrt(np.mean(res.fun**2))
    #print(f"Оптимизация Блока Кисти завершена!")
    #print(f"Конечная RMSE (только для кисти): {final_rmse * 1000:.4f} мм")
    
    return res

def measure_wrist_block_distances(model, num_poses=30):
    """
    Генерирует позы для кисти и вычисляет "истинное" расстояние
    вытягивания троса, используя параметры real_dh симуляции.
    """

    dataset_orig = read_dataset(model.contribution_dataset, model.fieldnames_options["wire_contributions"])
    dataset_orig[:, 0:6] *= DEG
    #positive_mask = dataset_orig[:, 6] >= 0
    mask = np.array([isinstance(val, str) for val in dataset_orig[:, 6]])
    dataset_orig = dataset_orig[mask]

    # 1. Генерируем сырые данные (углы возвращаются в градусах)
    #dataset_orig = generate_wrist_block_dataset(model, num_poses=num_poses)
    
    # 2. Создаем числовой массив для оптимизатора (Углы в радианах, дистанция в метрах/мм)
    numeric_dataset = np.zeros((len(dataset_orig), 8), dtype=float)
    
    # "Истинный" физический ноль (с учетом сгенерированных ошибок робота)
    zero_pos_real = model.get_transition_matrix(model.zero_wire_angles, "real")[0:3, 3]
    
    for i, row in enumerate(dataset_orig):
        angles_deg = np.array(row[0:6], dtype=float)
        angles_rad = angles_deg * DEG
        
        # Вычисляем истинное расстояние с помощью real_dh
        [x, y, z] = model.get_transition_matrix(angles_rad, "real")[0:3, 3]
        dist_real = sqrt((x - zero_pos_real[0])**2 + (y - zero_pos_real[1])**2 + (z - zero_pos_real[2])**2)
        
        numeric_dataset[i, 0:6] = angles_rad
        numeric_dataset[i, 7] = dist_real
        
    return numeric_dataset

def _calibrate_batch_wrist(model, tries_count, generate):
    copy_model = copy.deepcopy(model)
    
    # Строгий список параметров для 4, 5 и 6 звеньев
    target_params = [
        "alpha_6", "a_6", "d_6", "theta_6",
        "a_5", "alpha_5", "theta_5", "d_5",
        "a_4", "alpha_4", "theta_4"
    ]
    
    local_param_errors = np.zeros(len(target_params))
    local_nom_param_errors = np.zeros(len(target_params))
    local_est_param_errors = np.zeros(len(target_params))

    for _ in range(tries_count):
        copy_model.reset_estimated_dh()
        
        if generate:
            new_dh = generate_real_dh(copy_model) # Твоя функция генерации ошибок
            copy_model.change_real_dh(new_dh)
            
        # Генерируем 50 точек и "измеряем" их
        dataset = measure_wrist_block_distances(copy_model, num_poses=50)

        # Вызываем нашу изолированную функцию оптимизации
        # Важно: В calibration_wire.py в функции least_squares лучше поставить verbose=0,
        # иначе при многопроцессорности терминал сойдет с ума от логов.
        res = calibrate_wrist_block(copy_model, dataset) 
        
        # Извлекаем и оцениваем параметры кисти
        nom_params = copy_model.get_readable_params("nominal")
        real_params = copy_model.get_readable_params("real")
        est_params = copy_model.get_readable_params("estimated")
        
        for i, p_name in enumerate(target_params):
            p_num, p_let = copy_model.params_from_letters(p_name)
            
            nominal_val = nom_params[p_num][p_let]
            real_val = real_params[p_num][p_let]
            est_val = est_params[p_num][p_let]
            
            start_error = nominal_val - real_val
            final_error = est_val - real_val
            
            local_nom_param_errors[i] += abs(start_error)
            local_est_param_errors[i] += abs(final_error)
            # Насколько мы улучшили параметр (если положительное число - стало лучше)
            local_param_errors[i] += (abs(start_error) - abs(final_error)) 
            
    local_param_errors /= tries_count
    local_nom_param_errors /= tries_count
    local_est_param_errors /= tries_count

    return local_param_errors, local_nom_param_errors, local_est_param_errors

def test_wrist_calibration_parallel(model, tries_num: int, generate: bool):
    num_cores = cpu_count()
    chunk_size = max(1, (tries_num // num_cores))

    print(f"Запуск {tries_num} тестов калибровки кисти на {num_cores} ядрах...")

    tasks = []
    for _ in range(min(num_cores, tries_num)):
        if chunk_size > 0:
            tasks.append((model, chunk_size, generate))        

    with Pool(processes=num_cores) as pool:
        results = pool.starmap(_calibrate_batch_wrist, tasks)

    # Агрегируем результаты
    params_error = np.array([x[0] for x in results])
    params_nom_error = np.array([x[1] for x in results])
    params_est_error = np.array([x[2] for x in results])

    params_error_median = np.sum(params_error, axis=0) / len(params_error)
    params_nom_error_median = np.sum(params_nom_error, axis=0) / len(params_error)
    params_est_error_median = np.sum(params_est_error, axis=0) / len(params_error)

    target_params = [
        "alpha_6", "a_6", "d_6", "theta_6",
        "a_5", "alpha_5", "theta_5", "d_5",
        "a_4", "alpha_4", "theta_4"
    ]

    print("\n=== Результаты симуляции: Ошибка параметров Кисти (До -> После) ===")
    for i, p_name in enumerate(target_params):
        nom_err = params_nom_error_median[i]
        est_err = params_est_error_median[i]
        improvement = params_error_median[i]
        
        # Переводим углы в градусы, а миллиметры оставляем для наглядности (если длины в мм)
        # Если модель в радианах и метрах, то:
        if 'alpha' in p_name or 'theta' in p_name:
            nom_err_disp = nom_err #/ DEG
            est_err_disp = est_err #/ DEG
            unit = "град"
        else:
            nom_err_disp = nom_err #* 1000 # перевод в мм
            est_err_disp = est_err #* 1000 # перевод в мм
            unit = "мм"
            
        print(f"{p_name:>8}: Ошибка до: {nom_err_disp:.4f} {unit} | Ошибка после: {est_err_disp:.4f} {unit} | Улучшение: {improvement if unit=='мм' else improvement:.4f}")


def main(args):
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    model = hayati_model.HayatiModel(config)
    if args.mode == "calibration":
        if args.is_real:
            dataset_orig = read_dataset(model.contribution_dataset, model.fieldnames_options["wire_contributions"])
            dataset_orig[:, 0:6] *= DEG
            #positive_mask = dataset_orig[:, 6] >= 0
            mask = np.array([isinstance(val, str) for val in dataset_orig[:, 6]])
            dataset = dataset_orig[mask]
            calibrate_wrist_block(model, dataset)
            write_results(model)
            write_validation_dataset(model, args.is_real)
            data = read_dataset("results/ARM95/validation_results_real.csv", ['q1','q2','q3','q4','q5','q6','d_nom','d_est','d_encoder','diff'])
            data = data[:-1, :]
            if args.draw:
                plot_calibration_errors_from_data(data)
        else:
            test_wrist_calibration_parallel(model, tries_num=200, generate=True)
           
            
   # elif args.mode == "rate":
        #rate_poses_contributions(model, args.i_target, args.num_of_tries, args.floor)

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

    args = parser.parse_args()
    main(args)
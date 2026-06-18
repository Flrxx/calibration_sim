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

def execute_global_calibration(model, dataset=[], is_real=0):
    """
    Выполняет глобальную оптимизацию для параметров, считываемых из 6-го столбца.
    """
    if len(dataset) == 0:
        is_real = 1
        # Предполагается наличие read_dataset
        dataset = read_dataset(model.calibration_dataset, model.fieldnames_options["wire_samples"]) 
        dataset[:, 0:6] *= DEG
        dataset[:, 7] /= 1000

    # Считываем уникальные параметры из 6-го столбца (индекс 6).
    # Использование dict.fromkeys сохраняет порядок их появления в датасете.
    #params_to_calibrate = model.error_list[:-2]
    params_to_calibrate = list(dict.fromkeys(dataset[:, 6]))

    print(f"Запуск глобальной калибровки для {len(params_to_calibrate)} параметров...")
    print(f"Параметры: {params_to_calibrate}")
    
    # Стандартная обработка применена
    res = calibrate_global(model, dataset, params_to_calibrate)
    
    # Оценка финальной невязки
    final_rmse = np.sqrt(np.mean(res.fun**2))
    print(f"Глобальная калибровка завершена. Сообщение: {res.message}")
    print(f"Конечная RMSE: {final_rmse * 1000:.4f} мм")
    
    return res

def calibrate_global(model, dataset: Union[np.ndarray, list], params_to_calibrate: list): 
    """
    Целевая функция глобальной оптимизации.
    """
    real_distances = dataset[:, 7]
    
    # Подготовка индексов и начальных значений
    x0 = []
    param_indices = []
    for param_name in params_to_calibrate:
        p_num, p_let = model.params_from_letters(param_name)
        x0.append(model.estimated_dh[p_num][p_let])
        param_indices.append((p_num, p_let))
    
    x0 = np.array(x0)

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
        verbose=2
    )
    
    # Фиксация итоговых значений в модели
    for i, (p_num, p_let) in enumerate(param_indices):
        model.estimated_dh[p_num][p_let] = res.x[i]
        
    return res



def main(args):
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    model = hayati_model.HayatiModel(config)
    if args.mode == "calibration":
        if args.is_real:
            execute_global_calibration(model)
            write_results(model)
            write_validation_dataset(model, args.is_real)
            data = read_dataset("results/ARM95/validation_results_real.csv", ['q1','q2','q3','q4','q5','q6','d_nom','d_est','d_encoder','diff'])
            data = data[:-1, :]
            if args.draw:
                plot_calibration_errors_from_data(data)
        #else:
            #make_many_calibration_attempts(model, args.num_of_tries, args.generate, args.draw)  # ,params_error
            
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
import numpy as np
import csv
import argparse
import json

import hayati_model
from csv_routines import read_dataset, write_dataset
from math_routines import DEG
from math import sqrt

# Импортируем функции расчета из актуального модуля
try:
    from dataset_generation_wire import (
        calculate_contribution, 
        calculate_jac_DH_params
    )
except ImportError:
    from pose_generation import calculate_contribution, calculate_jac_DH_params
    # На случай если старые модули всё еще используются, пытаемся достать новую функцию напрямую
    from new_dataset_generation import evaluate_snr_curvature_penalty

def calculate_derivatives(model, angles, eps=1e-5):
    """
    Вычисляет первую и вторую производные длины троса по каждому параметру.
    """
    first_deriv = np.zeros(len(model.error_list))
    second_deriv = np.zeros(len(model.error_list))
    
    # Центральная точка
    zero_pos = model.get_transition_matrix(angles, "nominal")[0:3, 3]
    L_center = np.linalg.norm(zero_pos - model.zero_offset_nominal)
    
    for i, p_name in enumerate(model.error_list):
        # Шаг вперед (+eps)
        model.vary_nominal_dh(p_name, eps)
        pos_plus = model.get_transition_matrix(angles, "nominal")[0:3, 3]
        L_plus = np.linalg.norm(pos_plus - model.zero_offset_nominal)
        model.vary_nominal_dh(p_name, -eps) # Возврат
        
        # Шаг назад (-eps)
        model.vary_nominal_dh(p_name, -eps)
        pos_minus = model.get_transition_matrix(angles, "nominal")[0:3, 3]
        L_minus = np.linalg.norm(pos_minus - model.zero_offset_nominal)
        model.vary_nominal_dh(p_name, eps) # Возврат
        
        # Первая производная (Градиент) - метод центральных разностей
        first_deriv[i] = (L_plus - L_minus) / (2 * eps)
        
        # Вторая производная (Кривизна / Гессиан)
        second_deriv[i] = (L_plus - 2 * L_center + L_minus) / (eps**2)
        
    return first_deriv, second_deriv

def evaluate_snr_curvature_penalty(model, angles, numerical_target_index, nonlinearity_penalty=0.5):
    """
    Оценка SNR с учетом штрафа за нелинейность (стабильность градиента).
    Высокий SNR получат только те точки, которые Левенберг-Марквардт сможет 
    реально оптимизировать без "сваливания" в абсурдные углы.
    """
    first_deriv, second_deriv = calculate_derivatives(model, angles)
    
    # 1. Полезный сигнал (Первая производная)
    target_signal = abs(first_deriv[numerical_target_index])
    
    # 2. Нестабильность целевого параметра (Вторая производная)
    target_curvature = abs(second_deriv[numerical_target_index])
    
    # 3. Кинематический шум
    noise_uncalib = np.sum(first_deriv[numerical_target_index + 1:]**2)
    noise_calib = np.sum((first_deriv[0:numerical_target_index]**2) / 50.0)
    total_noise = np.sqrt(noise_uncalib) + np.sqrt(noise_calib)
    
    # НОВЫЙ SNR: Штрафуем за кривизну
    robust_snr = target_signal / (total_noise + nonlinearity_penalty * target_curvature + 0.01)
    
    return robust_snr

def update_dataset_contributions(model: hayati_model.HayatiModel, input_filepath: str, output_filepath: str = None):
    """
    Считывает датасет, заново вычисляет векторы влияния (contributions) и SNR 
    (с учетом штрафа за кривизну) для каждой точки в соответствии с текущим 
    model.error_list и сохраняет обновленный датасет.
    
    :param model: Объект модели робота (HayatiModel)
    :param input_filepath: Путь к исходному CSV файлу датасета
    :param output_filepath: Путь для сохранения (если None, перезапишет исходный файл)
    """
    if output_filepath is None:
        output_filepath = input_filepath
        
    print(f"Чтение датасета из: {input_filepath}")
    
    # 1. Чтение сырых данных
    raw_data = read_dataset(input_filepath, model.fieldnames_options["wire_contributions"])
    # if not raw_data:
    #     print("Датасет пуст или файл не найден.")
    #     return []
        
    dataset_orig = np.array(raw_data, dtype=object)
    
    # 2. Фильтрация валидных строк (где 7-я колонка - это имя параметра)
    mask = np.array([isinstance(val, str) for val in dataset_orig[:, 6]])
    valid_data = dataset_orig[mask]
    
    print(f"Найдено {len(valid_data)} валидных точек.")
    print(f"Перерасчет векторов влияния и SNR (Hessian Penalty) для {len(model.error_list)} параметров...")
    
    # 3. Инициализация нулевых параметров
    if not hasattr(model, 'jac_DH_0') or model.jac_DH_0 is None:
        model.jac_DH_0 = calculate_jac_DH_params(model, model.zero_wire_angles)
        
    updated_dataset = []
    
    # 4. Построчный перерасчет
    for i in range(len(valid_data)):
        row = valid_data[i]
        
        # Извлекаем углы в градусах и переводим в радианы
        angles_deg = row[0:6].astype(float)
        angles_rad = angles_deg * DEG
        
        target_param = row[6]
        
        new_contrib = calculate_contribution(model, angles_rad)
        # Перерасчет SNR с использованием новой функции штрафа за кривизну
        if target_param in model.error_list:
            numerical_target_index = model.error_list.index(target_param)
            new_snr = -(abs(new_contrib[numerical_target_index]) /( sqrt(np.sum(new_contrib[numerical_target_index + 1:]**2)) + 0.01  + sqrt(np.sum(new_contrib[0:numerical_target_index]**2/50))    ) )
        else:
            print(f"Внимание: Параметр {target_param} не найден в error_list. Установлен SNR = 0.")
            new_snr = 0.0
            
        # Генерируем новый вектор влияния
        
        
        # Собираем новую строку
        new_row = angles_deg.tolist() + [target_param, float(new_snr)] + new_contrib.flatten().tolist()
        updated_dataset.append(new_row)
        
    # 5. Сохранение результата
    print(f"Сохранение обновленного датасета в: {output_filepath}")
    write_dataset(
        updated_dataset, 
        output_filepath, 
        model.fieldnames_options["wire_contributions"], 
        tolerance=3
    )
    
    print("Обновление успешно завершено!")
    return updated_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/config/ARM95_calibration.json", help="Путь к конфигу модели")
    parser.add_argument("--input", default="datasets/ARM95/wire/contribution/contribution_dataset.csv", help="Входной датасет")
    parser.add_argument("--output", default="datasets/ARM95/wire/contribution/contribution_dataset.csv", help="Выходной датасет")
    args = parser.parse_args()

    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    model = hayati_model.HayatiModel(config)
    
    # Инициализация базового Якобиана
    model.jac_DH_0 = calculate_jac_DH_params(model, model.zero_wire_angles)
    
    update_dataset_contributions(model, args.input, args.output)
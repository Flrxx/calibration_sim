import numpy as np
import json
import argparse
import csv
from math import acos

import hayati_model
from math_routines import DEG

def calculate_pose_metrics(model: hayati_model.HayatiModel, angles_rad: np.ndarray):
    """
    Вычисляет длину вытянутого троса, угол отклонения троса у датчика и угол перегиба на фланце.
    Логика полностью идентична функции is_pose_valid.
    
    Возвращает (длина_троса_мм, угол_у_датчика_град, угол_на_фланце_град)
    """
    # Получаем матрицу трансформации номинальной модели
    matrix = model.get_transition_matrix(angles_rad, "nominal")
    
    z_dir = matrix[0:3, 2] # Направление оси Z фланца
    wire = matrix[0:3, 3] - model.zero_offset_nominal # Вектор троса от датчика к фланцу
    wire_len = np.linalg.norm(wire)
    
    # Защита от деления на ноль (если трос полностью втянут)
    if wire_len < 1e-6:
        return 0.0, 0.0, 0.0
        
    wire_norm = wire / wire_len
    
    # 1. Длина троса в мм (предполагая, что базовая модель в метрах)
    wire_len_mm = wire_len * 1000.0
    
    # 2. Угол вытягивания троса (у датчика)
    scalar_wire_forward = np.vdot(wire_norm, model.zero_wire_direction)
    alpha_rad = acos(np.clip(scalar_wire_forward, -1.0, 1.0))
    wire_angle_deg = alpha_rad / DEG
    
    # 3. Угол между фланцем и тросом (перегиб на дюзе/креплении)
    # Используем -wire_norm, так как вектор должен идти ОТ фланца К датчику для замера угла
    scalar_z = np.vdot(-wire_norm, z_dir)
    alpha_z_rad = acos(np.clip(scalar_z, -1.0, 1.0))
    flange_angle_deg = alpha_z_rad / DEG
    
    return wire_len_mm, wire_angle_deg, flange_angle_deg

def process_dataset_metrics(model: hayati_model.HayatiModel, input_csv: str, output_csv: str):
    """
    Считывает датасет, рассчитывает метрики (длину и углы) для каждой строки и сохраняет в новый файл.
    """
    print(f"Чтение файла: {input_csv}...")
    
    rows = []
    headers = []
    
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader, None)
        for row in reader:
            if not row: continue
            rows.append(row)
            
    if headers is None:
        print("Ошибка: Пустой файл.")
        return
        
    # Добавляем новые заголовки
    new_headers = headers + ["wire_len_mm", "wire_angle_deg", "flange_angle_deg"]
    updated_rows = []
    
    print(f"Обработка {len(rows)} точек...")
    
    for i, row in enumerate(rows):
        # Если это строка со средними значениями или пустая, просто копируем и добавляем пустышки
        if row[0].upper() == "AVERAGE" or row[0] == "-1":
            updated_rows.append(row + ["-1", "-1", "-1"])
            continue
            
        try:
            # Извлекаем первые 6 значений как углы (предполагаем, что в CSV они в градусах)
            angles_deg = np.array([float(x) for x in row[0:6]])
            angles_rad = angles_deg * DEG
            
            # Считаем длину и углы
            wire_len_mm, wire_angle, flange_angle = calculate_pose_metrics(model, angles_rad)
            
            # Добавляем в конец строки (форматируем до 3 знаков после запятой)
            new_row = row + [f"{wire_len_mm:.3f}", f"{wire_angle:.3f}", f"{flange_angle:.3f}"]
            updated_rows.append(new_row)
            
        except ValueError:
            # Если не удалось спарсить (например, битая строка), оставляем как есть
            updated_rows.append(row + ["", "", ""])
            
    # Запись в новый файл
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(new_headers)
        writer.writerows(updated_rows)
        
    print(f"Готово! Результаты сохранены в: {output_csv}")

def main(args):
    # Инициализация модели
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    model = hayati_model.HayatiModel(config)
    
    process_dataset_metrics(model, args.input, args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Анализ длины и углов вытягивания троса в датасете")
    parser.add_argument("-c", "--config", default="src/config/ARM95_calibration.json", help="Путь к конфигу робота")
    parser.add_argument("-i", "--input", required=True, help="Путь к исходному CSV датасету")
    parser.add_argument("-o", "--output", required=True, help="Путь для сохранения нового CSV")
    args = parser.parse_args()
    main(args)
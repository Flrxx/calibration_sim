import numpy as np
import csv
from typing import Any
import os

def format_value(value: Any, tolerance: int = 8) -> str:
    """Вспомогательная функция для форматирования значений."""
    # Если это число (int, float, np.number), форматируем с точностью
    if isinstance(value, (int, float, np.number)):
        # Если нужно сохранять логику «ноль как пустая строка» (из вашего комментария):
        # if np.isclose(value, 0, atol=1e-4): return ""
        toleranse_str = "." + str(tolerance) + "f"
        return f"{value:{toleranse_str}}"
    
    # Если это строка или что-то другое, возвращаем как есть (приведя к str)
    return str(value) if value is not None else ""

def write_dataset(dataset: np.ndarray, filename: str, fieldnames: list[str], tolerance: str = ".8f"):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in dataset:
            writer.writerow({
                name: format_value(row[index], tolerance) 
                for index, name in enumerate(fieldnames)
            })

def add_line_csv(row: np.ndarray | list, filename: str, fieldnames: list[str], tolerance: int=8):
    file_exists = os.path.isfile(filename) and os.path.getsize(filename) > 0

    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)        
        if not file_exists:
            writer.writeheader()        

        formatted_row = {
            name: format_value(row[index], tolerance) 
            for index, name in enumerate(fieldnames)
        }
        
        writer.writerow(formatted_row)

def read_dataset(filename: str, fieldnames: list[str]) -> np.ndarray:
    rows = []
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Пытаемся конвертировать в числа то, что можно, остальное оставляем строками
            row_converted = []
            for field in fieldnames:
                val = row[field]
                try:
                    # Проверяем, целое ли число или float
                    if val.isdigit():
                        row_converted.append(int(val))
                    else:
                        row_converted.append(float(val))
                except (ValueError, TypeError):
                    row_converted.append(val)  # Оставляем строкой
            
            rows.append(row_converted)
            
    # dtype='object' позволяет массиву NumPy хранить одновременно и строки, и числа
    return np.array(rows, dtype=object)
    
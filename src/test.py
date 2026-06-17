import numpy as np
from csv_routines import read_dataset, write_dataset
import argparse
import json
import hayati_model

def replace_column_by_lookup(array: np.ndarray, i: int, replacement_list: list) -> np.ndarray:
    """
    Берет числа из i-го столбца как индексы и заменяет их 
    на соответствующие элементы из replacement_list.
    
    :param array: Исходный двумерный массив numpy
    :param i: Индекс столбца, в котором лежат числовые id
    :param replacement_list: Список строк/значений для подстановки
    :return: Новый массив с замененными данными
    """
    # 1. Приводим массив к типу object, чтобы он мог хранить строки
    modified_array = array.astype(object)
    
    # 2. Вытаскиваем i-й столбец и превращаем его в чистые целые числа (индексы)
    # Используем .round().astype(int), чтобы избежать проблем с float (например, 2.0 -> 2)
    indices = modified_array[:, i].astype(float).round().astype(int)
    
    # 3. Превращаем список подстановок в массив numpy типа object
    lookup_array = np.array(replacement_list, dtype=object)
    
    # 4. Магия NumPy: подставляем элементы по индексам «одним махом»
    modified_array[:, i] = lookup_array[indices]
    
    return modified_array

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    with open("src/config/ARM95_calibration.json", 'r') as config_file:
        config = json.load(config_file)
    model = hayati_model.HayatiModel(config)
    start_dataset = read_dataset("datasets/ARM95/wire/calibration_dataset.csv", ['q1','q2','q3','q4','q5','q6','index','d'])
    end_dataset = replace_column_by_lookup(start_dataset, 6, model.error_list)
    
    write_dataset(end_dataset, "datasets/ARM95/wire/calibration_dataset.csv", ['q1','q2','q3','q4','q5','q6','index','d'])
import numpy as np
from csv_routines import read_dataset, write_dataset
import argparse
import json
import hayati_model

def replace_column_by_lookup(array: np.ndarray, i: int, replacement_list: list) -> np.ndarray:
    """
    Заменяет неотрицательные числа в i-м столбце на элементы из списка по индексу.
    Отрицательные числа остаются без изменений.
    """
    # 1. Приводим массив к типу object для поддержки разных типов данных
    modified_array = array.astype(object)
    
    # 2. Получаем значения целевого столбца в виде чисел с плавающей точкой
    col_values = modified_array[:, i].astype(float)
    
    # 3. Создаем маску: True только для элементов >= 0
    mask = col_values >= 0
    
    # 4. Превращаем только неотрицательные значения в целые числа-индексы
    indices = col_values[mask].round().astype(int)
    
    # 5. Готовим список подстановок
    lookup_array = np.array(replacement_list, dtype=object)
    
    # 6. Магия NumPy: меняем значения только там, где mask == True
    modified_array[mask, i] = lookup_array[indices]
    
    return modified_array

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    with open("src/config/ARM95_calibration.json", 'r') as config_file:
        config = json.load(config_file)
    model = hayati_model.HayatiModel(config)
    start_dataset = read_dataset("datasets/ARM95/wire/contribution_dataset.csv", model.fieldnames_options["wire_contributions"])
    end_dataset = replace_column_by_lookup(start_dataset, 6, model.error_list)
    
    write_dataset(end_dataset, "datasets/ARM95/wire/contribution_dataset.csv", model.fieldnames_options["wire_contributions"], tolerance=".3f")
import numpy as np
import csv
from typing import Union
import os

def write_dataset(dataset: np.ndarray, filename: str, fieldnames: list[str], tolerance=".8f"):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in dataset:
            writer.writerow({name: f"{row[index]:{tolerance}}" for index, name in enumerate(fieldnames)})

def add_line_csv(row: np.ndarray, filename: str, fieldnames: list[str], tolerance=".8f"):
    file_exists = os.path.isfile(filename) and os.path.getsize(filename) > 0

    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)        
        if not file_exists:
            writer.writeheader()        
        
        # Logic: If row[index] is 0, use "", else format with tolerance
        # formatted_row = {
        #     name: (f"{row[index]:{tolerance}}" if not np.isclose(row[index], 0, atol=1e-4) else "") 
        #     for index, name in enumerate(fieldnames)
        # }

        formatted_row = {
            name: (f"{row[index]:{tolerance}}") for index, name in enumerate(fieldnames)
        }
        
        writer.writerow(formatted_row)

def read_dataset(filename: str, fieldnames: list[str]) -> Union[np.ndarray, np.ndarray]:
    dataset = np.array([], dtype='float').reshape(0, len(fieldnames))
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dataset = np.concatenate((dataset, np.array([float(row[field]) for field in fieldnames]).reshape(1, -1)), axis=0)
    return dataset
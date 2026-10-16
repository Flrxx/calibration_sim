import numpy as np
import json
import argparse
import csv
from math import acos

import hayati_model
from math_routines import DEG

def calculate_new_point(model_old: hayati_model.HayatiModel, model_new: hayati_model.HayatiModel):
    new_angles = model_new.inverse_kinematics([*model_old.zero_offset_nominal[:3], *[90 * DEG, 0 * DEG, 180 * DEG]]) / DEG
    return new_angles

def main(args=None):
    with open("src/config/ARM165_calibration.json", 'r') as config_file:
        config_old = json.load(config_file)
    model_old = hayati_model.HayatiModel(config_old)

    with open("src/config/ARM95_calibration_75_deg.json", 'r') as config_file:
        config_new = json.load(config_file)
    model_new = hayati_model.HayatiModel(config_new)

    print(model_old.zero_wire_angles / DEG)
    print(model_old.zero_offset_nominal)

    new_joint = calculate_new_point(model_old, model_new)
    print(new_joint)
    print(model_new.get_transition_matrix(new_joint * DEG, "nominal")[:3, 3])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Анализ длины и углов вытягивания троса в датасете")
    args = parser.parse_args()
    main(args)
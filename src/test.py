import hayati_model
import argparse
import json
from math import pi
from math_routines import extract_zyx_euler
import numpy as np
from math_routines import DEG

def main(args):
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    model_old = hayati_model.HayatiModel(config)

    matrix_old = model_old.get_transition_matrix(model_old.zero_wire_angles, "nominal")
    zero_wire_point = np.concatenate([matrix_old[0:3, 3], extract_zyx_euler(matrix_old[0:3, 0:3])])

    print(model_old.zero_wire_angles / DEG)
    print(zero_wire_point)

    with open("src/config/ARM95_calibration_45_deg.json", 'r') as config_file:
        config = json.load(config_file)
    model_new = hayati_model.HayatiModel(config)

    zero_wire_angles_new = model_new.inverse_kinematics(zero_wire_point)
    print(zero_wire_angles_new / DEG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-c", "--config", default="src/config/ARM95_calibration_VKR.json", help="Путь к конфигу робота")
    args = parser.parse_args()
    main(args)
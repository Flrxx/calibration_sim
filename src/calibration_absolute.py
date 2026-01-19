import numpy as np
from math import cos, sin, pi, sqrt, atan2, asin, log10, acos, copysign
from typing import Union
import robot_visualization 
import argparse
import json
import time
import dataset_generation
import hayati_model
from dataset_generation import FIELDNAMES_OPTIONS, read_dataset

MEASURABLE_PARAMS_MASK = np.array([0, 1, 2, 3, 4, 5], dtype='int')
TASK_SCALE = np.diag([1, 1, 1, 0, 0, 0])

def optimize(model: hayati_model.HayatiModel):
        measurable_params_mask = np.array([0, 1, 2, 3, 4, 5], dtype='int')
        dataset = read_dataset(model.dataset_file, FIELDNAMES_OPTIONS["random"])
        while model.norm > model.precision:
            jac, error_vec = full_jac(dataset)
            model.norm = np.linalg.norm(error_vec)
            # print(error_vec)
            if self.norm > self.prev_norm and self.optimization_method == "levenberg_marquardt" and self.lm_koef < 1e5:
                self.lm_koef *= 10
            elif self.norm < self.prev_norm and self.optimization_method == "levenberg_marquardt" and self.lm_koef > 1e-10:
                self.lm_koef /= 10
            self.prev_norm = self.norm

            new_jac, scale = self.remove_redundant_params(jac)
            if self.optimization_method == "gauss_newton":
                inc = lsq_linear(new_jac, error_vec).x
            elif self.optimization_method == "levenberg_marquardt":
                inc = np.linalg.inv((new_jac.T @ new_jac + self.lm_koef * np.eye(new_jac.shape[1], new_jac.shape[1]))) @ new_jac.T @ error_vec
            else:
                return
            
            self.update_params(inc, scale)
            self.display_metrics()
            val = input()
            try:
                new_val = float(val)
            except ValueError:
                if val == 'exit':
                    self.write_results(self.results_file)
                    return
                if val == 'test':
                    self.test_estimated_params()
                if val == "plot":
                    self.draw_plot(dataset)
                if val == "rm":
                    threshold = float(input())
                    dataset = self.remove_outliers(dataset, threshold)
            else:
                self.koef = new_val

def full_jac(model: hayati_model.HayatiModel, dataset, base_only=False, offsets_only=False):
        model.init_metrics()
        jac = np.array([], dtype='float').reshape(0, 36)
        error_vec = np.array([], dtype='float')
        error_vec_2 = np.zeros(88)
        for row in dataset:
            # estimated_pose = self.fk(row[:6], 'estimated')
            # real_pose = self.pose_from_measurement(row[6:9], row[9:])
            # cur_err = self.pose_error(real_pose, estimated_pose)[MEASURABLE_PARAMS_MASK]
            # print("New error vec: ", cur_err)
            # print("Diff mat:\n", real_pose - estimated_pose)
            estimated_pose = model.get_transition_matrix(row[:6], 'estimated')
            estimated_coordinates = np.concatenate((estimated_pose[:3, 3], model.extract_zyx_euler(estimated_pose[:3, :3])))
            real_coordinates = row[6:]
            cur_err = real_coordinates[MEASURABLE_PARAMS_MASK] - estimated_coordinates[MEASURABLE_PARAMS_MASK]
            # error_vec_2[i] = np.linalg.norm(cur_err[:3])
            # if abs(cur_err[0]) > OUTLIER_THRESHOLD or abs(cur_err[1]) > OUTLIER_THRESHOLD or abs(cur_err[2]) > OUTLIER_THRESHOLD:
            #     continue
            if base_only or offsets_only:
                jac = np.concatenate((jac, calibration_jacobian(row[:6], model.estimated_base_params,
                                                                     model.estimated_dh, model.estimated_tool_params)), axis=0)
                error_vec = np.concatenate((error_vec, cur_err), axis=0)

            else:
                jac = np.concatenate((jac, TASK_SCALE @ calibration_jacobian(row[:6], model.estimated_base_params,
                                                                                    model.estimated_dh, model.estimated_tool_params)), axis=0)
                error_vec = np.concatenate((error_vec, TASK_SCALE @ cur_err), axis=0)
            
            self.calculate_metrics(cur_err)
            # i += 1                              
        # print(error_vec_2) 
        return jac, error_vec

# Optimization routines
def calibration_jacobian(model: hayati_model.HayatiModel, angles: Union[np.ndarray, list], base: Union[np.ndarray, list],
                            dh: Union[np.ndarray, list], tool: Union[np.ndarray, list]) -> np.ndarray:
    tfs = model.get_transforms(angles, dh)
    main_tf, tool = model.get_base_tool_tf(base, tool)

    np_estimated_dh = np.array(dh)
    offsets = np_estimated_dh[:, 3]
    parallel_mask = np_estimated_dh[:, 4]

    frames_cnt = len(tfs) + 2

    from_base_trans = [main_tf.copy()] * frames_cnt
    from_tool_trans = [tool.copy()] * frames_cnt

    for i in range(len(tfs)):
        from_base_trans[i + 1] = from_base_trans[i] @ tfs[i]
        from_tool_trans[frames_cnt - 2 - i] = tfs[len(tfs) - 1 - i] @ from_tool_trans[frames_cnt - 1 - i]

    from_base_trans[-1] = from_base_trans[-2] @ tool
    from_tool_trans[0] = main_tf @ from_tool_trans[1]

    # Jac structure: [[tx_b, ty_b, tz_b, rx_b, ry_b, rz_b], [d_a, d_alpha, d_d/d_beta, d_theta_off] - len(tfs) times, [tx_t, ty_t, tz_t, rx_t, ry_t, rz_t]]
    # [tx_b, ty_b, tz_b, rx_b, ry_b, rz_b] always equals eye(6, 6)
    # [tx_t, ty_t, tz_t, rx_t, ry_t, rz_t] = / R  0 \ ,  R - rotation matrix of tool relative to base
    #                                        \ 0  R /
    # Vertical order: [tx, ty, tz, rx, ry, rz]
    
    jac = np.eye(6, len(tfs)*4 + 12)
    last_mat = np.zeros((6, 6))
    last_mat[:3, :3] = from_base_trans[-1][:3, :3]
    last_mat[3:6, 3:6] = from_base_trans[-1][:3, :3]
    jac[:, (6+(frames_cnt-2)*4):(12+(frames_cnt-2)*4)] = last_mat

    for i in range(1, len(tfs) + 1):
        jac_sub = np.zeros((6, 4))
        p_vec = from_base_trans[i - 1][0:3, 0:3] @ from_tool_trans[i][0:3, 3]
        p_vec_plus = from_base_trans[i][0:3, 0:3] @ from_tool_trans[i + 1][0:3, 3]
        xi_rot = (from_base_trans[i - 1][0:3, 0:3] @ self.z_rot(offsets[i - 1] + angles[i - 1])[0:3, 0:3])[0:3, 0]
        z_vec = from_base_trans[i - 1][0:3, 2]
        y_vec_plus = from_base_trans[i][0:3, 1]
        # a
        jac_sub[:3, 0] = xi_rot
        # alpha
        jac_sub[:3, 1] = np.cross(xi_rot, p_vec_plus)
        jac_sub[3:6, 1] = np.flip(xi_rot)
        # theta
        jac_sub[:3, 3] = np.cross(z_vec, p_vec)
        jac_sub[3:6, 3] = np.flip(z_vec)

        if parallel_mask[i - 1] == 0:
            # d
            jac_sub[:3, 2] = z_vec
        else:
            # beta
            jac_sub[:3, 2] = np.cross(y_vec_plus, p_vec_plus)
            jac_sub[3:6, 2] = np.flip(y_vec_plus)

        jac[:, (6+(i-1)*4):(10+(i-1)*4)] = jac_sub

    return jac[MEASURABLE_PARAMS_MASK, :]


def main(args):
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    model = hayati_model.HayatiModel(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Name of .json configuration file. Default: ARM95.json", default="ARM95.json")
    args = parser.parse_args()
    main(args)
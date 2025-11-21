import numpy as np
import matplotlib.pyplot as plt 
from math import cos, sin, pi, sqrt, atan2, asin, log10, acos, copysign
from typing import Union
from math_routines import x_rot, y_rot, z_rot, arbitrary_axis_rot, trans
from robotic_transformations import dh_trans, hayati_trans
import multiprocessing as mp

class HayatiModel:
    def __init__(self, config):
        self.optimization_method = config['optimization_method']
        self.dataset_file = config['dataset_file']
        self.base_circles_dataset_file = config['base_circles_dataset_file']
        self.tool_circles_dataset_file = config['tool_circles_dataset_file']
        self.results_file = config["results_file"]

         # DH params: [a, alpha, d/beta, theta_offset, parallel_axis]. Angle beta is used instead of d if axis is nearly parallel to the previous
        self.nominal_dh = config['nominal_dh']
        self.nominal_base_params = config['nominal_base_params']
        self.nominal_tool_params = config['nominal_tool_params']
        
        self.estimated_dh = self.nominal_dh.copy()
        self.estimated_base_params = self.nominal_base_params.copy()
        self.estimated_tool_params = self.nominal_tool_params.copy()
        
        self.real_dh = config['real_dh']
        self.real_base_params = config['real_base_params']
        self.real_tool_params = config['real_tool_params']

        self.angle_dist = self.linear_dist = self.base_dist = self.tool_dist = 0
        
        self.joint_limits_general_h = config["joint_limits_general_h"]
        self.joint_limits_general_l = config["joint_limits_general_l"]

        self.joint_limits_circle_h = config["joint_limits_circle_h"]
        self.joint_limits_circle_l = config["joint_limits_circle_l"]

        self.cartesian_limits = config["cartesian_limits"]
        self.max_z_angle = config["max_z_angle"]
        
        self.general_samples_number = config["general_samples_number"]
        self.circle_samples_number = config["circle_samples_number"]

        self.zero_tracker_position = config["zero_tracker_position"]

        self.measurable_params_mask = np.array([0, 1, 2, 3, 4, 5], dtype='int')

        self.identifiability_mask = np.ones(36, dtype='int')
        self.koef = 0.001
        self.lm_koef = 0.01
        self.norm = 10
        self.prev_norm = 0
        self.num_point = 0
    
    def get_transforms(self, angles: Union[np.ndarray, list], params: list) -> list:
        tfs = []
        for index, unit in enumerate(params):
            if self.nominal_dh[index][-1] == 0:
                tfs.append(dh_trans(unit, angles[index]))
            elif self.nominal_dh[index][-1] == 1:
                tfs.append(hayati_trans(unit, angles[index]))
        return tfs
    
    def get_base_tool_tf(self, base_params, tool_params):
        main_tf = trans(base_params[:3]) @ z_rot(base_params[3]) @ y_rot(base_params[4]) @ x_rot(base_params[5])
        tool = trans(tool_params[:3]) @ z_rot(tool_params[3]) @ y_rot(tool_params[4]) @ x_rot(tool_params[5])
        return main_tf, tool

    def get_transition_matrix(self, angles: Union[np.ndarray, list], type: str) -> np.ndarray:
        if type == 'estimated':
            params = self.estimated_dh
            tool_params = self.estimated_tool_params
            base_params = self.estimated_base_params
        elif type == 'nominal':
            params = self.nominal_dh
            tool_params = self.nominal_tool_params
            base_params = self.nominal_base_params
        elif type == 'real':
            params = self.real_dh
            tool_params = self.real_tool_params
            base_params = self.real_base_params
        else:
            raise ValueError("type must be 'nominal', 'real' or 'estimated'")
        main_tf, tool = self.get_base_tool_tf(base_params, tool_params)

        tfs = self.get_transforms(angles, params)
        for tf in tfs:
            main_tf = main_tf @ tf
        return main_tf @ tool
    
    def get_joint_coordinates_and_transition_matrix(self, angles: Union[np.ndarray, list], type: str) -> np.ndarray:
        if type == 'estimated':
            params = self.estimated_dh
            tool_params = self.estimated_tool_params
            base_params = self.estimated_base_params
        elif type == 'nominal':
            params = self.nominal_dh
            tool_params = self.nominal_tool_params
            base_params = self.nominal_base_params
        elif type == 'real':
            params = self.real_dh
            tool_params = self.real_tool_params
            base_params = self.real_base_params
        else:
            raise ValueError("type must be 'nominal', 'real' or 'estimated'")
        main_tf, tool = self.get_base_tool_tf(base_params, tool_params)
        result = {"coords": [], "transition_matrix": []}
        result["coords"].append(main_tf[:3, 3])

        tfs = self.get_transforms(angles, params)
        for i, tf in enumerate(tfs):
            main_tf = main_tf @ tf
            result["coords"].append(main_tf[:3, 3])
        
        result["transition_matrix"] = main_tf @ tool
        result["coords"].append([result["transition_matrix"][0][-1], result["transition_matrix"][1][-1], result["transition_matrix"][2][-1]])

        return result
    
    def satisfies_cartesian_limits(self, pose):
        position = pose[:3, 3]
        z_angle = acos(pose[:3, 2] @ [0, 0, 1])
        return (position[0] > self.cartesian_limits[0][0] and position[0] < self.cartesian_limits[0][1] and \
                position[1] > self.cartesian_limits[1][0] and position[1] < self.cartesian_limits[1][1] and \
                position[2] > self.cartesian_limits[2][0] and position[2] < self.cartesian_limits[2][1] and \
                abs(z_angle) < self.max_z_angle)

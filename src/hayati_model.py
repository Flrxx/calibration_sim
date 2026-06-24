import numpy as np
import matplotlib.pyplot as plt 
from math import cos, sin, pi, sqrt, atan2, asin, log10, acos, copysign
from typing import Union
from math_routines import x_rot, y_rot, z_rot, arbitrary_axis_rot, trans
from robotic_transformations import dh_trans, hayati_trans
import multiprocessing as mp
import copy
from math_routines import DEG
from scipy.optimize import minimize

np.set_printoptions(
    suppress=True, 
    formatter={'all': lambda x: f'{x:0.3f}' if isinstance(x, (int, float, np.number)) else str(x)}
)

class HayatiModel:
    def __init__(self, config):
        self.generation_output = config['generation_output']
        self.contribution_dataset = config['contribution_dataset']
        self.calibration_dataset = config['calibration_dataset']

        self.test_dataset_file = config['test_dataset_file']
        self.results_file = config["results_file"]

        # DH params: [a, alpha, d/beta, theta_offset, parallel_axis]. Angle beta is used instead of d if axis is nearly parallel to the previous
        self.nominal_dh = np.array(config['nominal_dh'])
        self.nominal_base_params = config['nominal_base_params']
        self.nominal_tool_params = config['nominal_tool_params']
        
        self.estimated_dh = copy.deepcopy(self.nominal_dh)
        self.estimated_base_params = copy.deepcopy(self.nominal_base_params)
        self.estimated_tool_params = copy.deepcopy(self.nominal_tool_params)
        
        self.real_dh = np.array(config['real_dh'])
        self.real_base_params = config['real_base_params']
        self.real_tool_params = config['real_tool_params']
        
        self.joint_limits_general_h = config["joint_limits_general_h"]
        self.joint_limits_general_l = config["joint_limits_general_l"]
        self.bounds = np.array([(self.joint_limits_general_l[i], self.joint_limits_general_h[i]) for i in range(6)])

        self.cartesian_limits = config["cartesian_limits"]
        self.max_z_angle = config["max_z_angle"]
        
        self.test_samples_number = config["test_samples_number"]

        self.jac_DH_0 = None
        self.zero_wire_angles = np.array(config["zero_wire_angles"]) * DEG
        self.zero_offset_nominal = self.get_transition_matrix(self.zero_wire_angles, "nominal")[0:3, 3]
        self.zero_offset_real = self.get_transition_matrix(self.zero_wire_angles, "real")[0:3, 3]
        
        zero_wire_vec = -(self.get_transition_matrix(self.zero_wire_angles, "nominal")[:3, :3] @ np.eye(3))[:, 2]
        norm = np.linalg.norm(zero_wire_vec)
        if norm > 0:
            self.zero_wire_direction = zero_wire_vec / norm
        else:
            self.zero_wire_direction = [None, None, None]

        self.encoder_abs_tolerance = config["encoder_abs_tolerance_percent"] / 100
        self.encoder_resolution = config["encoder_resolution_mm"] / 1000
        self.error_list = config["error_list"]
        
        self.angle_idexes = [i for i, e in enumerate(self.error_list) if any(sub in e for sub in ("alpha", "theta", "beta"))]
        self.fieldnames_options = {
            "wire_contributions" : ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'index', 'obj', *self.error_list],
            "wire_samples": ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'index', 'd'],
            "random_poses" : ['q1', 'q2', 'q3', 'q4', 'q5', 'q6'],
            "test_result": ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'd_nom', 'd_est', 'd_encoder', 'diff']
            
        }

        self.wire_limits = [config["wire_limits_l_mm"] / 1000, config["wire_limits_h_mm"] / 1000]
        self.angle_limit_z = config["angle_limit_z_deg"] * DEG
        self.angle_limit_wire = config["angle_limit_wire_deg"] * DEG

        self.distance_delta = config["distance_delta_mm"] / 1000
        self.angle_delta = config["angle_delta_deg"] * DEG
        
        self.a_delta = config["a_delta_mm"] / 1000
        self.alpha_delta = config["alpha_delta_deg"] * DEG
        self.d_delta = config["d_delta_mm"] / 1000
        self.theta_delta = config["theta_delta_deg"] * DEG
        self.beta_delta = config["beta_delta_deg"] * DEG

        self.ignore_error_list = config["ignore_errors_list"]
    
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
    
    def params_from_letters(self, parametr: str):
        letter, num = parametr.split("_")
        num = int(num) - 1
        if letter == 'a':
            i = 0
        elif letter == 'alpha':
            i = 1
        elif letter == 'd' or letter == 'beta':
            i = 2
        elif letter == 'theta':
            i = 3  
        return (num, i)     # 2, alpha
    
    def vary_nominal_dh(self, parametr: str, eps: float):
        (num, i) = self.params_from_letters(parametr)
        self.nominal_dh[num][i] += eps

    def reset_estimated_dh(self):
        self.estimated_dh = copy.deepcopy(self.nominal_dh)
    
    def change_real_dh(self, new_dh: list):
        self.real_dh = new_dh.copy()
        self.zero_offset_real = self.get_transition_matrix(self.zero_wire_angles, "real")[0:3, 3]
    
    def get_all_transition_matrixes(self, angles: Union[np.ndarray, list], type: str) -> np.ndarray:
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
        transition_matrixes = [main_tf]

        tfs = self.get_transforms(angles, params)
        for i, tf in enumerate(tfs):
            main_tf = main_tf @ tf
            transition_matrixes.append(main_tf)
        
        transition_matrixes.append(main_tf @ tool)
        return transition_matrixes
    
    def satisfies_cartesian_limits(self, pose):
        position = pose[:3, 3]
        z_angle = acos(pose[:3, 2] @ [0, 0, 1])
        return (position[0] > self.cartesian_limits[0][0] and position[0] < self.cartesian_limits[0][1] and \
                position[1] > self.cartesian_limits[1][0] and position[1] < self.cartesian_limits[1][1] and \
                position[2] > self.cartesian_limits[2][0] and position[2] < self.cartesian_limits[2][1] and \
                abs(z_angle) < self.max_z_angle)
    
    def satisfies_wire_limits(self, angles_in, bounds: Union[np.ndarray, list], angle_limit: float, wire_limits: Union[np.ndarray, list]):
        
        def objective(angles):
            return 0  #- max(np.concatenate([output[0:i_target], output[i_target + 1: border]]))**2
        cons = []

        def wire_cons(angles):    
            # z limits

            matrix = self.get_transition_matrix(angles, "nominal")
            z_dir = matrix[0:3, 2]
            scalar_z = np.vdot(z_dir, -self.zero_wire_direction)
            alpha_z = acos(scalar_z)
            angle_cons_z = np.array([alpha_z, angle_limit - alpha_z]) # so angle would fit

            wire = matrix[0:3, 3] - self.zero_offset_nominal
            wire_len = np.linalg.norm(wire) 
            scalar_wire = np.vdot(-(wire / wire_len), z_dir)
            alpha_wire = acos(scalar_wire)
            angle_cons_wire = np.array([alpha_wire, angle_limit - alpha_wire]) # so angle would fit
            
            # wire limits        
            len_cons = np.array([wire_len - wire_limits[0], wire_limits[1] - wire_len]) # so wire len would fit

            scalar = np.vdot(wire / wire_len, self.zero_wire_direction)  # > 0 so wire stretch forward        
            alpha = acos(scalar)
            angle_cons = np.array([alpha, angle_limit - alpha]) # so angle would fit

            return np.concatenate(([scalar_z], [scalar_wire], angle_cons_z, angle_cons_wire, [scalar], angle_cons, len_cons))

        cons.append({'type': 'ineq', 'fun': wire_cons})

        res = minimize(objective, angles_in, method='SLSQP', 
            bounds=bounds, constraints=cons, tol=1e-3)
        #return res.x
        if res.success:
            return(res.x)
        else:
            return([None for _ in range(6)])

    def get_readable_params(self, param_type: str):
        if param_type == 'estimated':
            res = copy.deepcopy(self.estimated_dh)
        elif param_type == 'nominal':
            res = copy.deepcopy(self.nominal_dh)
        elif param_type == 'real':
            res = copy.deepcopy(self.real_dh)
        else:
            raise ValueError("type must be 'nominal', 'real' or 'estimated'")

        mask = np.zeros(shape=self.nominal_dh.shape, dtype=bool)
        mask[:, [1, 3]] = True
        mask[1:3, [2]] = True
        res[mask] /= DEG

        mask = np.zeros(shape=self.nominal_dh.shape, dtype=bool)
        mask[:, [0, 2]] = True
        mask[1:3, [2]] = False
        res[mask] *= 1000   
        return res 
    
    # Metrics and validation routines
    def init_metrics(self):
        self.cond = None
        self.norm = None
        self.max_x_error = self.max_y_error = self.max_z_error = self.max_dist_error = 0
        self.min_x_error = self.min_y_error = self.min_z_error = self.min_dist_error = BIG_NUMBER

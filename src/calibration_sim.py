import numpy as np
import matplotlib.pyplot as plt 
from math import cos, sin, pi, sqrt, atan2, asin, log10, acos, copysign
from typing import Union
from math_routines import x_rot, y_rot, z_rot, arbitrary_axis_rot, trans
from robotic_transformations import dh_trans, hayati_trans
import plotting 
import pygame
import joystick 
import argparse
import json
import time
import ctypes
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
    

def test(args):
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    model = HayatiModel(config)

    running = mp.Value("i", 1)
    angles_nominal = mp.Array(ctypes.c_float, 6)
    angles_real = mp.Array(ctypes.c_float, 6)
    # for i in range(6):
    #     angles_nominal[i] = 0
    #     angles_real[i] = 0

    #joints_joysticks = joystick.JointJoysticks(model.joint_limits_general_h, model.joint_limits_general_l, "Joints")
    joystick_proc = mp.Process(target=plotting.joystick_process, args=(model.joint_limits_general_h, model.joint_limits_general_l, angles_nominal, angles_real, running))
    joystick_proc.start()

    plot_display = plotting.Optimized3DPlot()
    def on_key(event):
        if event.key == 'r':
            # Reset view
            plot_display.ax.view_init(elev=20, azim=45)
            plot_display.fig.canvas.draw()
        elif event.key == 'c':
            # Clear trajectories
            plot_display.clear_trajectories()
    
    plot_display.fig.canvas.mpl_connect('key_press_event', on_key)
    # Optimization: Update plot less frequently
    plot_update_interval = 0.016 * 2  # ~60 FPS
    last_update_time = time.time()

    while running:
        # joints_joysticks.draw_joint_joysticks()
        # joints_joysticks.clock.tick(60)

        # angles_nominal = joints_joysticks.get_nominal_joint_values()
        # angles_real = joints_joysticks.get_real_joint_values()

        nom_trans_m = model.get_transition_matrix(angles_nominal, "nominal")
        real_trans_m = model.get_transition_matrix(angles_real, "real")

        cart_nominal = [nom_trans_m[0][-1], nom_trans_m[1][-1], nom_trans_m[2][-1]]
        cart_real = [real_trans_m[0][-1], real_trans_m[1][-1], real_trans_m[2][-1]]

        print(f"Nominal: {cart_nominal}")
        print(f"Real: {cart_real}")

        #Throttle plot updates
        current_time = time.time()
        if current_time - last_update_time >= plot_update_interval:
                
            plot_display.update_plot_optimized(cart_nominal)
            last_update_time = current_time
            
            # # Check if plot window is closed
            # if not plt.fignum_exists(plot_display.fig.number):
            #     shared_data.running.value = 0
            #     break
                
            # Small sleep to prevent CPU spinning
            time.sleep(0.001)

    pygame.quit()


def main(args):
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    model = HayatiModel(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Name of .json configuration file. Default: ARM95.json", default="ARM95.json")
    parser.add_argument("-g", "--generate", help="Generate dataset for selected method. Default: false", type=bool, default=False)
    args = parser.parse_args()
    test(args)
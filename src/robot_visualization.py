import pygame
import argparse
import json
import sys
import multiprocessing as mp
from typing import Union
import time
from matplotlib import pyplot as plt
import numpy as np
from collections import deque
import joystick 
from ctypes import c_float
import hayati_model
#from dataset_generation_absolute import read_dataset, FIELDNAMES_OPTIONS
from dataset_generation_wire import FIELDNAMES_OPTIONS
from csv_routines import read_dataset

DEG = np.pi / 180

class ShowRobot:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.trajectory_max_points = 50  # Reduced for performance
        self.trajectory = deque(maxlen=self.trajectory_max_points)

        self.setup()
        
    def setup(self):
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y') 
        self.ax.set_zlabel('Z')
        
        # Set fixed limits
        self.ax.set_xlim([-0.8, 0.8])
        self.ax.set_ylim([-0.8, 0.8])
        self.ax.set_zlim([0, 1.1])
        #self.ax.set_box_aspect([1, 1, 1])
        
        # Disable some expensive features
        self.ax.grid(True, alpha=0.2)  # Lighter grid
        # Remove axis background for better performance
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False

        # Initialize plots for all points
        self.current_points = []
        self.connection_lines = []
        self.tool_axis = []

        # Add joint values text annotation (top-left)
        self.joint_text = self.fig.text(0.02, 0.8, '', transform=self.fig.transFigure, 
                                       fontsize=11, fontfamily='monospace', fontweight='bold',
                                       bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.9))

        self.axis_colors = ['red', 'green', 'blue']
        self.axis_labels = ['X', 'Y', 'Z']

        # Trajectory line
        self.trajectory_line, = self.ax.plot([], [], [], "purple", alpha=0.7, linewidth=1.0)

        for i in range(8):
            # Current points
            point = self.ax.scatter([0], [0], [0], c='blue', 
                                  s=80, marker='o', 
                                  label=f'Point {i+1}')
            self.current_points.append(point)
        
        # Connection lines between consecutive points
        for i in range(7):  # 7 connections for 8 points
            conn_line, = self.ax.plot([], [], [], 'blue', alpha=0.7, 
                                    linewidth=2.0,
                                    label=f'P{i+1}-P{i+2}' if i == 0 else "")
            self.connection_lines.append(conn_line)

        # Draw wire

        # Draw base frame
        original_axes = np.eye(3) * 0.2
        last_trans_matrix = np.eye(4)
        R = last_trans_matrix[:3, :3]
        rotated_axes = R @ original_axes
        for i in range(3):
            self.ax.quiver(0, 0, 0,
                    rotated_axes[0, i], rotated_axes[1, i], rotated_axes[2, i],
                    color=self.axis_colors[i], label=self.axis_labels[i], linewidth=2,
                    arrow_length_ratio=0.1)
        
        # Initial draw
        self.fig.canvas.draw()
        
    def update_robot(self, matrixes, joint_values):
        points_coords = [m[:3, 3] for m in matrixes]
        # for i in range(8):
        #     points_coords.append(matrixes[i][:3, 3])
        
        # Update trajectory lines
        self.trajectory.append(points_coords[-1])

        # Convert trajectory to numpy array for plotting
        if len(self.trajectory) > 1:
            traj_array = np.array(self.trajectory)
            self.trajectory_line.set_data(traj_array[:, 0], traj_array[:, 1])
            self.trajectory_line.set_3d_properties(traj_array[:, 2])
        else:
            # Clear the line if we don't have enough points
            self.trajectory_line.set_data([], [])
            self.trajectory_line.set_3d_properties([])
        
        # Update current points
        for i in range(8):
            x, y, z = points_coords[i]
            self.current_points[i]._offsets3d = ([x], [y], [z])
        
        # Update connection lines between consecutive points
        total_length = 0
        for i in range(7):
            x1, y1, z1 = points_coords[i]
            x2, y2, z2 = points_coords[i + 1]
            
            self.connection_lines[i].set_data([x1, x2], [y1, y2])
            self.connection_lines[i].set_3d_properties([z1, z2])
            
            # Calculate segment length
            segment_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
            total_length += segment_length
  
        # Draw each axis
        axis_length = 0.1
        original_axes = np.eye(3) * axis_length
        R = matrixes[-1][:3, :3]
        rotated_axes = R @ original_axes

        for axis in self.tool_axis:
            if axis in self.ax.collections:
                axis.remove()
        self.tool_axis.clear()
        for i in range(3):
            axis = self.ax.quiver(points_coords[-1][0], points_coords[-1][1], points_coords[-1][2],
                            rotated_axes[0, i], rotated_axes[1, i], rotated_axes[2, i],
                            color=self.axis_colors[i], label=self.axis_labels[i], linewidth=2,
                            arrow_length_ratio=0.5)
            self.tool_axis.append(axis)
                    
        if hasattr(self, 'last_title_update'):
            if time.time() - self.last_title_update > 0.2:  # Update title every 200ms
                self.ax.set_title(f'X: {points_coords[-1][0]:.3f}, Y: {points_coords[-1][1]:.3f}, Z: {points_coords[-1][2]:.3f}')
                self.last_title_update = time.time()
        else:
            self.ax.set_title(f'X: {points_coords[-1][0]:.3f}, Y: {points_coords[-1][1]:.3f}, Z: {points_coords[-1][2]:.3f}')
            self.last_title_update = time.time()

        joint_info = f"JOINT VALUES:\nJ1: {joint_values[0]:.3f}\nJ2: {joint_values[1]:.3f}\nJ3: {joint_values[2]:.3f}\nJ4: {joint_values[3]:.3f}\nJ5: {joint_values[4]:.3f}\nJ6: {joint_values[5]:.3f}"
        self.joint_text.set_text(joint_info)

        # Use blitting for faster updates
        try:
            self.fig.canvas.draw_idle()
        except:
            self.fig.canvas.draw()
        
        self.fig.canvas.flush_events()
    
    def clear_trajectory(self):
        self.trajectory.clear()
        self.trajectory_line.set_data([], [])
        self.trajectory_line.set_3d_properties([])
        self.fig.canvas.draw()
    
    def close(self):
        plt.close(self.fig)

    def is_open(self):
        return plt.fignum_exists(self.fig.number)

def visualize_pose(model: hayati_model.HayatiModel, plot: ShowRobot, angles_values: Union[np.ndarray, list], model_type):
    matrixes = model.get_all_transition_matrixes(angles_values, model_type)
    plot.update_robot(matrixes, angles_values)

def vizualize_forward_kinematics(model: hayati_model.HayatiModel, model_type="nominal"):
    running = mp.Value("i", 1)
    angles_values = mp.Array(c_float, 6)
   
    joystick_proc = mp.Process(target=joystick.joystick_process, args=(model.joint_limits_general_h, model.joint_limits_general_l, angles_values, running, model_type))
    joystick_proc.start()
    time.sleep(1)

    robot_display = ShowRobot()
    plot_update_interval = 0.0016/10 # more than 60 FPS
    last_update_time = time.time()
    while running: 
        current_time = time.time()
        if (current_time - last_update_time) >= plot_update_interval:
            visualize_pose(model, robot_display, angles_values, model_type)
            last_update_time = current_time

            # Check if plot window is closed
            if not robot_display.is_open():
                running.value = 0
                break

    joystick_proc.join()
    joystick_proc.close()
    robot_display.close()
    pygame.quit()

def on_key_press(event, counter, max):
    if event.key == 'n' or event.key == 'N':
        if(counter.value < max):
            counter.value += 1
    elif event.key == 'p' or event.key == 'P':
        if(counter.value > 0):
            counter.value -= 1

def visualize_dataset(model: hayati_model.HayatiModel, dataset, model_type):
    robot_display = ShowRobot()
    num = mp.Value('i', 0)
    robot_display.fig.canvas.mpl_connect('key_press_event', 
                                lambda event: on_key_press(event, num, len(dataset) - 1))
    
    plot_update_interval = 0.0016/10 # more than 60 FPS
    last_update_time = time.time()
    print("Press 'n' and 'p' to navigate through samples")
    while True:
        current_time = time.time()
        if (current_time - last_update_time) >= plot_update_interval:
            visualize_pose(model, robot_display, np.array(dataset[num.value][:6]) * DEG, model_type)
            last_update_time = current_time

            # Check if plot window is closed
            if not robot_display.is_open():
                return
    
def main(args):
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    model = hayati_model.HayatiModel(config)

    if args.visualization_type == "forward_kinematics":
        vizualize_forward_kinematics(model, args.model_type)
    elif args.visualization_type == "dataset":
        if ("circle" in args.dataset):
            dataset = read_dataset(args.dataset, FIELDNAMES_OPTIONS["circles"])
        else:
            dataset = read_dataset(args.dataset, FIELDNAMES_OPTIONS["wire"])
        visualize_dataset(model, dataset, args.model_type)
    else:
        raise ValueError("visualization_type should be 'forward_kinematics' or 'dataset'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Name of .json configuration file. Default: ARM95.json", default="ARM95.json")
    parser.add_argument("-m", "--model_type", help="Choose model to be visualized. Default: nominal", type=str, default="nominal")
    parser.add_argument("-t", "--visualization_type", help="Choose what you want to display: forward_kinematics or dataset. Default: forward_kinematics", type=str, default="forward_kinematics")
    parser.add_argument("-d", "--dataset", help="Choose dataset to display. Default: datasets/ARM95/data_ARM95.csv", type=str, default="datasets/ARM95/data_ARM95.csv")
    args = parser.parse_args()
    main(args)
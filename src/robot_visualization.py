import pygame
import sys
import multiprocessing as mp
from typing import Union
import time
from matplotlib import pyplot as plt
import numpy as np
from collections import deque
import matplotlib
import joystick 

class ShowRobot:
    def __init__(self, limits):
        plt.ion()
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.trajectory_max_points = 50  # Reduced for performance
        self.trajectory = deque(maxlen=self.trajectory_max_points)

        self.limits = limits 
        self.setup()
        
    def setup(self):
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y') 
        self.ax.set_zlabel('Z')
        
        # Set fixed limits
        self.ax.set_xlim(self.limits[0])
        self.ax.set_ylim(self.limits[1])
        self.ax.set_zlim(self.limits[2])
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

        # Trajectory line
        self.trajectory_line, = self.ax.plot([], [], [], "black", alpha=0.7, linewidth=1.5)

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
        
        # Add legend
        #self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Initial draw
        self.fig.canvas.draw()
        
    def update_robot(self, points_coords):
        
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
        
        if hasattr(self, 'last_title_update'):
            if time.time() - self.last_title_update > 0.2:  # Update title every 200ms
                self.ax.set_title(f'X: {points_coords[-1][0]:.3f}, Y: {points_coords[-1][1]:.3f}, Z: {points_coords[-1][2]:.3f}')
                self.last_title_update = time.time()
        else:
            self.ax.set_title(f'X: {points_coords[-1][0]:.3f}, Y: {points_coords[-1][1]:.3f}, Z: {points_coords[-1][2]:.3f}')
            self.last_title_update = time.time()
        
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
import pygame
import sys
import multiprocessing as mp
import time
from matplotlib import pyplot as plt
import numpy as np
from collections import deque
import matplotlib
import joystick 

def joystick_process(upper_limit: float, lower_limit: float, angles_nominal:list, angles_real:list, running: int):

    joints_joysticks = joystick.JointJoysticks(upper_limit, lower_limit, "Joints")
    while(running):
        joints_joysticks.draw_joint_joysticks()
        joints_joysticks.clock.tick(60)
        res = joints_joysticks.get_all_joystick_values()
        for i in range(3):
            angles_nominal[i] = res[i]
            angles_real[i] = res[i + 3]
        # angles_nominal = joints_joysticks.get_nominal_joint_values()
        # angles_real = joints_joysticks.get_real_joint_values()

class SharedData:
    def __init__(self):
        self.x = mp.Value('d', 0.0)
        self.y = mp.Value('d', 0.0) 
        self.z = mp.Value('d', 0.0)
        self.running = mp.Value('i', 1)

class Optimized3DPlot:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Use deque for trajectory - it's simpler and works correctly
        self.trajectory_max_points = 100
        self.trajectory = deque(maxlen=self.trajectory_max_points)
        
        self.setup_optimized_plot()
        
    def setup_optimized_plot(self):
        """Optimized plot setup"""
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y') 
        self.ax.set_zlabel('Z')
        
        # Set fixed limits to avoid recalculating
        self.ax.set_xlim([-120, 120])
        self.ax.set_ylim([-120, 120])
        self.ax.set_zlim([-120, 120])
        self.ax.set_box_aspect([1, 1, 1])
        
        # Disable some expensive features
        self.ax.grid(True, alpha=0.2)  # Lighter grid
        # Remove axis background for better performance
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        
        # Initialize plots with empty data
        self.trajectory_line, = self.ax.plot([], [], [], 'b-', alpha=0.7, linewidth=1.5)
        self.current_point = self.ax.scatter([0], [0], [0], c='red', s=80, marker='o')
        
        # Initial draw
        self.fig.canvas.draw()
        
    def update_plot_optimized(self, data: list):
        """Fixed trajectory plotting"""
        # Add new point to trajectory

        self.trajectory.append(data)
        
        # Convert trajectory to numpy array for plotting
        if len(self.trajectory) > 1:
            traj_array = np.array(self.trajectory)
            self.trajectory_line.set_data(traj_array[:, 0], traj_array[:, 1])
            self.trajectory_line.set_3d_properties(traj_array[:, 2])
        else:
            # Clear the line if we don't have enough points
            self.trajectory_line.set_data([], [])
            self.trajectory_line.set_3d_properties([])
        
        # Update current point
        self.current_point._offsets3d = (data)
        
        # Update title less frequently
        if hasattr(self, 'last_title_update'):
            if time.time() - self.last_title_update > 0.2:  # Update title every 200ms
                self.ax.set_title(f'X: {data[0]:.1f}, Y: {data[1]:.1f}, Z: {data[2]:.1f}')
                self.last_title_update = time.time()
        else:
            self.ax.set_title(f'X: {data[0]:.1f}, Y: {data[1]:.1f}, Z: {data[2]:.1f}')
            self.last_title_update = time.time()
        
        # Use blitting for faster updates
        try:
            self.fig.canvas.draw_idle()
        except:
            self.fig.canvas.draw()
        
        self.fig.canvas.flush_events()
    
    def clear_trajectory(self):
        """Clear the trajectory"""
        self.trajectory.clear()
        self.trajectory_line.set_data([], [])
        self.trajectory_line.set_3d_properties([])
        self.fig.canvas.draw()
    
    def close(self):
        plt.close(self.fig)

def main_optimized():
    """Optimized main process"""
    
    # Use shared memory instead of queue
    shared_data = SharedData()
    
    # Start joystick process
    joystick_proc = mp.Process(target=joystick_process, args=(shared_data,))
    joystick_proc.start()
    
    # Give PyGame time to start
    time.sleep(1)
    
    # Create optimized plot
    plot_3d = Optimized3DPlot()
    
    # Add interactive controls
    def on_key(event):
        if event.key == 'r':
            # Reset view
            plot_3d.ax.view_init(elev=20, azim=45)
            plot_3d.fig.canvas.draw()
        elif event.key == 'c':
            # Clear trajectory
            plot_3d.clear_trajectory()
    
    plot_3d.fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Optimization: Update plot less frequently
    plot_update_interval = 0.016  # ~60 FPS
    last_update_time = time.time()
    
    try:
        while shared_data.running.value:
            current_time = time.time()
            
            # Throttle plot updates
            if current_time - last_update_time >= plot_update_interval:
                x = shared_data.x.value
                y = shared_data.y.value
                z = shared_data.z.value
                
                plot_3d.update_plot_optimized(x, y, z)
                last_update_time = current_time
            
            # Check if plot window is closed
            if not plt.fignum_exists(plot_3d.fig.number):
                shared_data.running.value = 0
                break
                
            # Small sleep to prevent CPU spinning
            time.sleep(0.001)
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        shared_data.running.value = 0
        joystick_proc.join(timeout=2)
        if joystick_proc.is_alive():
            joystick_proc.terminate()
        plot_3d.close()
        print("Application closed.")

if __name__ == "__main__":
    mp.freeze_support()
    main_optimized()
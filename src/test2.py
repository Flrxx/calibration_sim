import pygame
import sys
import multiprocessing as mp
import time
from matplotlib import pyplot as plt
import numpy as np
from collections import deque
import matplotlib

class SharedData:
    def __init__(self):
        # Coordinates for 6 points
        self.coords = [mp.Value('d', 0.0) for _ in range(18)]  # 6 points * 3 coordinates
        self.running = mp.Value('i', 1)
    
    def get_point(self, point_idx):
        """Get coordinates for a specific point (0-5)"""
        if 0 <= point_idx < 6:
            base_idx = point_idx * 3
            return (self.coords[base_idx].value, 
                    self.coords[base_idx + 1].value, 
                    self.coords[base_idx + 2].value)
        return (0, 0, 0)
    
    def set_point(self, point_idx, x, y, z):
        """Set coordinates for a specific point (0-5)"""
        if 0 <= point_idx < 6:
            base_idx = point_idx * 3
            self.coords[base_idx].value = x
            self.coords[base_idx + 1].value = y
            self.coords[base_idx + 2].value = z

def joystick_process(shared_data):
    """PyGame joystick process for 18 joysticks (6 points)"""
    pygame.init()
    screen = pygame.display.set_mode((800, 900))  # Larger window for 18 joysticks
    pygame.display.set_caption("6-Point 3D Chain Control")
    clock = pygame.time.Clock()
    
    class OptimizedJoystick:
        def __init__(self, x, y, width, height, limits, joystick_id, label):
            self.rect = pygame.Rect(x, y, width, height)
            self.knob_rect = pygame.Rect(x, y, 20, height)
            self.limits = limits
            self.lower_limit = limits[0]
            self.upper_limit = limits[1]
            self.value = (self.lower_limit + self.upper_limit) / 2
            self.dragging = False
            self.joystick_id = joystick_id
            self.label = label
            self.update_knob_position_from_value()
            self.font = pygame.font.Font(None, 20)  # Smaller font for more labels
            
        def update_knob_position_from_value(self):
            value_range = self.upper_limit - self.lower_limit
            normalized_value = (self.value - self.lower_limit) / value_range
            knob_x = self.rect.left + (normalized_value * self.rect.width)
            self.knob_rect.centerx = knob_x
            
        def draw(self, surface):
            # Draw track
            pygame.draw.rect(surface, (100, 100, 100), self.rect)
            pygame.draw.rect(surface, (50, 50, 50), self.rect, 2)
            
            # Draw limits
            lower_text = self.font.render(f"{self.lower_limit:.0f}", True, (200, 200, 200))
            upper_text = self.font.render(f"{self.upper_limit:.0f}", True, (200, 200, 200))
            surface.blit(lower_text, (self.rect.left - 25, self.rect.centery - 8))
            surface.blit(upper_text, (self.rect.right + 5, self.rect.centery - 8))
            
            # Draw label
            id_text = self.font.render(f"{self.label}", True, (255, 255, 255))
            surface.blit(id_text, (self.rect.centerx - 15, self.rect.top - 20))
            
            # Draw knob
            pygame.draw.rect(surface, (200, 50, 50), self.knob_rect)
            pygame.draw.rect(surface, (255, 255, 255), self.knob_rect, 2)
            
            # Draw current value
            value_text = self.font.render(f"{self.value:.1f}", True, (255, 255, 0))
            surface.blit(value_text, (self.knob_rect.centerx - 15, self.knob_rect.top - 20))
            
        def handle_event(self, event):
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.knob_rect.collidepoint(event.pos):
                    self.dragging = True
            elif event.type == pygame.MOUSEBUTTONUP:
                self.dragging = False
            elif event.type == pygame.MOUSEMOTION and self.dragging:
                knob_x = max(self.rect.left, min(event.pos[0], self.rect.right))
                self.knob_rect.centerx = knob_x
                normalized_pos = (knob_x - self.rect.left) / self.rect.width
                self.value = self.lower_limit + (normalized_pos * (self.upper_limit - self.lower_limit))
                
        def get_value(self):
            return self.value
    
    # Create 18 joysticks - 3 for each of 6 points
    joystick_limits = [-100.0, 100.0]
    all_joysticks = []
    
    # Colors for different points
    point_colors = [
        (255, 100, 100),   # Point 1 - Red
        (255, 165, 0),     # Point 2 - Orange
        (255, 255, 0),     # Point 3 - Yellow
        (0, 255, 0),       # Point 4 - Green
        (0, 0, 255),       # Point 5 - Blue
        (128, 0, 128)      # Point 6 - Purple
    ]
    
    # Create joysticks in a grid layout
    for point_idx in range(6):
        point_color = point_colors[point_idx]
        
        # Point title
        title_font = pygame.font.Font(None, 24)
        
        for coord_idx in range(3):
            # Calculate position in grid
            col = point_idx % 2  # 0 or 1 for left/right column
            row = point_idx // 2 # 0, 1, 2 for rows
            
            x = 50 + col * 380
            y = 50 + row * 140 + coord_idx * 40
            
            coord_labels = ["X", "Y", "Z"]
            label = f"P{point_idx+1} {coord_labels[coord_idx]}"
            
            joystick = OptimizedJoystick(x, y, 150, 25, joystick_limits, 
                                       point_idx * 3 + coord_idx, label)
            all_joysticks.append(joystick)
    
    font = pygame.font.Font(None, 24)
    title_font = pygame.font.Font(None, 28)
    
    frame_counter = 0
    update_frequency = 2
    
    while shared_data.running.value:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                shared_data.running.value = 0
                break
            for joystick in all_joysticks:
                joystick.handle_event(event)
        
        if not shared_data.running.value:
            break
        
        screen.fill((30, 30, 30))
        
        # Draw titles for each point group
        for point_idx in range(6):
            col = point_idx % 2
            row = point_idx // 2
            x = 50 + col * 380
            y = 30 + row * 140
            
            title_text = title_font.render(f"Point {point_idx+1}", True, point_colors[point_idx])
            screen.blit(title_text, (x, y))
        
        # Draw all joysticks
        for joystick in all_joysticks:
            joystick.draw(screen)
        
        # Get current values and update shared data
        point_values = []
        for point_idx in range(6):
            point_vals = []
            for coord_idx in range(3):
                joystick_idx = point_idx * 3 + coord_idx
                point_vals.append(all_joysticks[joystick_idx].get_value())
            point_values.append(point_vals)
        
        # Update shared data less frequently
        frame_counter += 1
        if frame_counter >= update_frequency:
            for point_idx in range(6):
                shared_data.set_point(point_idx, *point_values[point_idx])
            frame_counter = 0
        
        # Display current coordinates for all points
        y_offset = 750
        for point_idx in range(6):
            x, y, z = point_values[point_idx]
            coord_text = font.render(
                f"P{point_idx+1}: ({x:6.1f}, {y:6.1f}, {z:6.1f})", 
                True, point_colors[point_idx]
            )
            screen.blit(coord_text, (50 + (point_idx % 3) * 250, y_offset + (point_idx // 3) * 30))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

class SixPoint3DPlot:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Trajectories for all points
        self.trajectory_max_points = 50  # Reduced for performance
        self.trajectories = [deque(maxlen=self.trajectory_max_points) for _ in range(6)]
        
        self.setup_optimized_plot()
        
    def setup_optimized_plot(self):
        """Setup the 6-point 3D plot"""
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y') 
        self.ax.set_zlabel('Z')
        
        # Set fixed limits
        self.ax.set_xlim([-120, 120])
        self.ax.set_ylim([-120, 120])
        self.ax.set_zlim([-120, 120])
        self.ax.set_box_aspect([1, 1, 1])
        
        # Disable some expensive features
        self.ax.grid(True, alpha=0.2)
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        
        # Colors for points
        self.point_colors = [
            'red',    # Point 1
            'orange', # Point 2
            'yellow', # Point 3
            'green',  # Point 4
            'blue',   # Point 5
            'purple'  # Point 6
        ]
        
        self.point_markers = ['o', 's', '^', 'D', 'v', '>']
        
        # Initialize plots for all points
        self.trajectory_lines = []
        self.current_points = []
        self.connection_lines = []
        
        for i in range(6):
            # Trajectory lines
            traj_line, = self.ax.plot([], [], [], color=self.point_colors[i], 
                                    alpha=0.5, linewidth=1.0, 
                                    label=f'P{i+1} Trajectory')
            self.trajectory_lines.append(traj_line)
            
            # Current points
            point = self.ax.scatter([0], [0], [0], c=self.point_colors[i], 
                                  s=80, marker=self.point_markers[i], 
                                  label=f'Point {i+1}')
            self.current_points.append(point)
        
        # Connection lines between consecutive points
        for i in range(5):  # 5 connections for 6 points
            conn_line, = self.ax.plot([], [], [], 'gray', alpha=0.7, 
                                    linewidth=2.0, linestyle='--',
                                    label=f'P{i+1}-P{i+2}' if i == 0 else "")
            self.connection_lines.append(conn_line)
        
        # Add legend
        self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Initial draw
        self.fig.canvas.draw()
        
    def update_plot_optimized(self, points_coords):
        """Update plot with all 6 points' positions"""
        # Update trajectories
        for i in range(6):
            self.trajectories[i].append(points_coords[i])
        
        # Update trajectory lines
        for i in range(6):
            if len(self.trajectories[i]) > 1:
                traj_array = np.array(self.trajectories[i])
                self.trajectory_lines[i].set_data(traj_array[:, 0], traj_array[:, 1])
                self.trajectory_lines[i].set_3d_properties(traj_array[:, 2])
            else:
                self.trajectory_lines[i].set_data([], [])
                self.trajectory_lines[i].set_3d_properties([])
        
        # Update current points
        for i in range(6):
            x, y, z = points_coords[i]
            self.current_points[i]._offsets3d = ([x], [y], [z])
        
        # Update connection lines between consecutive points
        total_length = 0
        for i in range(5):
            x1, y1, z1 = points_coords[i]
            x2, y2, z2 = points_coords[i + 1]
            
            self.connection_lines[i].set_data([x1, x2], [y1, y2])
            self.connection_lines[i].set_3d_properties([z1, z2])
            
            # Calculate segment length
            segment_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
            total_length += segment_length
        
        # Update title
        if hasattr(self, 'last_title_update'):
            if time.time() - self.last_title_update > 0.3:  # Update less frequently
                coord_text = " | ".join([f"P{i+1}({p[0]:.1f},{p[1]:.1f},{p[2]:.1f})" 
                                       for i, p in enumerate(points_coords)])
                self.ax.set_title(f'6-Point 3D Chain\n{coord_text}\nTotal Length: {total_length:.1f}', 
                                fontsize=10)
                self.last_title_update = time.time()
        else:
            coord_text = " | ".join([f"P{i+1}({p[0]:.1f},{p[1]:.1f},{p[2]:.1f})" 
                                   for i, p in enumerate(points_coords)])
            self.ax.set_title(f'6-Point 3D Chain\n{coord_text}\nTotal Length: {total_length:.1f}', 
                            fontsize=10)
            self.last_title_update = time.time()
        
        # Use blitting for faster updates
        try:
            self.fig.canvas.draw_idle()
        except:
            self.fig.canvas.draw()
        
        self.fig.canvas.flush_events()
    
    def clear_trajectories(self):
        """Clear all trajectories"""
        for trajectory in self.trajectories:
            trajectory.clear()
        
        for line in self.trajectory_lines:
            line.set_data([], [])
            line.set_3d_properties([])
        
        for line in self.connection_lines:
            line.set_data([], [])
            line.set_3d_properties([])
        
        self.fig.canvas.draw()
    
    def close(self):
        plt.close(self.fig)

def main_optimized():
    """Optimized main process for 6-point chain control"""
    print("Starting 6-POINT 3D CHAIN Visualization...")
    print("Controls: Drag joysticks in PyGame window | Rotate 3D view with mouse")
    print("Press 'r' to reset view | Press 'c' to clear trajectories")
    print("Points are connected in sequence: P1 → P2 → P3 → P4 → P5 → P6")
    
    # Use shared memory
    shared_data = SharedData()
    
    # Start joystick process
    joystick_proc = mp.Process(target=joystick_process, args=(shared_data,))
    joystick_proc.start()
    
    # Give PyGame time to start
    time.sleep(1)
    
    # Create 6-point plot
    plot_3d = SixPoint3DPlot()
    
    # Add interactive controls
    def on_key(event):
        if event.key == 'r':
            # Reset view
            plot_3d.ax.view_init(elev=20, azim=45)
            plot_3d.fig.canvas.draw()
        elif event.key == 'c':
            # Clear trajectories
            plot_3d.clear_trajectories()
    
    plot_3d.fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Optimization: Update plot less frequently
    plot_update_interval = 0.02  # ~50 FPS
    last_update_time = time.time()
    
    try:
        while shared_data.running.value:
            current_time = time.time()
            
            # Throttle plot updates
            if current_time - last_update_time >= plot_update_interval:
                # Get all points' coordinates
                points_coords = [shared_data.get_point(i) for i in range(6)]
                plot_3d.update_plot_optimized(points_coords)
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
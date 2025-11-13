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
        self.x = mp.Value('d', 0.0)
        self.y = mp.Value('d', 0.0) 
        self.z = mp.Value('d', 0.0)
        self.running = mp.Value('i', 1)

def joystick_process(shared_data):
    """Optimized PyGame joystick process"""
    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("3D Control Joysticks")
    clock = pygame.time.Clock()
    
    class OptimizedJoystick:
        def __init__(self, x, y, width, height, limits, joystick_id):
            self.rect = pygame.Rect(x, y, width, height)
            self.knob_rect = pygame.Rect(x, y, 20, height)
            self.limits = limits
            self.lower_limit = limits[0]
            self.upper_limit = limits[1]
            self.value = (self.lower_limit + self.upper_limit) / 2
            self.dragging = False
            self.joystick_id = joystick_id
            self.update_knob_position_from_value()
            self.font = pygame.font.Font(None, 24)  # Cache font
            
        def update_knob_position_from_value(self):
            value_range = self.upper_limit - self.lower_limit
            normalized_value = (self.value - self.lower_limit) / value_range
            knob_x = self.rect.left + (normalized_value * self.rect.width)
            self.knob_rect.centerx = knob_x
            
        def draw(self, surface):
            # Pre-render static elements (could be optimized further)
            pygame.draw.rect(surface, (100, 100, 100), self.rect)
            pygame.draw.rect(surface, (50, 50, 50), self.rect, 2)
            
            lower_text = self.font.render(f"{self.lower_limit:.1f}", True, (200, 200, 200))
            upper_text = self.font.render(f"{self.upper_limit:.1f}", True, (200, 200, 200))
            surface.blit(lower_text, (self.rect.left - 30, self.rect.centery - 10))
            surface.blit(upper_text, (self.rect.right + 5, self.rect.centery - 10))
            
            id_text = self.font.render(f"J{self.joystick_id+1}", True, (255, 255, 255))
            surface.blit(id_text, (self.rect.centerx - 10, self.rect.top - 25))
            
            pygame.draw.rect(surface, (200, 50, 50), self.knob_rect)
            pygame.draw.rect(surface, (255, 255, 255), self.knob_rect, 2)
            
            value_text = self.font.render(f"{self.value:.2f}", True, (255, 255, 0))
            surface.blit(value_text, (self.knob_rect.centerx - 20, self.knob_rect.top - 25))
            
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
    
    # Create joysticks
    joystick_limits = [[-100.0, 100.0]] * 3
    joysticks = []
    for i in range(3):
        x, y = 50, 50 + i * 80
        joysticks.append(OptimizedJoystick(x, y, 300, 30, joystick_limits[i], i))
    
    font = pygame.font.Font(None, 36)
    labels = ["X", "Y", "Z"]
    colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]
    
    # Pre-render static text
    value_texts = []
    for i in range(3):
        value_texts.append(font.render(f"{labels[i]}: 000.0", True, colors[i]))
    
    frame_counter = 0
    update_frequency = 2  # Update shared data every 2 frames
    
    while shared_data.running.value:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                shared_data.running.value = 0
                break
            for joystick in joysticks:
                joystick.handle_event(event)
        
        if not shared_data.running.value:
            break
        
        screen.fill((30, 30, 30))
        
        # Draw joysticks
        for joystick in joysticks:
            joystick.draw(screen)
        
        # Get current values
        current_values = [joystick.get_value() for joystick in joysticks]
        
        # Update shared data less frequently
        frame_counter += 1
        if frame_counter >= update_frequency:
            shared_data.x.value = current_values[0]
            shared_data.y.value = current_values[1] 
            shared_data.z.value = current_values[2]
            frame_counter = 0
        
        # Update display values
        y_offset = 250
        for i, value in enumerate(current_values):
            # Only update text when value changes significantly
            value_texts[i] = font.render(f"{labels[i]}: {value:.1f}", True, colors[i])
            screen.blit(value_texts[i], (50, y_offset + i * 25))
        
        pygame.display.flip()
        clock.tick(60)  # Limit to 60 FPS
    
    pygame.quit()

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
        
    def update_plot_optimized(self, x, y, z):
        """Fixed trajectory plotting"""
        # Add new point to trajectory
        self.trajectory.append([x, y, z])
        
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
        self.current_point._offsets3d = ([x], [y], [z])
        
        # Update title less frequently
        if hasattr(self, 'last_title_update'):
            if time.time() - self.last_title_update > 0.2:  # Update title every 200ms
                self.ax.set_title(f'X: {x:.1f}, Y: {y:.1f}, Z: {z:.1f}')
                self.last_title_update = time.time()
        else:
            self.ax.set_title(f'X: {x:.1f}, Y: {y:.1f}, Z: {z:.1f}')
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
    print("Starting OPTIMIZED 3D Visualization...")
    print("Controls: Drag joysticks in PyGame window | Rotate 3D view with mouse")
    print("Press 'r' to reset view | Press 'c' to clear trajectory")
    
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
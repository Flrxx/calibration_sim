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
        # Arrow direction vector (normalized)
        self.arrow_dir = [mp.Value('d', 0.0) for _ in range(3)]  # dx, dy, dz direction
        self.running = mp.Value('i', 1)
    
    def get_arrow_data(self):
        """Get arrow direction"""
        dir_vec = (self.arrow_dir[0].value, self.arrow_dir[1].value, self.arrow_dir[2].value)
        return dir_vec
    
    def set_arrow_data(self, dir_vec):
        """Set arrow direction"""
        self.arrow_dir[0].value, self.arrow_dir[1].value, self.arrow_dir[2].value = dir_vec

def joystick_process(shared_data):
    """PyGame joystick process for arrow direction control"""
    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("3D Arrow Direction Control")
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
            self.font = pygame.font.Font(None, 24)
            
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
            lower_text = self.font.render(f"{self.lower_limit:.1f}", True, (200, 200, 200))
            upper_text = self.font.render(f"{self.upper_limit:.1f}", True, (200, 200, 200))
            surface.blit(lower_text, (self.rect.left - 30, self.rect.centery - 10))
            surface.blit(upper_text, (self.rect.right + 5, self.rect.centery - 10))
            
            # Draw label
            id_text = self.font.render(f"{self.label}", True, (255, 255, 255))
            surface.blit(id_text, (self.rect.centerx - 15, self.rect.top - 25))
            
            # Draw knob
            pygame.draw.rect(surface, (200, 50, 50), self.knob_rect)
            pygame.draw.rect(surface, (255, 255, 255), self.knob_rect, 2)
            
            # Draw current value
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
    
    # Create 3 joysticks for arrow direction
    joystick_limits = [-1.0, 1.0]  # Normalized direction components
    arrow_joysticks = []
    arrow_labels = ["Direction X", "Direction Y", "Direction Z"]
    
    for i in range(3):
        x = 50
        y = 50 + i * 80
        joystick = OptimizedJoystick(x, y, 300, 30, joystick_limits, i, arrow_labels[i])
        arrow_joysticks.append(joystick)
    
    font = pygame.font.Font(None, 36)
    title_font = pygame.font.Font(None, 32)
    
    frame_counter = 0
    update_frequency = 2
    
    while shared_data.running.value:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                shared_data.running.value = 0
                break
            for joystick in arrow_joysticks:
                joystick.handle_event(event)
        
        if not shared_data.running.value:
            break
        
        screen.fill((30, 30, 30))
        
        # Draw title
        title_text = title_font.render("3D Arrow Direction Control", True, (255, 255, 255))
        screen.blit(title_text, (80, 15))
        
        # Draw arrow joysticks
        for joystick in arrow_joysticks:
            joystick.draw(screen)
        
        # Get current direction values
        arrow_dir = [joystick.get_value() for joystick in arrow_joysticks]
        
        # Normalize direction vector
        dir_magnitude = np.sqrt(sum(d**2 for d in arrow_dir))
        if dir_magnitude > 0:
            normalized_dir = [d / dir_magnitude for d in arrow_dir]
        else:
            normalized_dir = [0, 0, 0]
        
        # Update shared data less frequently
        frame_counter += 1
        if frame_counter >= update_frequency:
            shared_data.set_arrow_data(normalized_dir)
            frame_counter = 0
        
        # Display current direction
        dir_text = font.render(
            f"Direction: ({normalized_dir[0]:.3f}, {normalized_dir[1]:.3f}, {normalized_dir[2]:.3f})", 
            True, (255, 255, 0)
        )
        screen.blit(dir_text, (50, 250))
        
        # Display magnitude
        mag_text = font.render(f"Magnitude: {dir_magnitude:.3f}", True, (0, 255, 255))
        screen.blit(mag_text, (50, 280))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

class Arrow3DPlot:
    def __init__(self, x_limits=(-1.5, 1.5), y_limits=(-1.5, 1.5), z_limits=(-1.5, 1.5)):
        plt.ion()
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Store axis limits and calculate aspect ratios
        self.x_limits = x_limits
        self.y_limits = y_limits  
        self.z_limits = z_limits
        
        x_range = x_limits[1] - x_limits[0]
        y_range = y_limits[1] - y_limits[0]
        z_range = z_limits[1] - z_limits[0]
        max_range = max(x_range, y_range, z_range)
        self.aspect_ratios = [x_range/max_range, y_range/max_range, z_range/max_range]
        
        # Arrow visualization
        self.arrow_quiver = None
        self.arrow_start = [0, 0, 0]  # Fixed start position at origin
        self.arrow_length = 1.0  # Fixed length for normalized direction
        
        # Store direction history for trail
        self.direction_history = deque(maxlen=50)
        
        self.setup_optimized_plot()
        
    def setup_optimized_plot(self):
        """Setup the 3D arrow plot"""
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y') 
        self.ax.set_zlabel('Z')
        
        # Set custom axis limits
        self.ax.set_xlim(self.x_limits)
        self.ax.set_ylim(self.y_limits)
        self.ax.set_zlim(self.z_limits)
        self.ax.set_box_aspect(self.aspect_ratios)
        
        self.ax.grid(True, alpha=0.3)
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        
        # Draw coordinate system axes
        self.ax.quiver(0, 0, 0, 1, 0, 0, color='red', alpha=0.5, linewidth=2, label='X axis', arrow_length_ratio=0.1)
        self.ax.quiver(0, 0, 0, 0, 1, 0, color='green', alpha=0.5, linewidth=2, label='Y axis', arrow_length_ratio=0.1)
        self.ax.quiver(0, 0, 0, 0, 0, 1, color='blue', alpha=0.5, linewidth=2, label='Z axis', arrow_length_ratio=0.1)
        
        # Initialize arrow (will be created in update_plot)
        self.arrow_quiver = None
        
        # Initialize direction trail
        self.trail_line, = self.ax.plot([], [], [], 'cyan', alpha=0.3, linewidth=1, label='Direction History')
        
        self.ax.legend()
        
        # Initial draw
        self.fig.canvas.draw()
        
    def update_plot(self, arrow_dir):
        """Update plot with arrow direction"""
        # Add to direction history for trail
        if any(d != 0 for d in arrow_dir):
            self.direction_history.append(arrow_dir)
        
        # Update direction trail
        if len(self.direction_history) > 1:
            # Create trail showing previous directions
            trail_array = np.array(self.direction_history)
            # Scale directions for visualization
            scaled_trail = trail_array * 0.3  # Smaller trail for better visualization
            self.trail_line.set_data(scaled_trail[:, 0], scaled_trail[:, 1])
            self.trail_line.set_3d_properties(scaled_trail[:, 2])
        
        # Update arrow
        if any(d != 0 for d in arrow_dir):  # Only draw if direction is non-zero
            # Remove existing arrow if it exists
            if self.arrow_quiver is not None:
                self.arrow_quiver.remove()
            
            # Create new arrow
            x, y, z = self.arrow_start
            u, v, w = arrow_dir
            
            # Scale the arrow direction
            scale = self.arrow_length
            self.arrow_quiver = self.ax.quiver(x, y, z, u*scale, v*scale, w*scale, 
                                             color='yellow', alpha=0.9, linewidth=4,
                                             arrow_length_ratio=0.2, label='Current Direction')
        
        # Calculate spherical coordinates
        if any(d != 0 for d in arrow_dir):
            # Convert to spherical coordinates
            dx, dy, dz = arrow_dir
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            theta = np.arctan2(dy, dx)  # Azimuthal angle (in xy-plane)
            phi = np.arccos(dz / r) if r > 0 else 0  # Polar angle (from z-axis)
            
            # Convert to degrees
            theta_deg = np.degrees(theta)
            phi_deg = np.degrees(phi)
        else:
            theta_deg = 0
            phi_deg = 0
        
        # Update title
        if hasattr(self, 'last_title_update'):
            if time.time() - self.last_title_update > 0.2:
                self.ax.set_title(
                    f'3D Arrow Direction\n'
                    f'Direction: ({arrow_dir[0]:.3f}, {arrow_dir[1]:.3f}, {arrow_dir[2]:.3f})\n'
                    f'Azimuth: {theta_deg:.1f}° | Polar: {phi_deg:.1f}°',
                    fontweight='bold'
                )
                self.last_title_update = time.time()
        else:
            self.ax.set_title(
                f'3D Arrow Direction\n'
                f'Direction: ({arrow_dir[0]:.3f}, {arrow_dir[1]:.3f}, {arrow_dir[2]:.3f})\n'
                f'Azimuth: {theta_deg:.1f}° | Polar: {phi_deg:.1f}°',
                fontweight='bold'
            )
            self.last_title_update = time.time()
        
        # Update plot
        try:
            self.fig.canvas.draw_idle()
        except:
            self.fig.canvas.draw()
        
        self.fig.canvas.flush_events()
    
    def clear_history(self):
        """Clear direction history trail"""
        self.direction_history.clear()
        self.trail_line.set_data([], [])
        self.trail_line.set_3d_properties([])
        self.fig.canvas.draw()
    
    def close(self):
        plt.close(self.fig)
        
    def is_open(self):
        return plt.fignum_exists(self.fig.number)

def main_optimized():
    """Main process for arrow direction control"""
    print("Starting 3D Arrow Direction Control...")
    print("Controls: Drag joysticks to control arrow direction")
    print("Press 'r' to reset view | Press 'c' to clear direction history")
    print("Arrow starts at origin (0,0,0) with fixed length")
    
    # Use shared memory
    shared_data = SharedData()
    
    # Start joystick process
    joystick_proc = mp.Process(target=joystick_process, args=(shared_data,))
    joystick_proc.start()
    
    # Give PyGame time to start
    time.sleep(1)
    
    # Create arrow plot
    plot_3d = Arrow3DPlot(
        x_limits=(-1.2, 1.2),
        y_limits=(-1.2, 1.2), 
        z_limits=(-1.2, 1.2)
    )
    
    # Add interactive controls
    def on_key(event):
        if event.key == 'r':
            # Reset view
            plot_3d.ax.view_init(elev=20, azim=45)
            plot_3d.fig.canvas.draw()
        elif event.key == 'c':
            # Clear direction history
            plot_3d.clear_history()
    
    plot_3d.fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Optimization: Update plot less frequently
    plot_update_interval = 0.016  # ~60 FPS
    last_update_time = time.time()
    
    try:
        while shared_data.running.value:
            current_time = time.time()
            
            # Throttle plot updates
            if current_time - last_update_time >= plot_update_interval:
                # Get arrow direction
                arrow_dir = shared_data.get_arrow_data()
                plot_3d.update_plot(arrow_dir)
                last_update_time = current_time
            
            # Check if plot window is closed
            if not plot_3d.is_open():
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
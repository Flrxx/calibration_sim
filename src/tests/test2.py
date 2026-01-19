import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import time
from collections import deque
import threading
from queue import Queue

class RealTimePlotter:
    def __init__(self, max_points=100):
        self.max_points = max_points
        self.num_channels = 6
        self.data_queues = [deque(maxlen=max_points) for _ in range(self.num_channels)]
        self.time_points = deque(maxlen=max_points)
        
        # Setup the figure and axes
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 8))
        self.fig.suptitle('Real-Time 6-Channel Float Plotter', fontsize=16)
        
        # Flatten axes for easier iteration
        self.ax_list = self.axes.flatten()
        
        # Initialize plots
        self.lines = []
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
        channel_names = ['Channel 1', 'Channel 2', 'Channel 3', 
                        'Channel 4', 'Channel 5', 'Channel 6']
        
        for i in range(self.num_channels):
            line, = self.ax_list[i].plot([], [], color=colors[i], linewidth=2)
            self.lines.append(line)
            self.ax_list[i].set_title(channel_names[i])
            self.ax_list[i].set_xlabel('Time (s)')
            self.ax_list[i].set_ylabel('Value')
            self.ax_list[i].grid(True, alpha=0.3)
            self.ax_list[i].set_ylim(-1.5, 1.5)  # Initial y-limits
        
        # Adjust layout
        plt.tight_layout()
        
        # Data input queue
        self.input_queue = Queue()
        
        # Start the data generator in a separate thread
        self.running = True
        self.data_thread = threading.Thread(target=self.data_generator, daemon=True)
        self.data_thread.start()
    
    def data_generator(self):
        """Simulates data input at regular intervals"""
        # Modify this function to match your actual data source
        interval = 0.1  # seconds between data points
        while self.running:
            # Generate 6 random floats between -1 and 1
            # Replace this with your actual data input
            new_data = [random.uniform(-1, 1) for _ in range(6)]
            
            # Add timestamp
            current_time = time.time()
            
            # Put data in queue
            self.input_queue.put((current_time, new_data))
            
            # Wait for next data point
            time.sleep(interval)
    
    def get_real_data(self):
        """Override this method to get real data from your source"""
        # Example: reading from serial port, network, file, etc.
        # Return format: (timestamp, [float1, float2, float3, float4, float5, float6])
        current_time = time.time()
        simulated_data = [random.uniform(-1, 1) for _ in range(6)]
        return current_time, simulated_data
    
    def update_plot(self, frame):
        """Update function for animation"""
        # Process all available data points
        while not self.input_queue.empty():
            timestamp, values = self.input_queue.get()
            
            # Update time points
            self.time_points.append(timestamp)
            
            # Update data for each channel
            for i in range(self.num_channels):
                self.data_queues[i].append(values[i])
        
        # Update plot if we have data
        if len(self.time_points) > 0:
            # Calculate relative time (seconds from start)
            start_time = self.time_points[0]
            relative_times = [t - start_time for t in self.time_points]
            
            # Update each line
            for i in range(self.num_channels):
                self.lines[i].set_data(relative_times, list(self.data_queues[i]))
                
                # Auto-adjust x and y limits
                if len(self.data_queues[i]) > 0:
                    self.ax_list[i].relim()
                    self.ax_list[i].autoscale_view()
            
            # Add current values as text annotations
            if len(self.data_queues[0]) > 0:
                for i in range(self.num_channels):
                    current_val = self.data_queues[i][-1]
                    self.ax_list[i].set_title(f'Channel {i+1}: {current_val:.3f}')
        
        return self.lines
    
    def run(self):
        """Start the animation"""
        # Create animation
        self.ani = FuncAnimation(
            self.fig, 
            self.update_plot, 
            interval=50,  # Update plot every 50ms
            blit=True,
            cache_frame_data=False
        )
        
        # Show the plot
        plt.show()
    
    def stop(self):
        """Clean shutdown"""
        self.running = False
        plt.close('all')

def main():
    # Example 1: Using with simulated data (as shown above)
    print("Starting real-time plotter with simulated data...")
    print("Press Ctrl+C to exit")
    
    plotter = RealTimePlotter(max_points=200)
    
    try:
        plotter.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        plotter.stop()

def example_with_custom_data_source():
    """
    Example showing how to adapt the code for your specific data source
    """
    class CustomDataPlotter(RealTimePlotter):
        def __init__(self, max_points=100):
            super().__init__(max_points)
            
        def data_generator(self):
            """Replace this with your actual data acquisition"""
            # Example: Read from a file
            # with open('data.txt', 'r') as f:
            #     while self.running:
            #         line = f.readline()
            #         if line:
            #             values = [float(x) for x in line.strip().split(',')]
            #             if len(values) == 6:
            #                 self.input_queue.put((time.time(), values))
            #         time.sleep(0.1)
            
            # Example: Read from serial port
            # import serial
            # ser = serial.Serial('COM3', 9600)
            # while self.running:
            #     if ser.in_waiting:
            #         data = ser.readline().decode().strip()
            #         values = [float(x) for x in data.split(',')]
            #         if len(values) == 6:
            #             self.input_queue.put((time.time(), values))
            
            # For now, use parent's simulated data
            super().data_generator()
    
    plotter = CustomDataPlotter(max_points=150)
    plotter.run()

if __name__ == "__main__":
    main()
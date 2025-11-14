import pygame
import sys

class LinearJoystick:
    def __init__(self, x, y, width, height, limits, joystick_id):
        self.rect = pygame.Rect(x, y, width, height)
        self.knob_rect = pygame.Rect(x, y, 20, height)
        self.limits = limits  # [lower_limit, upper_limit]
        self.lower_limit = limits[1]
        self.upper_limit = limits[0]
        self.value = (self.lower_limit + self.upper_limit) / 2  # Start at middle
        self.dragging = False
        self.joystick_id = joystick_id
        
        # Calculate initial knob position based on starting value
        self.update_knob_position_from_value()
        
    def update_knob_position_from_value(self):
        # Convert value to knob position
        value_range = self.upper_limit - self.lower_limit
        normalized_value = (self.value - self.lower_limit) / value_range
        knob_x = self.rect.left + (normalized_value * self.rect.width)
        self.knob_rect.centerx = knob_x
        
    def draw(self, surface):
        # Draw track
        pygame.draw.rect(surface, (100, 100, 100), self.rect)
        pygame.draw.rect(surface, (50, 50, 50), self.rect, 2)
        
        # Draw limits labels
        font = pygame.font.Font(None, 24)
        lower_text = font.render(f"{self.lower_limit:.1f}", True, (200, 200, 200))
        upper_text = font.render(f"{self.upper_limit:.1f}", True, (200, 200, 200))
        surface.blit(lower_text, (self.rect.left - 55, self.rect.centery - 10))
        surface.blit(upper_text, (self.rect.right + 20, self.rect.centery - 10))
        
        # Draw ID label
        id_text = font.render(f"J{self.joystick_id+1}", True, (255, 255, 255))
        surface.blit(id_text, (self.rect.centerx - 10, self.rect.top - 55))
        
        # Draw knob
        pygame.draw.rect(surface, (200, 50, 50), self.knob_rect)
        pygame.draw.rect(surface, (255, 255, 255), self.knob_rect, 2)
        
        # Draw current value above knob
        value_text = font.render(f"{self.value:.2f}", True, (255, 255, 0))
        surface.blit(value_text, (self.knob_rect.centerx - 20, self.knob_rect.top - 25))
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.knob_rect.collidepoint(event.pos):
                self.dragging = True
                
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
            
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            # Calculate new knob position
            knob_x = max(self.rect.left, min(event.pos[0], self.rect.right))
            self.knob_rect.centerx = knob_x
            
            # Calculate value based on position and limits
            normalized_pos = (knob_x - self.rect.left) / self.rect.width
            self.value = self.lower_limit + (normalized_pos * (self.upper_limit - self.lower_limit))
            
    def get_value(self):
        return self.value

class JointJoysticks:
    def __init__(self, joysticks_limits_h: list, joysticks_limits_l: list, name: str):
        pygame.init()
        self.screen = pygame.display.set_mode((1800, 900))
        pygame.display.set_caption(name)
        self.clock = pygame.time.Clock()
        
        # Create 6 joysticks
        self.joysticks = []
        for i in range(6):
            x = 100
            y = 100 + i * 120
            width = 400
            height = 30
            joystick_i = LinearJoystick(x, y, width, height, [joysticks_limits_h[i], joysticks_limits_l[i]], i)
            if(i == 2 or i == 4):
                joystick_i.value = 1.57
                joystick_i.update_knob_position_from_value()
            self.joysticks.append(joystick_i)
        self.font = pygame.font.Font(None, 36)
        self.big_font = pygame.font.Font(None, 48)

        # Create another 6 joysticks
        for i in range(6):
            x = 900
            y = 100 + i * 120
            width = 400
            height = 30
            joystick_i = LinearJoystick(x, y, width, height, [joysticks_limits_h[i], joysticks_limits_l[i]], i)
            if(i == 2 or i == 4):
                joystick_i.value = 1.57
                joystick_i.update_knob_position_from_value()
            self.joysticks.append(joystick_i)
        self.font = pygame.font.Font(None, 36)
        
    def draw_labels(self, surface: pygame.Surface):
        id_text = self.big_font.render(f"Nominal", True, (255, 255, 255))
        surface.blit(id_text, (100, 25))
        id_text = self.big_font.render(f"Real", True, (255, 255, 255))
        surface.blit(id_text, (900, 25))
        
    def draw_joint_joysticks(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            for joystick in self.joysticks:
                joystick.handle_event(event)
        
        self.screen.fill((30, 30, 30))
        
        # Draw all joysticks
        self.draw_labels(self.screen)
        for joystick in self.joysticks:
            joystick.draw(self.screen)
        
        # Display all current values
        current_values = self.get_all_joystick_values()
        y_offset = 100
        x_offset = 0
        i_offset = 0
        for i, value in enumerate(current_values):
            value_text = self.font.render(f"Joint {i+1 - i_offset}: {value:.2f}", True, (255, 255, 255))
            self.screen.blit(value_text, (600 + x_offset, y_offset + (i - i_offset) * 30))
            if(i == 5):
                x_offset = 800
                i_offset = 6

        pygame.display.flip()


    def get_all_joystick_values(self):
        """Returns a list of current values from all joysticks"""
        return [joystick.get_value() for joystick in self.joysticks]
    
    def get_nominal_joint_values(self):
        """Returns a list of current values from all joysticks"""
        return [joystick.get_value() for joystick in self.joysticks[:6]]
    
    def get_real_joint_values(self):
        """Returns a list of current values from all joysticks"""
        return [joystick.get_value() for joystick in self.joysticks[6:]]

    def get_joystick_limits(self):
        """Returns a list of limits for all joysticks"""
        return [joystick.limits for joystick in self.joysticks]

def joystick_process(upper_limit: float, lower_limit: float, angles_nominal:list, angles_real:list, running: int):
    joints_joysticks = JointJoysticks(upper_limit, lower_limit, "Joints")
    while(running):
        joints_joysticks.draw_joint_joysticks()
        joints_joysticks.clock.tick(60)
        res = joints_joysticks.get_all_joystick_values()
        for i in range(6):
            angles_nominal[i] = res[i]
            angles_real[i] = res[i + 6]

# # Function that uses the joystick values
# def process_joystick_data(joysticks):
#     """Example function showing how to use the joystick values in your program"""
#     values = get_all_joystick_values(joysticks)
#     limits = get_joystick_limits(joysticks)
    
#     print("Current joystick values:")
#     for i, (value, limit) in enumerate(zip(values, limits)):
#         print(f"  J{i+1}: {value:.2f} (range: {limit[0]:.1f} to {limit[1]:.1f})")
    
#     # You can now use these values for whatever you need
#     return values

# Main program
# def main():
#     pygame.init()
#     screen = pygame.display.set_mode((900, 900))
#     pygame.display.set_caption("6 Linear Joysticks Demo")
#     clock = pygame.time.Clock()
    
#     # Define limits for each joystick: [lower_limit, upper_limit]
#     joystick_limits = [
#         [-1.0, 1.0],      # Joystick 1: -1 to 1
#         [0.0, 100.0],     # Joystick 2: 0 to 100
#         [-50.0, 50.0],    # Joystick 3: -50 to 50
#         [0.0, 1.0],       # Joystick 4: 0 to 1
#         [-10.0, 10.0],    # Joystick 5: -10 to 10
#         [0.0, 255.0]      # Joystick 6: 0 to 255
#     ]
    
#     # Create 6 joysticks
#     joysticks = []
#     for i in range(6):
#         x = 100
#         y = 100 + i * 120
#         width = 400
#         height = 30
#         joysticks.append(LinearJoystick(x, y, width, height, joystick_limits[i], i))
    
#     font = pygame.font.Font(None, 36)
    
#     running = True
#     while running:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
#             for joystick in joysticks:
#                 joystick.handle_event(event)
        
#         screen.fill((30, 30, 30))
        
#         # Draw all joysticks
#         for joystick in joysticks:
#             joystick.draw(screen)
        
#         # Display all current values
#         current_values = JointJoysticks.get_all_joystick_values(joysticks)
#         y_offset = 100
#         for i, value in enumerate(current_values):
#             value_text = font.render(f"Joint {i+1}: {value:.2f}", True, (255, 255, 255))
#             screen.blit(value_text, (600, y_offset + i * 30))
        
#         # Display limits info
#         # limits_text = font.render("Joystick Limits:", True, (200, 200, 255))
#         # screen.blit(limits_text, (500, 550))
        
#         # limits = get_joystick_limits(joysticks)
#         # for i, limit in enumerate(limits):
#         #     limit_text = font.render(f"J{i+1}: [{limit[0]:.1f}, {limit[1]:.1f}]", True, (200, 200, 255))
#         #     screen.blit(limit_text, (500, 580 + i * 30))
        
#         # Example of using the values in your program
#         # You can call get_all_joystick_values(joysticks) anywhere in your program
#         # to get the current values as a list
        
#         pygame.display.flip()
#         clock.tick(60)
    
#     pygame.quit()
#     sys.exit()

# if __name__ == "__main__":
#     main()
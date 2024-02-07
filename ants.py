import itertools
import numpy as np
import pygame
import pygame_gui
import math
import random
from environment import Swarm

k = 0.25


class AntSwarm(Swarm):
    def __init__(
        self,
        num_ants=20,
        grid_size=50,
        num_food_sources=5,
        cell_size=15,
        max_food_per_source=10,
    ):
        self.num_ants = num_ants
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.num_food_sources = num_food_sources
        self.max_food_per_source = max_food_per_source
        self.food_cell_size = 10  # Assuming this is used for drawing

        # Predefined positions for food sources, evenly spread
        self.predefined_food_positions = self.calculate_even_food_positions()

        self.init_environment()
        self.simulation_speed = 8
        self.food_cell_size = 10

        # Pygame initialization
        
        self.screen = pygame.display.set_mode(
            (self.grid_size * self.cell_size, self.grid_size * self.cell_size)
        )
        pygame.display.set_caption("Ant Swarm Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)

        self.ui_manager = pygame_gui.UIManager((300, 300))
        self.speed_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((150, 150), (100, 20)),
            start_value=self.simulation_speed,
            value_range=(1, 20),
            manager=self.ui_manager
        )
        

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
            if (
                event.type == pygame.USEREVENT
                and event.user_type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED
            ):
                if event.ui_element == self.speed_slider:
                    self.simulation_speed = int(self.speed_slider.get_current_value())
    def calculate_even_food_positions(self):
        # Define a maximum number of food sources you plan to support
        max_food_sources = 10  # Example value, adjust based on your grid size
        positions = []

        # Calculate even distribution across the grid
        rows = cols = int(math.sqrt(max_food_sources))
        row_spacing = self.grid_size // rows
        col_spacing = self.grid_size // cols

        for i, j in itertools.product(range(rows), range(cols)):
            x = (i * row_spacing + row_spacing // 2) % self.grid_size
            y = (j * col_spacing + col_spacing // 2) % self.grid_size
            positions.append((x, y))

        return positions[:max_food_sources]

    def init_environment(self):
        # Use predefined positions for the actual number of food sources
        self.food_sources = {
            self.predefined_food_positions[i]: random.randint(1, self.max_food_per_source)
            for i in range(min(self.num_food_sources, len(self.predefined_food_positions)))
        }
        self.nest_location = np.array([self.grid_size // 2, self.grid_size // 2])
        self.pheromone_trail = np.zeros((self.grid_size, self.grid_size))
        self.all_directions = np.array(
            [[0, 1], [1, 0], [0, -1], [-1, 0], [1, 1], [-1, 1], [1, -1], [-1, -1], [2, 1], [2, -1], [-2, 1], [-2, -1], [1, 2], [-1, 2], [1, -2], [-1, -2]]
        )  
        self.directions = np.array(
            [[0, 1], [1, 0], [0, -1], [-1, 0]]
            )# Up, Right, Down, Left
        self.ants = self.create_agent()
        self.food_collected = 0

    def reset(self):
        self.init_environment()

    def step(self, learning=True):
        if learning:
            for ant in self.ants:
                self.move_ant(ant)
        for ant in self.ants:
            if ant["has_food"] and np.array_equal(ant["position"], self.nest_location):
                ant["has_food"] = False
                self.food_collected += 1
        # Pheromone evaporation
        decay_factor = np.exp(-k)  # Calculate decay factor using your chosen k value
        threshold = (
            0.1  # Set a threshold below which pheromone levels are considered zero
        )

        # Apply exponential decay
        self.pheromone_trail *= decay_factor

        # Apply thresholding
        self.pheromone_trail[self.pheromone_trail < threshold] = 0
        self.diffuse_pheromones()
        
    def diffuse_pheromones(self):
        diffusion_probability = 0.4  # Adjust as needed
        diffusion_rate = 0.3  # Fraction of pheromone that diffuses
        direct_pheromone_threshold = 6  # Threshold to differentiate direct pheromones

        new_pheromone_trail = np.copy(self.pheromone_trail)

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.pheromone_trail[x, y] >= direct_pheromone_threshold and np.random.rand() < diffusion_probability:
                    diffused_amount = self.pheromone_trail[x, y] * diffusion_rate
                    new_pheromone_trail[x, y] -= diffused_amount
                
                    # Spread to adjacent cells
                    for dx, dy in self.all_directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                            new_pheromone_trail[nx, ny] += diffused_amount / 4

        self.pheromone_trail = new_pheromone_trail

    def move_ant(self, ant, action=0):
        """if ant["has_food"]:

        self.return_to_nest(ant)"""

        if action == 0:
            self.search_food(ant)

        elif action == 1:
            direction = self.choose_direction_based_on_pheromone(ant["position"])
            self.search_food(ant, direction)
        elif action == 2:
            self.drop_pheromone(ant)
            self.return_to_nest(ant)
        elif action == 3:
            self.return_to_nest(ant)

    def search_food(self, ant, move_direction=None):
        # Determine the probability of following the pheromone trail
        
        move_direction = np.random.choice(range(4))

        # Update ant's position
        new_position = ant["position"] + self.directions[move_direction]
        new_position = np.clip(new_position, 0, self.grid_size - 1)
        ant["position"] = new_position

        # Check for food at new position
        self.check_for_food_at_new_position(ant)

    def create_agent(self):
        return [
            {"position": self.nest_location, "has_food": False}
            for _ in range(self.num_ants)
        ]

    def get_nearest_food_distance(self, ant_position):
        min_distance = float("inf")
        for food_pos in self.food_sources:
            distance = np.linalg.norm(np.array(food_pos) - np.array(ant_position))
            if distance < min_distance:
                min_distance = distance

        if min_distance == float("inf"):
            # No food sources available, return a default high value
            return self.grid_size * math.sqrt(2)  # You can adjust this value as needed

        return min_distance

    def calculate_pheromone_follow_probability(self, position):
        # determine the probability of following pheromones based on the intensity of pheromones at the current position
        pheromone_level = self.pheromone_trail[tuple(position)]
        # probability increases with pheromone level
        max_pheromone_level = (
            self.pheromone_trail.max() + 1e-8
        )  # Add small constant to avoid division by zero
        return min(1.0, pheromone_level / max_pheromone_level)

    def choose_direction_based_on_pheromone(self, position):
        # Find the cell with the highest pheromone level in the environment
        max_pheromone_level = np.max(self.pheromone_trail)
        if max_pheromone_level == 0:
            # If there is no pheromone, choose a random direction
            return np.random.choice(range(4))
        
        # Get the position of the cell with the highest pheromone level
        max_pheromone_positions = np.argwhere(self.pheromone_trail == max_pheromone_level)
        
        # Choose the closest cell with the highest pheromone level
        ant_position = np.array(position)
        distances = np.linalg.norm(max_pheromone_positions - ant_position, axis=1)
        closest_max_pheromone_position = max_pheromone_positions[np.argmin(distances)]
        
        # Determine the direction to move towards the chosen pheromone cell
        direction_vector = closest_max_pheromone_position - ant_position
        direction_vector = np.clip(direction_vector, -1, 1)  # Limit to one step in any direction
        
        # Convert the direction vector to a direction index (0, 1, 2, 3)
        direction_mapping = {
            (0, 1): 0,  # Up
            (1, 0): 1,  # Right
            (0, -1): 2, # Down
            (-1, 0): 3  # Left
        }
        return direction_mapping.get(tuple(direction_vector), np.random.choice(range(4)))
        



    def check_for_food_at_new_position(self, ant):
        ant_pos = np.array([ant["position"][0] * self.cell_size + self.cell_size // 2, ant["position"][1] * self.cell_size + self.cell_size // 2])
        for food_position, food_quantity in list(self.food_sources.items()):
            food_source_center = np.array([food_position[0] * self.cell_size + self.cell_size // 2, food_position[1] * self.cell_size + self.cell_size // 2])
            distance = np.linalg.norm(ant_pos - food_source_center)
            
            # Assuming a fixed radius for simplicity, adjust based on your food drawing logic
            if distance <= self.cell_size * math.sqrt(food_quantity) / 2:
                ant["has_food"] = True
                self.food_sources[food_position] -= 1
                if self.food_sources[food_position] <= 0:
                    del self.food_sources[food_position]
                break

    def return_to_nest(self, ant):
        # Move towards the nest and leave a pheromone trail
        direction_to_nest = self.nest_location - ant["position"]
        move_direction = np.argmax(np.abs(direction_to_nest))
        step_direction = 1 if direction_to_nest[move_direction] > 0 else -1
        ant["position"][move_direction] += step_direction
        self.pheromone_trail[tuple(ant["position"])] += 15

    def drop_pheromone(self, ant):
        pher_intensity = random.randint(7, 15)
        self.pheromone_trail[tuple(ant["position"])] += pher_intensity

    def render(self):
        self.screen.fill((0, 0, 0))  # Dark background

        self.draw_pheromone_trails()
        self.draw_food_sources()
        self.draw_nest()
        self.draw_ants()
        self.display_info()

        pygame.display.flip()
        self.clock.tick(self.simulation_speed)

    def draw_pheromone_trails(self):
        # Assuming pheromone_trail is a 2D array with values indicating intensity
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                intensity = self.pheromone_trail[x, y]
                if intensity > 0:
                    # Light green to indicate trails
                    color = self.get_pheromone_color(intensity)
                    pygame.draw.rect(
                        self.screen,
                        color,
                        (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                    )

    def draw_food_sources(self):
        spacing = 10  # Space between food units

        for food_position, food_quantity in self.food_sources.items():
        # Define the center of the food source
            center_x = food_position[0] * self.cell_size
            center_y = food_position[1] * self.cell_size

            # Parameters for drawing
            food_unit_radius = max(2, self.food_cell_size)  # Radius of each food unit
            units_drawn = 0
            current_radius = 0  # Start with the smallest circle

            while units_drawn < food_quantity:
                # Calculate the circumference of the current circle
                circumference = 2 * math.pi * current_radius + 20

                # Determine how many units can fit around the current circle
                units_that_fit = int(circumference // spacing)

                if units_that_fit == 0:
                    pygame.draw.circle(
                        self.screen,
                        (0, 191, 255),
                        (int(center_x), int(center_y)),
                        food_unit_radius
                    )
                    units_drawn += 1
                    break  # No more units can fit in this layout

                # Calculate the angle between each unit
                angle_between_units = 2 * math.pi / units_that_fit

                for i in range(units_that_fit):
                    if units_drawn >= food_quantity:
                        break  # Stop if we've drawn all units

                    # Calculate unit position
                    angle = angle_between_units * i
                    unit_x = center_x + math.cos(angle) * current_radius
                    unit_y = center_y + math.sin(angle) * current_radius

                    # Draw food unit
                    pygame.draw.circle(
                        self.screen,
                        (0, 191, 255),
                        (int(unit_x), int(unit_y)),
                        food_unit_radius
                    )

                    units_drawn += 1

                current_radius += spacing  # Increase radius for the next circle of units


    def draw_nest(self):
        center_x = self.nest_location[0] * self.cell_size + self.cell_size // 2
        center_y = self.nest_location[1] * self.cell_size + self.cell_size // 2
        
        # Define hexagon points
        num_sides = 8
        radius = self.cell_size + 20
        points = [
            (center_x + math.cos(2 * math.pi / num_sides * i) * radius,
            center_y + math.sin(2 * math.pi / num_sides * i) * radius)
            for i in range(num_sides)
        ]
        
        # Draw hexagon
        pygame.draw.polygon(self.screen, (138, 43, 226), points)
        
        

    def draw_ants(self):
        for ant in self.ants:
            color = (255, 0, 0) if not ant['has_food'] else (255, 215, 0)  # gold for ants with food
            pygame.draw.circle(
                self.screen,
                color,
                (int(ant['position'][0] * self.cell_size + self.cell_size / 2), int(ant['position'][1] * self.cell_size + self.cell_size / 2)),
                self.cell_size / 3
            )

    def display_info(self):
        info_text = self.font.render(f"Food Collected: {self.food_collected}", True, (255, 255, 255))
        self.screen.blit(info_text, (5, 5))
    def get_pheromone_color(self, intensity):
        # Improved color gradient for pheromone trails
        max_val = self.pheromone_trail.max()
        return (0, 255 * intensity / max_val, 0)

    def close(self):
        pygame.quit()


# Usage example
if __name__ == "__main__":
    ant_swarm = AntSwarm()
    running = True
    episodes = 1000
    while running and episodes > 0:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        episodes -= 1

        ant_swarm.step()
        ant_swarm.render()
        # print(f'Episodes left: {episodes}')

    ant_swarm.close()

import numpy as np
import pygame
import random
from environment import Swarm


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
        self.init_environment()

        # Pygame initialization
        pygame.init()
        self.screen = pygame.display.set_mode(
            (grid_size * cell_size, grid_size * cell_size)
        )
        pygame.display.set_caption("Ant Swarm Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)

    def init_environment(self):
        self.food_sources = {
            tuple(np.random.randint(0, self.grid_size, 2)): random.randint(
                1, self.max_food_per_source
            )
            for _ in range(self.num_food_sources)
        }
        self.nest_location = np.array([self.grid_size // 2, self.grid_size // 2])
        self.pheromone_trail = np.zeros((self.grid_size, self.grid_size))
        self.directions = np.array(
            [[0, 1], [1, 0], [0, -1], [-1, 0]]
        )  # Up, Right, Down, Left
        self.ants = self.create_agent()
        self.food_collected = 0

    def reset(self):
        self.init_environment()

    def step(self):
        '''for ant in self.ants:
            self.move_ant(ant)'''
        for ant in self.ants:
            if ant["has_food"] and np.array_equal(ant["position"], self.nest_location):
                ant["has_food"] = False
                self.food_collected += 1
        self.pheromone_trail *= 0.97  # Pheromone evaporation
        self.diffuse_pheromones()

    def diffuse_pheromones(self):
        diffusion_probability = 0.1  # Adjust as needed
        diffusion_rate = 0.1  # Fraction of pheromone that diffuses
        direct_pheromone_threshold = 10  # Threshold to differentiate direct pheromones

        new_pheromone_trail = np.copy(self.pheromone_trail)

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.pheromone_trail[x, y] >= direct_pheromone_threshold:
                    if np.random.rand() < diffusion_probability:
                        diffused_amount = self.pheromone_trail[x, y] * diffusion_rate
                        new_pheromone_trail[x, y] -= diffused_amount

                        # Spread to adjacent cells
                        for dx, dy in self.directions:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                                new_pheromone_trail[nx, ny] += diffused_amount / 4

        self.pheromone_trail = new_pheromone_trail
    def move_ant(self, ant, action=0):
        '''if ant["has_food"]:
            
            self.return_to_nest(ant)'''
        
        
        if action == 0:
            self.search_food(ant)
        
        elif action == 1:
            direction = self.choose_direction_based_on_pheromone(ant["position"])
            self.search_food(ant, direction)
        elif action == 2:
            self.return_to_nest(ant)
            

    def search_food(self, ant, move_direction=None):
        # Determine the probability of following the pheromone trail
        '''follow_pheromone_prob = self.calculate_pheromone_follow_probability(
            ant["position"]
        )

        if np.random.uniform(0, 1) < follow_pheromone_prob:
            # Follow pheromone trail
            move_direction = self.choose_direction_based_on_pheromone(ant["position"])
        else:
            # Random walk'''
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
        min_distance = float('inf')
        for food_pos in self.food_sources:
            distance = np.linalg.norm(np.array(food_pos) - np.array(ant_position))
            if distance < min_distance:
                min_distance = distance

        if min_distance == float('inf'):
            # No food sources available, return a default high value
            return 100  # You can adjust this value as needed

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
        # Choose a direction based on the surrounding pheromone levels
        # Check pheromone levels in adjacent cells
        pheromone_levels = [
            self.pheromone_trail[
                tuple(np.clip(position + direction, 0, self.grid_size - 1))
            ]
            for direction in self.directions
        ]
        # Probabilistically choose a direction based on pheromone levels
        total_pheromone = sum(pheromone_levels)
        if total_pheromone > 0:
            probabilities = [level / total_pheromone for level in pheromone_levels]
            return np.random.choice(range(4), p=probabilities)
        else:
            return np.random.choice(range(4))

    def check_for_food_at_new_position(self, ant):
        for food_position in list(self.food_sources.keys()):
            if np.array_equal(food_position, ant["position"]):
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

        

    def render(self):
        self.screen.fill((250, 250, 210))  # Light background

        # Draw pheromone trails
        max_trail = np.max(self.pheromone_trail)
        if max_trail > 0:
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    intensity = self.pheromone_trail[x, y] / max_trail
                    if intensity > 0:
                        pygame.draw.rect(
                            self.screen,
                            (int(220*intensity), 220, 220, intensity * 255),
                            (
                                x * self.cell_size,
                                y * self.cell_size,
                                self.cell_size,
                                self.cell_size,
                            ),
                        )

        # Draw food sources

        for food_position, quantity in self.food_sources.items():
            food_pos_x, food_pos_y = food_position
            pygame.draw.rect(
                self.screen,
                (34, 139, 34),
                (
                    food_pos_x * self.cell_size,
                    food_pos_y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                ),
            )  # Dark green squares for food

        # Draw nest
        pygame.draw.rect(
            self.screen,
            (178, 34, 34),
            (*self.nest_location * self.cell_size, self.cell_size, self.cell_size),
        )  # Red square for nest

        # Draw ants
        for ant in self.ants:
            ant_color = (
                (139, 69, 19) if ant["has_food"] else (0, 0, 0)
            )  # Brown for ants with food, black otherwise
            pygame.draw.circle(
                self.screen,
                ant_color,
                ant["position"] * self.cell_size + self.cell_size // 2,
                self.cell_size // 3,
            )

        # Display info
        info_text = self.font.render(
            f"Food Collected: {self.food_collected}", True, (0, 0, 0)
        )
        self.screen.blit(info_text, (5, 5))

        pygame.display.flip()
        self.clock.tick(50)  # Control simulation speed

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

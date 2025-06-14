import numpy as np
import pygame
from PIL import Image
import math
import random
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import networkx as nx
from environment import SwarmEnvironment

class AntRole(Enum):
    WORKER = "worker"
    SCOUT = "scout"
    GUARD = "guard"
    LEADER = "leader"

@dataclass
class Ant:
    position: np.ndarray
    role: AntRole
    has_food: bool
    energy: float
    memory: List[Tuple[int, int]]
    communication_buffer: Optional[np.ndarray]
    id: int
    
    def __post_init__(self):
        self.max_energy = 100.0
        self.energy_consumption_rate = 0.1
        self.memory_capacity = 20
        
class DynamicFood:
    """Dynamic food source that can move or deplete"""
    def __init__(self, position, quantity, max_quantity=100):
        self.position = np.array(position)
        self.quantity = quantity
        self.max_quantity = max_quantity
        self.movement_pattern = random.choice(['static', 'drift', 'seasonal'])
        self.drift_velocity = np.random.randn(2) * 0.1
        self.depletion_rate = 0.95  # Food regenerates slowly
        
    def update(self, grid_size):
        # Move food sources
        if self.movement_pattern == 'drift':
            self.position += self.drift_velocity
            self.position = np.clip(self.position, 0, grid_size - 1)
        elif self.movement_pattern == 'seasonal':
            # Circular movement pattern
            angle = np.random.random() * 0.1
            rotation = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
            center = np.array([grid_size // 2, grid_size // 2])
            self.position = center + rotation @ (self.position - center)
            self.position = np.clip(self.position, 0, grid_size - 1)
            
        # Regenerate food slowly
        if self.quantity < self.max_quantity:
            self.quantity = min(self.quantity * self.depletion_rate + 1, self.max_quantity)

class Obstacle:
    """Environmental obstacles that ants must navigate around"""
    def __init__(self, position, size, obstacle_type='rock'):
        self.position = np.array(position)
        self.size = size
        self.type = obstacle_type
        self.passable = obstacle_type == 'water'  # Some obstacles might be passable with energy cost

class AntEnvironment(SwarmEnvironment):
    def __init__(self, num_actions, num_states, grid_size, num_ants, 
                 num_food_sources, max_food_per_source, 
                 enable_hierarchy=True, enable_communication=True,
                 enable_obstacles=True, dynamic_environment=True, **kwargs):
        
        self.num_actions = num_actions
        self.num_states = num_states
        self.grid_size = grid_size
        self.num_ants = num_ants
        self.num_food_sources = num_food_sources
        self.max_food_per_source = max_food_per_source
        
        # Advanced features
        self.enable_hierarchy = enable_hierarchy
        self.enable_communication = enable_communication
        self.enable_obstacles = enable_obstacles
        self.dynamic_environment = dynamic_environment
        
        # Multi-objective tracking
        self.objectives = {
            'food_collected': 0,
            'area_explored': set(),
            'energy_consumed': 0,
            'communication_efficiency': 0,
            'swarm_cohesion': 0
        }
        
        # Initialize environment
        self.reset()
        
        # Visualization
        self.visualize = kwargs.get('visualize', False)
        self.screen = None
        self.clock = None
        self.font = None
        self.latest_frame_image = None
        
        if self.visualize:
            pygame.font.init()
            self.screen = pygame.Surface(
                (self.grid_size * 15, self.grid_size * 15)
            )
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 16)
            
        # Graph for ant relationships
        self.ant_network = nx.Graph()
        
    def reset(self):
        """Reset environment to initial state"""
        # Initialize ant network
        self.ant_network = nx.Graph()
        
        # Initialize ants with roles
        self.ants = []
        for i in range(self.num_ants):
            if self.enable_hierarchy:
                if i == 0:
                    role = AntRole.LEADER
                elif i < self.num_ants * 0.2:
                    role = AntRole.SCOUT
                elif i < self.num_ants * 0.3:
                    role = AntRole.GUARD
                else:
                    role = AntRole.WORKER
            else:
                role = AntRole.WORKER
                
            ant = Ant(
                position=np.array([self.grid_size // 2, self.grid_size // 2]),
                role=role,
                has_food=False,
                energy=100.0,
                memory=[],
                communication_buffer=None,
                id=i
            )
            self.ants.append(ant)
            
        # Initialize dynamic food sources
        self.food_sources = []
        positions = self._compute_even_food_positions()
        for i in range(min(self.num_food_sources, len(positions))):
            food = DynamicFood(
                position=positions[i],
                quantity=random.randint(50, self.max_food_per_source),
                max_quantity=self.max_food_per_source
            )
            self.food_sources.append(food)
            
        # Initialize obstacles
        self.obstacles = []
        if self.enable_obstacles:
            n_obstacles = self.grid_size // 10
            for _ in range(n_obstacles):
                pos = np.random.randint(0, self.grid_size, size=2)
                # Avoid placing obstacles at nest
                while np.linalg.norm(pos - [self.grid_size//2, self.grid_size//2]) < 5:
                    pos = np.random.randint(0, self.grid_size, size=2)
                obstacle = Obstacle(
                    position=pos,
                    size=random.randint(2, 5),
                    obstacle_type=random.choice(['rock', 'water'])
                )
                self.obstacles.append(obstacle)
                
        # Initialize pheromone trails (multiple types)
        self.pheromone_trails = {
            'food': np.zeros((self.grid_size, self.grid_size)),
            'danger': np.zeros((self.grid_size, self.grid_size)),
            'exploration': np.zeros((self.grid_size, self.grid_size))
        }
        
        # Dynamic nest that can move in extreme conditions
        self.nest_location = np.array([self.grid_size // 2, self.grid_size // 2])
        self.nest_energy = 1000.0  # Nest has energy reserves
        
        # Reset objectives
        self.objectives = {
            'food_collected': 0,
            'area_explored': set(),
            'energy_consumed': 0,
            'communication_efficiency': 0,
            'swarm_cohesion': 0
        }
        
        # Reset ant network
        self.ant_network.clear()
        for ant in self.ants:
            self.ant_network.add_node(ant.id)
            
        return self.get_state()
    
    def _compute_even_food_positions(self):
        """Compute evenly distributed food positions"""
        positions = []
        grid_sections = int(np.sqrt(self.num_food_sources))
        section_size = self.grid_size // grid_sections
        
        for i in range(grid_sections):
            for j in range(grid_sections):
                if len(positions) >= self.num_food_sources:
                    break
                x = i * section_size + section_size // 2
                y = j * section_size + section_size // 2
                positions.append([x, y])
                
        return positions[:self.num_food_sources]
    
    def step(self, actions):
        """Execute one time step with given actions"""
        # Apply actions to ants
        for ant, action in zip(self.ants, actions):
            self.apply_action(ant, action)
            
        # Update dynamic environment
        if self.dynamic_environment:
            self._update_dynamic_elements()
            
        # Update pheromone trails
        self._update_pheromones()
        
        # Update ant relationships
        self._update_ant_network()
        
        # Calculate multi-objective rewards
        rewards = self._calculate_rewards()
        
        # Check termination conditions
        done = self._check_termination()
        
        # Update objectives
        self._update_objectives()
        
        return self.get_state(), rewards, done
    
    def apply_action(self, ant: Ant, action: int):
        """Apply action to ant with energy consumption"""
        # Consume energy for movement
        ant.energy -= ant.energy_consumption_rate
        
        # Role-specific energy consumption
        if ant.role == AntRole.SCOUT:
            ant.energy -= 0.05  # Scouts use less energy
        elif ant.role == AntRole.GUARD:
            ant.energy -= 0.15  # Guards use more energy
            
        # Check if ant has energy to move
        if ant.energy <= 0:
            return  # Ant is exhausted
            
        # Execute action based on ant's role
        if ant.role == AntRole.LEADER:
            self._leader_action(ant, action)
        elif ant.role == AntRole.SCOUT:
            self._scout_action(ant, action)
        elif ant.role == AntRole.GUARD:
            self._guard_action(ant, action)
        else:
            self._worker_action(ant, action)
            
        # Update ant's memory
        ant.memory.append(tuple(ant.position))
        if len(ant.memory) > ant.memory_capacity:
            ant.memory.pop(0)
            
    def _leader_action(self, ant: Ant, action: int):
        """Leader-specific actions including command broadcasting"""
        if action < 4:  # Basic movement
            self._move_ant(ant, action)
        elif action == 4:  # Broadcast food location
            self._broadcast_command(ant, 'food_location')
        elif action == 5:  # Rally call
            self._broadcast_command(ant, 'rally')
        elif action == 6:  # Danger warning
            self._broadcast_command(ant, 'danger')
        else:  # Establish new nest location
            self._broadcast_command(ant, 'new_nest')
            
    def _scout_action(self, ant: Ant, action: int):
        """Scout-specific actions with exploration bonuses"""
        self._move_ant(ant, action)
        
        # Scouts leave stronger exploration pheromones
        x, y = int(ant.position[0]), int(ant.position[1])
        self.pheromone_trails['exploration'][x, y] += 2.0
        
        # Scouts have better perception
        self._enhanced_perception(ant)
        
    def _guard_action(self, ant: Ant, action: int):
        """Guard-specific actions for nest protection"""
        # Guards stay near nest
        dist_to_nest = np.linalg.norm(ant.position - self.nest_location)
        if dist_to_nest > 5:
            # Move back towards nest
            direction = self.nest_location - ant.position
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            ant.position += direction
        else:
            self._move_ant(ant, action)
            
        # Guards mark danger areas
        if self._detect_threat(ant):
            x, y = int(ant.position[0]), int(ant.position[1])
            self.pheromone_trails['danger'][x, y] += 3.0
            
    def _worker_action(self, ant: Ant, action: int):
        """Standard worker actions"""
        self._move_ant(ant, action)
        
        # Check for food
        if not ant.has_food:
            self._check_for_food(ant)
        else:
            # Return to nest with food
            self._return_to_nest(ant)
            
    def _move_ant(self, ant: Ant, direction: int):
        """Move ant in given direction with obstacle avoidance"""
        movements = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
        new_position = ant.position + movements[direction]
        
        # Check boundaries
        new_position = np.clip(new_position, 0, self.grid_size - 1)
        
        # Check obstacles
        if not self._is_position_blocked(new_position, ant):
            ant.position = new_position
            
    def _is_position_blocked(self, position: np.ndarray, ant: Ant) -> bool:
        """Check if position is blocked by obstacles"""
        if not self.enable_obstacles:
            return False
            
        for obstacle in self.obstacles:
            dist = np.linalg.norm(position - obstacle.position)
            if dist < obstacle.size:
                if obstacle.type == 'water' and ant.energy > 20:
                    # Can pass water with extra energy cost
                    ant.energy -= 5
                    return False
                return True
        return False
        
    def _broadcast_command(self, leader: Ant, command_type: str):
        """Leader broadcasts command to nearby ants"""
        command_vectors = {
            'food_location': np.array([1, 0, 0, 0]),
            'rally': np.array([0, 1, 0, 0]),
            'danger': np.array([0, 0, 1, 0]),
            'new_nest': np.array([0, 0, 0, 1])
        }
        
        for ant in self.ants:
            if ant.id != leader.id:
                dist = np.linalg.norm(ant.position - leader.position)
                if dist < 10:  # Communication range
                    ant.communication_buffer = command_vectors[command_type]
                    
    def _enhanced_perception(self, scout: Ant):
        """Scouts can detect distant food sources"""
        perception_range = 15
        for food in self.food_sources:
            dist = np.linalg.norm(scout.position - food.position)
            if dist < perception_range:
                # Mark food trail
                direction = food.position - scout.position
                steps = int(dist)
                for i in range(steps):
                    pos = scout.position + direction * (i / steps)
                    x, y = int(pos[0]), int(pos[1])
                    if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                        self.pheromone_trails['food'][x, y] += 0.5
                        
    def _detect_threat(self, guard: Ant) -> bool:
        """Guards detect threats (low energy ants, depleted areas)"""
        threat_detected = False
        
        # Check for exhausted ants
        for ant in self.ants:
            if ant.id != guard.id:
                dist = np.linalg.norm(ant.position - guard.position)
                if dist < 5 and ant.energy < 20:
                    threat_detected = True
                    break
                    
        return threat_detected
        
    def _check_for_food(self, ant: Ant):
        """Check if ant found food"""
        for food in self.food_sources:
            dist = np.linalg.norm(ant.position - food.position)
            if dist < 2 and food.quantity > 0:
                ant.has_food = True
                food.quantity -= 1
                # Drop food pheromone
                x, y = int(ant.position[0]), int(ant.position[1])
                self.pheromone_trails['food'][x, y] += 5.0
                break
                
    def _return_to_nest(self, ant: Ant):
        """Ant returns to nest with food"""
        direction = self.nest_location - ant.position
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
            ant.position += direction
            
        # Drop pheromone trail
        x, y = int(ant.position[0]), int(ant.position[1])
        self.pheromone_trails['food'][x, y] += 3.0
        
        # Check if reached nest
        if np.linalg.norm(ant.position - self.nest_location) < 2:
            ant.has_food = False
            self.objectives['food_collected'] += 1
            self.nest_energy += 10
            
    def _update_dynamic_elements(self):
        """Update dynamic environment elements"""
        # Update food sources
        for food in self.food_sources:
            food.update(self.grid_size)
            
        # Potentially move nest if under threat
        if self.nest_energy < 200:  # Low energy
            # Find safest location
            safety_scores = np.zeros((self.grid_size, self.grid_size))
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    safety_scores[x, y] = -self.pheromone_trails['danger'][x, y]
                    
            # Move nest gradually to safer location
            safe_pos = np.unravel_index(np.argmax(safety_scores), safety_scores.shape)
            direction = np.array(safe_pos) - self.nest_location
            self.nest_location += direction * 0.1
            self.nest_location = np.clip(self.nest_location, 0, self.grid_size - 1)
            
    def _update_pheromones(self):
        """Update all pheromone trails with evaporation and diffusion"""
        evaporation_rates = {
            'food': 0.95,
            'danger': 0.90,
            'exploration': 0.98
        }
        
        for ptype, trail in self.pheromone_trails.items():
            # Evaporation
            trail *= evaporation_rates[ptype]
            
            # Diffusion
            kernel = np.array([[0.05, 0.1, 0.05],
                              [0.1, 0.4, 0.1],
                              [0.05, 0.1, 0.05]])
            
            # Simple convolution for diffusion
            new_trail = np.zeros_like(trail)
            for i in range(1, self.grid_size - 1):
                for j in range(1, self.grid_size - 1):
                    new_trail[i, j] = np.sum(trail[i-1:i+2, j-1:j+2] * kernel)
                    
            self.pheromone_trails[ptype] = new_trail
            
    def _update_ant_network(self):
        """Update graph of ant relationships"""
        # Clear existing edges
        self.ant_network.clear_edges()
        
        # Add edges based on proximity and communication
        for i, ant1 in enumerate(self.ants):
            for j, ant2 in enumerate(self.ants[i+1:], i+1):
                dist = np.linalg.norm(ant1.position - ant2.position)
                if dist < 5:  # Close proximity
                    self.ant_network.add_edge(ant1.id, ant2.id, weight=1/dist)
                    
    def _calculate_rewards(self):
        """Calculate multi-objective rewards for each ant"""
        rewards = []
        
        for ant in self.ants:
            reward = 0
            
            # Food collection reward
            if ant.has_food:
                reward += 10
                
            # Exploration reward
            pos_tuple = (int(ant.position[0]), int(ant.position[1]))
            if pos_tuple not in self.objectives['area_explored']:
                self.objectives['area_explored'].add(pos_tuple)
                reward += 2
                
            # Energy conservation penalty
            energy_penalty = (100 - ant.energy) * 0.01
            reward -= energy_penalty
            
            # Role-specific rewards
            if ant.role == AntRole.SCOUT:
                # Scouts get bonus for exploration
                reward += len(ant.memory) * 0.1
            elif ant.role == AntRole.GUARD:
                # Guards get bonus for staying near nest
                dist_to_nest = np.linalg.norm(ant.position - self.nest_location)
                reward += max(0, 10 - dist_to_nest) * 0.5
            elif ant.role == AntRole.LEADER:
                # Leaders get bonus for swarm cohesion
                cohesion = self._calculate_swarm_cohesion()
                reward += cohesion * 2
                
            # Communication efficiency bonus
            if ant.communication_buffer is not None:
                reward += 1
                
            rewards.append(reward)
            
        return rewards
        
    def _calculate_swarm_cohesion(self) -> float:
        """Calculate how well the swarm stays together"""
        positions = np.array([ant.position for ant in self.ants])
        center = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - center, axis=1)
        cohesion = 1.0 / (1.0 + np.mean(distances))
        return cohesion
        
    def _check_termination(self) -> bool:
        """Check if episode should terminate"""
        # All food collected
        if all(food.quantity == 0 for food in self.food_sources):
            return True
            
        # All ants exhausted
        if all(ant.energy <= 0 for ant in self.ants):
            return True
            
        # Nest depleted
        if self.nest_energy <= 0:
            return True
            
        return False
        
    def _update_objectives(self):
        """Update multi-objective tracking"""
        self.objectives['energy_consumed'] = sum(100 - ant.energy for ant in self.ants)
        self.objectives['swarm_cohesion'] = self._calculate_swarm_cohesion()
        
        # Communication efficiency
        active_comms = sum(1 for ant in self.ants if ant.communication_buffer is not None)
        self.objectives['communication_efficiency'] = active_comms / len(self.ants)
        
    def get_state(self):
        """Get current state for all ants"""
        states = []
        
        for ant in self.ants:
            # Basic state
            state = []
            
            # Distance to nearest food
            min_food_dist = float('inf')
            food_direction = np.zeros(2)
            for food in self.food_sources:
                dist = np.linalg.norm(ant.position - food.position)
                if dist < min_food_dist:
                    min_food_dist = dist
                    food_direction = food.position - ant.position
                    
            state.extend([min_food_dist / self.grid_size])
            state.extend(food_direction / (np.linalg.norm(food_direction) + 1e-8))
            
            # Pheromone levels at current position
            x, y = int(ant.position[0]), int(ant.position[1])
            for ptype in ['food', 'danger', 'exploration']:
                state.append(self.pheromone_trails[ptype][x, y])
                
            # Ant-specific features
            state.append(float(ant.has_food))
            state.append(ant.energy / 100.0)
            state.append(float(ant.role.value == 'leader'))
            
            # Distance to nest
            nest_dist = np.linalg.norm(ant.position - self.nest_location)
            state.append(nest_dist / self.grid_size)
            
            # Communication buffer
            if ant.communication_buffer is not None:
                state.extend(ant.communication_buffer)
            else:
                state.extend([0, 0, 0, 0])
                
            # Nearby ants count
            nearby_ants = sum(1 for other in self.ants 
                            if other.id != ant.id and 
                            np.linalg.norm(other.position - ant.position) < 5)
            state.append(nearby_ants / len(self.ants))
            
            states.append(np.array(state))
            
        return states
        
    def get_adjacency_matrix(self):
        """Get adjacency matrix for ant network"""
        n = len(self.ants)
        adj_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if self.ant_network.has_edge(i, j):
                    adj_matrix[i, j] = self.ant_network[i][j]['weight']
                    
        return adj_matrix
        
    def render(self):
        """Render the environment"""
        if not self.visualize:
            return
            
        self.screen.fill((20, 20, 30))  # Dark background
        
        # Draw pheromone trails
        self._draw_pheromones()
        
        # Draw obstacles
        self._draw_obstacles()
        
        # Draw food sources
        self._draw_food_sources()
        
        # Draw nest
        self._draw_nest()
        
        # Draw ants
        self._draw_ants()
        
        # Draw info
        self._draw_info()
        
    def _draw_pheromones(self):
        """Draw pheromone trails with different colors"""
        colors = {
            'food': (0, 255, 0),
            'danger': (255, 0, 0),
            'exploration': (0, 0, 255)
        }
        
        for ptype, trail in self.pheromone_trails.items():
            max_intensity = np.max(trail) + 1e-8
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    intensity = trail[x, y] / max_intensity
                    if intensity > 0.1:
                        color = tuple(int(c * intensity) for c in colors[ptype])
                        pygame.draw.rect(
                            self.screen, color,
                            (x * 15, y * 15, 15, 15), 1
                        )
                        
    def _draw_obstacles(self):
        """Draw obstacles"""
        for obstacle in self.obstacles:
            color = (100, 100, 100) if obstacle.type == 'rock' else (0, 100, 200)
            pygame.draw.circle(
                self.screen, color,
                (int(obstacle.position[0] * 15 + 7),
                 int(obstacle.position[1] * 15 + 7)),
                obstacle.size * 7
            )
            
    def _draw_food_sources(self):
        """Draw dynamic food sources"""
        for food in self.food_sources:
            if food.quantity > 0:
                size = int(np.sqrt(food.quantity / self.max_food_per_source) * 10) + 3
                pygame.draw.circle(
                    self.screen, (0, 200, 255),
                    (int(food.position[0] * 15 + 7),
                     int(food.position[1] * 15 + 7)),
                    size
                )
                
    def _draw_nest(self):
        """Draw nest with energy indicator"""
        nest_color = (138, 43, 226) if self.nest_energy > 500 else (200, 100, 100)
        pygame.draw.polygon(
            self.screen, nest_color,
            self._get_hexagon_points(
                self.nest_location[0] * 15 + 7,
                self.nest_location[1] * 15 + 7,
                20
            )
        )
        
    def _get_hexagon_points(self, x, y, size):
        """Get hexagon points for nest drawing"""
        points = []
        for i in range(6):
            angle = i * np.pi / 3
            px = x + size * np.cos(angle)
            py = y + size * np.sin(angle)
            points.append((px, py))
        return points
        
    def _draw_ants(self):
        """Draw ants with role-specific colors"""
        role_colors = {
            AntRole.WORKER: (255, 0, 0),
            AntRole.SCOUT: (0, 255, 255),
            AntRole.GUARD: (255, 255, 0),
            AntRole.LEADER: (255, 0, 255)
        }
        
        for ant in self.ants:
            color = role_colors[ant.role]
            if ant.has_food:
                color = (255, 215, 0)  # Gold
                
            # Fade color based on energy
            energy_factor = ant.energy / 100.0
            color = tuple(int(c * energy_factor) for c in color)
            
            pygame.draw.circle(
                self.screen, color,
                (int(ant.position[0] * 15 + 7),
                 int(ant.position[1] * 15 + 7)),
                5
            )
            
            # Draw communication indicator
            if ant.communication_buffer is not None:
                pygame.draw.circle(
                    self.screen, (255, 255, 255),
                    (int(ant.position[0] * 15 + 7),
                     int(ant.position[1] * 15 + 7)),
                    8, 1
                )
                
    def _draw_info(self):
        """Draw information overlay"""
        info_texts = [
            f"Food: {self.objectives['food_collected']}",
            f"Explored: {len(self.objectives['area_explored'])}",
            f"Nest Energy: {self.nest_energy:.0f}",
            f"Cohesion: {self.objectives['swarm_cohesion']:.2f}"
        ]
        
        for i, text in enumerate(info_texts):
            rendered = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(rendered, (5, 5 + i * 20))
            
    def capture_frame_for_streamlit(self):
        """Capture frame for Streamlit visualization"""
        self.render()
        if self.screen:
            rgb_array = pygame.surfarray.array3d(self.screen)
            try:
                transposed_array = np.transpose(rgb_array, (1, 0, 2))
                self.latest_frame_image = Image.fromarray(transposed_array)
            except ValueError as e:
                print(f"Error during frame capture: {e}")
                self.latest_frame_image = None
                
    def close(self):
        """Clean up resources"""
        if self.visualize:
            pygame.font.quit()
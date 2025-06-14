from environment import SwarmEnvironment
from ants import AntSwarm, AntSwarmRL
import numpy as np
import pygame # For pygame.init()
# from PIL import Image # Not directly used here, but AntSwarmRL uses it.


# Define the environment for the ant swarm
class AntEnvironment(SwarmEnvironment):
    def __init__(
        self,
        num_actions,
        num_states,
        grid_size,
        num_ants,
        num_food_sources,
        max_food_per_source,
        **rl_params
    ):
        # Ensure Pygame is initialized
        # It's generally safe to call pygame.init() multiple times.
        pygame.init()

        # Store RL parameters for later use (if needed)
        self.rl_params = rl_params

        # Initialize AntSwarm with only the relevant parameters
        self.ant_swarm = AntSwarmRL(
            grid_size=grid_size,
            num_ants=num_ants,
            num_food_sources=num_food_sources,
            max_food_per_source=max_food_per_source,
        )

        self.num_actions = num_actions
        self.num_states = num_states

    def reset(self):
        
        self.ant_swarm.reset()
        return self.get_state()

    def step(self, actions):
        # Apply actions to the ant swarm
        for ant, action in zip(self.ant_swarm.ants, actions):
            self.apply_action(ant, action)

        # Update the ant swarm for a time step
        self.ant_swarm.step(learning=False)

        # Return the next state and rewards
        next_state = self.get_state()
        rewards = self.calculate_rewards()
        done = self.check_if_done()
        return next_state, rewards, done

    def apply_action(self, ant, action):
        # Apply the action to the ant
        # actions: 0 = move randomly, 1 = follow pheromone trail, 2 = drop pheromone and return to nest, 3 = return to nest only

        # check if action is possible, otherwise default is random walk
        if not ant["has_food"] and action > 1:
            action = 0
        if ant["has_food"] and action <= 1:
            action = 3
        self.ant_swarm.move_ant(ant, action)

    def calculate_rewards(self):
        # Define and calculate rewards for the ants' actions

        return [self.get_reward(ant) for ant in self.ant_swarm.ants]

    def get_reward(self, ant):
        if ant["has_food"]:
            if np.array_equal(ant["position"], self.ant_swarm.nest_location):
                ant["has_food"] = False
                return 10000  # Reward for delivering food
            else:
                return 2500  # Reward for finding food
        elif (
            ant["has_food"]
            and self.ant_swarm.pheromone_trail[tuple(ant["position"])] > 0
        ):
            return 5000  # Increased reward for finding food following pheromone trail

        return -10  # Small penalty to encourage exploration

    def check_if_done(self):
        # Example termination condition
        return len(self.ant_swarm.food_sources) == 0  # All food collected

    def get_state(self):
        states = []
        for ant in self.ant_swarm.ants:
            nearest_food_distance = self.ant_swarm.get_nearest_food_distance(
                ant["position"]
            )
            pheromone_level = self.ant_swarm.pheromone_trail[tuple(ant["position"])]
            state = (nearest_food_distance, pheromone_level, ant["has_food"])
            states.append(state)
        return states

    def render(self):
        # This now calls the core drawing logic in AntSwarmRL without flip/tick
        self.ant_swarm.render()

    def capture_frame_for_streamlit(self):
        """Captures the current simulation frame from AntSwarmRL."""
        self.ant_swarm.capture_frame_for_streamlit()

    @property
    def latest_frame_image(self):
        """Provides access to the latest captured frame from AntSwarmRL."""
        if self.ant_swarm:
            return self.ant_swarm.latest_frame_image
        return None

    def close(self):
        self.ant_swarm.close()

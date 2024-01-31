from environment import SwarmEnvironment
from ants import AntSwarm
import numpy as np


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
        # Store RL parameters for later use (if needed)
        self.rl_params = rl_params

        # Initialize AntSwarm with only the relevant parameters
        self.ant_swarm = AntSwarm(
            grid_size=grid_size, num_ants=num_ants, num_food_sources=num_food_sources, max_food_per_source=max_food_per_source
        )

        self.num_actions = num_actions
        self.num_states = num_states

    def reset(self):
        # Assuming AntSwarm has a method to reset its state
        self.ant_swarm.reset()
        return self.get_state()

    def step(self, actions):
        # Apply actions to the ant swarm
        # This part depends on how actions influence the AntSwarm
        # For example, if actions are movements, apply these to each ant
        for ant, action in zip(self.ant_swarm.ants, actions):
            self.apply_action(ant, action)

        # Update the ant swarm for a time step
        self.ant_swarm.step()

        # Return the next state and rewards
        next_state = self.get_state()
        rewards = self.calculate_rewards()
        done = self.check_if_done()
        return next_state, rewards, done

    def apply_action(self, ant, action):
        # Apply the action to the ant
        # actions: 0 = move randomly, 1 = follow pheromone trail, 2 = return to nest
        
        # check if action is possible, otherwise deafult is random walk
        if ant["has_food"] and action != 2:
            action = 2
        if action == 2 and not ant["has_food"]:
            action = 0
        self.ant_swarm.move_ant(ant, action)
        

    def calculate_rewards(self):
        # Define and calculate rewards for the ants' actions
        
        return [self.get_reward(ant) for ant in self.ant_swarm.ants]

    def get_reward(self, ant):
        if ant["has_food"] and np.array_equal(ant["position"], self.ant_swarm.nest_location):
            ant["has_food"] = False
            return 100  # Reward for delivering food
        elif ant["has_food"] and self.ant_swarm.pheromone_trail[tuple(ant["position"])] > 0:
            return 50  # Increased reward for finding food following pheromone trail
        elif self.ant_swarm.pheromone_trail[tuple(ant["position"])] > 0:
            return 5  # Reward for following pheromone trail
        return -0.05  # Small penalty to encourage exploration

    def check_if_done(self):
        # Example termination condition
        return len(self.ant_swarm.food_sources) == 0  # All food collected

    
    def get_state(self):
        states = []
        for ant in self.ant_swarm.ants:
            nearest_food_distance = self.ant_swarm.get_nearest_food_distance(ant["position"])
            pheromone_level = self.ant_swarm.pheromone_trail[tuple(ant["position"])]
            state = (nearest_food_distance, pheromone_level, ant["has_food"])
            states.append(state)
        return states



    def render(self):
        self.ant_swarm.render()

    def close(self):
        self.ant_swarm.close()

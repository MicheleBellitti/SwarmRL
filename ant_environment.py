from environment import SwarmEnvironment
from ants import AntSwarm
import numpy as np

# Define the environment for the ant swarm
class AntEnvironment(SwarmEnvironment):
    def __init__(self, num_actions, num_states, grid_size, num_ants, num_food_sources, **rl_params):
        # Store RL parameters for later use (if needed)
        self.rl_params = rl_params

        # Initialize AntSwarm with only the relevant parameters
        self.ant_swarm = AntSwarm(grid_size=grid_size, num_ants=num_ants, num_food_sources=num_food_sources)

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
        # Example action space: 0 = Up, 1 = Right, 2 = Down, 3 = Left
        # directions = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])  # Up, Right, Down, Left
        if 0 <= action < 4:
            self.ant_swarm.move_ant(ant)
        elif action == 4:  # Assume 4 is to pick up food
            # Implement food pick-up logic (if at food location) and returning to nest
            self.ant_swarm.return_to_nest(ant)
       


    def calculate_rewards(self):
        # Define and calculate rewards for the ants' actions
        # Example: positive reward for food collection
        return [self.get_reward(ant) for ant in self.ant_swarm.ants]

    def get_reward(self, ant):
        # Example reward structure
        if ant['has_food'] and np.array_equal(ant['position'], self.ant_swarm.nest_location):
            return 100  # Reward for delivering food
        elif not ant['has_food']:
            for food_pos in self.ant_swarm.food_sources.keys():
                if np.array_equal(ant['position'], food_pos):
                    return 10  # Reward for finding food
        return -0.1  # Small negative reward otherwise

    def check_if_done(self):
        # Example termination condition
        return len(self.ant_swarm.food_sources) == 0  # All food collected

    def get_state(self):
        # Define and return the current state of the ant swarm
        # This could be as simple as positions of all ants and food status
        return [(ant['position'], ant['has_food']) for ant in self.ant_swarm.ants]

    
    def render(self):
        self.ant_swarm.render()

    def close(self):
        self.ant_swarm.close()

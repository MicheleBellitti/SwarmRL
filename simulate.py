from ant_environment import AntEnvironment
from rl_agent import QLearning
from rl_trainer import RLTrainer

# keep track of time of completion
import time
import pygame
import pygame_gui
from params import *
import pandas as pd

# To run the interface
if __name__ == "__main__":
    start_time = time.time()

    # Initialize the ant environment
    ant_env = AntEnvironment(
        num_actions=3,
        num_states=10* 3 * 2,
        grid_size=50,
        num_ants=100,
        num_food_sources=5,
        max_food_per_source=100
    )
    # ant_env.render()
    # Get the number of states and actions
    n_states = ant_env.num_states
    n_actions = ant_env.num_actions

    # Create RL agents
    agents = [QLearning(n_states, n_actions) for _ in range(ant_env.ant_swarm.num_ants)]

    # Create and run the RL trainer
    trainer = RLTrainer(AntEnvironment, QLearning)
    configs = [config1, config2, config3, config4, config5]
    for config in configs:
        start_time = time.time()
        
        results = trainer.train(config, episodes=200)
        
        end = time.time() - start_time
        print("Experiment time: ", end)
        trainer.analyze_results(results)
            
    for i, agent in enumerate(agents):
        
        print(f"Agent {i}: {agent.q_table}")

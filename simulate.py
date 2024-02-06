from ant_environment import AntEnvironment
from rl_agent import QLearningAgent
from rl_trainer import RLTrainer

# keep track of time of completion
import time
import pygame
import pygame_gui
from params import *
import pandas as pd
import random
# To run the interface
if __name__ == "__main__":
    start_time = time.time()

    # Initialize the environment
    n_ants = config1["num_ants"]
    n_actions=config1["num_actions"]
    n_states=config1["num_states"]
    grid_size=config1["grid_size"]
    num_food_sources=config1["num_food_sources"]
    max_food_per_source=config1["max_food_per_source"]
    
    # Get the number of states and actions
    random.seed(time.time())

    # Create RL agents
    agents = [QLearningAgent(n_states, n_actions) for _ in range(n_ants)]

    # Create and run the RL trainer
    trainer = RLTrainer(AntEnvironment, QLearningAgent)
    configs = [config1, config2, config3, config4]
    for i, config in enumerate(configs):
        start_time = time.time()
        
        results = trainer.train(config, episodes=150)
        
        end = time.time() - start_time
        print("Experiment time: ", end)
        trainer.analyze_results(results, i)
            
    for i, agent in enumerate(agents):
        
        print(f"Agent {i}: {agent.q_table}")

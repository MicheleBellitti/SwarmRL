import time
import random
from threading import Thread, Event
from ant_environment import AntEnvironment
from rl_agent import QLearningAgent
from rl_trainer import RLTrainer
from params import *
import pygame


class SimulationManager:
    def __init__(self):
        self.configs = [config1, config2, config3, config4]
        self.current_config_index = 0
        self.trainer = RLTrainer(AntEnvironment, QLearningAgent)
        self.state = "stopped"  # Possible states: stopped, running, paused

    def start(self):
        if self.state in ["running", "paused"]:
            print("Simulation is already running or paused.")
            return

        self.state = "running"
        return self.run_simulation()

    def run_simulation(self):
        config = self.configs[self.current_config_index]
        self.trainer = RLTrainer(AntEnvironment, QLearningAgent)
        n_ants = config["num_ants"]
        n_states = config["num_states"]
        n_actions = config["num_actions"]
        # agents = [QLearningAgent(n_states, n_actions) for _ in range(n_ants)]

        results = self.trainer.train(config, config["episodes"])
        print(results.head())
        return results

    def pause(self):
        if self.state != "running":
            print("Simulation is not running.")
            return
        self.state = "paused"
        self.trainer.set_state(self.state)

    def resume(self):
        if self.state != "paused":
            print("Simulation is not paused.")
            return

        self.state = "running"
        self.trainer.set_state(self.state)

    def quit(self):

        self.state = "stopped"
        self.trainer.set_state(self.state)

    def get_state(self):
        # Implement fetching the current state of the simulation

        return {"state": self.state}


# Example usage
if __name__ == "__main__":
    pygame.init()
    sim_manager = SimulationManager()
    
    df = sim_manager.start()  # Start the simulation
    sim_manager.trainer.analyze_results(df, idx=sim_manager.current_config_index)

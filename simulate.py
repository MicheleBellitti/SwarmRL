import time
import random
import argparse
from threading import Thread, Event
from ant_environment import AntEnvironment
from rl_agent import PPOAgent, GNNAgent
from rl_trainer import RLTrainer
from params import *
import pygame


class SimulationManager:
    def __init__(self):
        # TODO: Agent class could be configurable here if PPO is re-added
        self.trainer = RLTrainer(AntEnvironment, PPOAgent)
        self.state = "stopped"  # Possible states: stopped, running, paused
        self.latest_results_df = None
        self.latest_plot_paths = None

    def start(self, config): # Modified to accept config
        if self.state in ["running", "paused"]:
            print("Simulation is already running or paused.")
            return None # Return None or raise an error

        if config is None:
            print("Error: Configuration must be provided to start the simulation.")
            return None

        self.state = "running"
        # Pass the selected config to run_simulation
        return self.run_simulation(config)

    def run_simulation(self, config): # Modified to accept config
        # Re-initialize trainer with the agent type from config if that feature is added
        self.trainer = RLTrainer(AntEnvironment, GNNAgent)
        n_actions = config["num_actions"]

        # The train method in RLTrainer now handles episodes from the config.
        self.latest_results_df = self.trainer.train(config, config["episodes"])
        if self.latest_results_df is not None and not self.latest_results_df.empty:
            print(self.latest_results_df.head())
            # Use a config identifier for plot filenames, e.g., a 'name' field in the config dict
            config_name = config.get("name", "default_run")
            self.latest_plot_paths = self.trainer.analyze_results(self.latest_results_df, idx=config_name)
        else:
            print("Training produced no results or an error occurred.")
            self.latest_results_df = None # Ensure it's None if training fails
            self.latest_plot_paths = []
        return self.latest_results_df # Return the results df

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

    def run_inference(self, weights_filepath, num_episodes, config):
        """
        Runs the simulation in inference mode using pre-trained weights.
        """
        if self.state == "running": # Should not happen if called from main script logic
            print("Simulation is already running. Stop it before starting inference.")
            return None

        print(f"Initializing trainer for inference with config: {config}")
        # Ensure trainer is initialized. It's typically done in __init__ or start.
        # If it might not be, uncomment the next line.
        # self.trainer = RLTrainer(AntEnvironment, QLearningAgent)

        self.state = "running" # Or a specific "inferring" state if needed

        # The RLTrainer's load_and_run_inference method handles environment setup internally
        inference_results = self.trainer.load_and_run_inference(config, num_episodes, weights_filepath)

        self.state = "stopped"
        return inference_results

    def get_state(self):
        # Try to get a more detailed state from the trainer if it's active
        if hasattr(self, 'trainer') and self.trainer is not None and self.trainer._state != "stopped":
            trainer_state = self.trainer.get_state()
            # Combine with SimulationManager's own state if necessary, or just return trainer's
            return {
                "simulation_manager_state": self.state,
                "trainer_state": trainer_state
            }
        return {"simulation_manager_state": self.state, "trainer_state": "N/A"}


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ant Colony Simulation (CLI for inference)")
    parser.add_argument("--mode", type=str, choices=["train", "infer"], default="infer", # Default to infer for CLI
                        help="Mode to run the simulation: 'train' (via dashboard) or 'infer' (CLI).")
    parser.add_argument("--weights_file", type=str, default="weights/q_table_agent_0.npy",
                        help="Path to the weights file for inference.")
    parser.add_argument("--inference_episodes", type=int, default=10,
                        help="Number of episodes to run in inference mode.")

    args = parser.parse_args()

    if args.mode == "infer":
        sim_manager = SimulationManager() # Simpler init now
        # For CLI inference, we'll use a default config, e.g., config1 from params.py
        # The main way to select configs will be the dashboard.
        print(f"Starting CLI inference using default config (config1) and weights from {args.weights_file}...")

        # Directly use config1 or make it selectable if more CLI flexibility is needed later
        selected_config = config1

        inference_results_df = sim_manager.run_inference(args.weights_file, args.inference_episodes, selected_config)
        if inference_results_df is not None and not inference_results_df.empty:
            print("Inference completed.")
            print(inference_results_df.head())
            # Optionally analyze results
            # sim_manager.trainer.analyze_results(inference_results_df, idx="cli_inference")
        else:
            print("Inference did not produce results.")

    pygame.quit()
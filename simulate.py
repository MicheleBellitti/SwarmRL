import time
import random
import argparse
from threading import Thread, Event
from ant_environment import AntEnvironment
from rl_agent import QLearningAgent
from rl_trainer import RLTrainer
from params import *
import pygame


class SimulationManager:
    def __init__(self):
        # self.configs = [config1, config2, config3, config4] # Removed
        # self.current_config_index = 0 # Removed
        # TODO: Agent class could be configurable here if PPO is re-added
        self.trainer = RLTrainer(AntEnvironment, QLearningAgent)
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
        # config = self.configs[self.current_config_index] # Removed
        # Re-initialize trainer with the agent type from config if that feature is added
        self.trainer = RLTrainer(AntEnvironment, QLearningAgent) # QLearningAgent for now
        # n_ants = config["num_ants"] # Not directly used by SimulationManager
        # n_states = config["num_states"] # Not directly used by SimulationManager
        n_actions = config["num_actions"]
        # agents = [QLearningAgent(n_states, n_actions) for _ in range(n_ants)] # This logic is in RLTrainer

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
    # Config selection for CLI inference will use a default (e.g., config1)
    # Full config selection is primarily for the dashboard.
    # parser.add_argument("--config_index", type=int, default=0, choices=range(4),
    #                     help="Index of the configuration to use from params.py (0-3).")

    args = parser.parse_args()

    pygame.init() # Pygame might be needed by AntEnvironment's rendering aspects even if not displayed via CLI.
                  # Consider making rendering optional in AntEnvironment for pure CLI runs.

    if args.mode == "infer":
        sim_manager = SimulationManager() # Simpler init now
        # For CLI inference, we'll use a default config, e.g., config1 from params.py
        # The main way to select configs will be the dashboard.
        print(f"Starting CLI inference using default config (config1) and weights from {args.weights_file}...")

        # Directly use config1 or make it selectable if more CLI flexibility is needed later
        selected_config = config1

        inference_results_df = sim_manager.run_inference(args.weights_file, args.inference_episodes, selected_config)
        if inference_results_df is not None and not inference_results_df.empty:
            print("Training completed.") # Should be "Inference completed."
            print(inference_results_df.head())
            # Ensure trainer is available for analyze_results
            # analyze_results might not be suitable for inference data or may need adjustment
            # if hasattr(sim_manager, 'trainer') and sim_manager.trainer is not None:
            #      sim_manager.trainer.analyze_results(inference_results_df, idx="cli_inference")
            else:
                print("Trainer not available for analyzing results.")
        else:
            print("Training did not produce results.")

    elif args.mode == "infer":
        print(f"Starting inference with config {args.config_index} using weights from {args.weights_file}...")
        # We need a method in SimulationManager to handle inference
        # For now, let's assume we add it: run_inference(self, weights_filepath, num_episodes, config)
        # This method will be added in the next step.
        # Placeholder for calling the new inference method:
        # inference_results_df = sim_manager.run_inference(args.weights_file, args.inference_episodes, selected_config)
        # if inference_results_df is not None and not inference_results_df.empty:
        #     print("Inference completed.")
        #     print(inference_results_df.head())
        # else:
        #     print("Inference did not produce results or method not yet implemented.")
        # print("Inference mode: Functionality to call inference method will be added next.") # Removed placeholder
        inference_results_df = sim_manager.run_inference(args.weights_file, args.inference_episodes, selected_config)
        if inference_results_df is not None and not inference_results_df.empty:
            print("Inference completed.")
            print(inference_results_df.head())
            # Optionally, analyze inference results (e.g., plot rewards)
            # sim_manager.trainer.analyze_results(inference_results_df, idx=f"inference_{args.config_index}")
        else:
            print("Inference did not produce results.")

    pygame.quit()

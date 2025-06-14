from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import seaborn as sns
from plotly import graph_objects as go
import plotly.figure_factory as ff
import os


# utility functions
def moving_average(data, window_size):
    """Computes moving average using discrete linear convolution of two one-dimensional sequences."""
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(data, window, "valid")


def plot_top_agents_q_tables(agents: list, top_n=3):
    """
    Plots the Q-table heatmaps of the top N agents based on their performance.

    Parameters:
    - agents: List of agents. Each agent should have a `q_table` attribute and a performance metric attribute.
    - top_n: Number of top agents to plot. Default is 3.
    """

    # Sort agents based on a performance metric, e.g., cumulative_reward
    # This assumes each agent has a `cumulative_reward` attribute for simplicity
    top_agents = sorted(agents, key=lambda agent: agent.q_table.sum(), reverse=True)[
        :top_n
    ]

    for i, agent in enumerate(top_agents, start=1):
        # Convert the Q-table to a DataFrame for better labeling (optional)
        states = [f"State {i}" for i in range(agent.q_table.shape[0])]
        actions = [f"Action {i}" for i in range(agent.q_table.shape[1])]
        q_table_df = pd.DataFrame(agent.q_table, index=states, columns=actions)

        fig = ff.create_annotated_heatmap(
        z=q_table_df.values, 
        x=list(q_table_df.columns), 
        y=list(q_table_df.index),
        annotation_text=np.around(q_table_df.values, decimals=2),
        colorscale='Viridis',
        showscale=True)
        fig.update_layout(width=1000, height=800, title_text='Interactive Heatmap of Q-table')
        fig.show()


class RLTrainer:
    def __init__(self, environment_class, agent_class):
        self.environment_class = environment_class
        self.agent_class = agent_class
        self._state = "stopped"  # Possible states: stopped, running, paused
        self.config = None
        self.episodes = 0
        self.current_episode = 0
        self.results = []
        # self.done = False
        

    def set_state(self, state):
        """Set the simulation's state."""
        self._state = state

    def get_state(self):
        """Return the current state of the simulation."""
        return {
            "state": self._state,
            "current_episode": self.current_episode,
            "total_episodes": self.episodes,
            "latest_results": self.results[-1] if self.results else {}
        }

    def train(self, config, episodes):
        """Train the simulation with given configuration and number of episodes."""
        self.config = config
        self.episodes = episodes
        self.current_episode = 0
        self.results = [] # For detailed results, returned at the end
        self._state = "running"

        # Initialize for live metrics
        self.live_metrics = {
            "rewards": [],
            "food_collected": [],
            "steps": []
        }

        consecutive_all_food_episodes = 0 # Initialize for early stopping

        # Create the environment and agents
        self.environment = self.environment_class(**self.config)

        # Agent selection based on config
        agent_type_str = self.config.get("agent_type", "qlearning").lower()
        if agent_type_str == "sarsa":
            from rl_agent import SARSAAgent
            self.agent_class_to_use = SARSAAgent
            print("Using SARSAAgent")
        else: # Default to QLearningAgent
            from rl_agent import QLearningAgent
            self.agent_class_to_use = QLearningAgent
            print("Using QLearningAgent")

        agents = [self.agent_class_to_use(self.environment.num_states, self.environment.num_actions,
                                          learning_rate=self.config.get("learning_rate", 0.1),
                                          gamma=self.config.get("gamma", 0.95),
                                          epsilon=self.config.get("epsilon", 0.5)
                                         ) for _ in range(config["num_agents"])]

        while self.current_episode < episodes:
            if self._state == "stopped":
                plot_top_agents_q_tables(agents)
                return pd.DataFrame(self.results)
            
            self.environment.running_state = self._state
            if self._state == "paused":
                time.sleep(0.1)  # Sleep briefly to reduce CPU usage while paused
                
                continue  # Skip the rest of the loop and check the state again

            # Run the episode and collect data
            max_steps_per_episode = self.config.get("max_steps_per_episode", 50) # Get max_steps from config
            episode_data = self.run_episode(self.environment, agents, max_steps=max_steps_per_episode)
            self.results.append(episode_data) # Append full episode data for final DataFrame

            # Update live metrics
            self.live_metrics["rewards"].append(episode_data["total_reward"])
            self.live_metrics["food_collected"].append(episode_data["food_collected"])
            self.live_metrics["steps"].append(episode_data["steps"])

            print(f"Episode {self.current_episode + 1}/{episodes} complete. Total Reward: {episode_data['total_reward']}")

            # Early stopping logic
            if self.config.get("early_stopping_enabled", True):
                if episode_data.get("early_stop_all_food_collected", False):
                    consecutive_all_food_episodes += 1
                else:
                    consecutive_all_food_episodes = 0 # Reset counter

                required_consecutive = self.config.get("early_stopping_consecutive_episodes", 5)
                if consecutive_all_food_episodes >= required_consecutive:
                    print(f"Early stopping: All food collected for {required_consecutive} consecutive episodes.")
                    self.current_episode += 1 # Ensure episode count reflects this last one
                    break # Exit the main training loop

            self.current_episode += 1

            # Update the epsilon value for agents
            for agent in agents:
                agent.update_epsilon(self.current_episode) # Pass true current_episode

        self._state = "stopped"
        plot_top_agents_q_tables(agents)

        # Save the Q-table of the first agent
        if agents:
            if not os.path.exists("weights"):
                os.makedirs("weights")
            weights_filepath = "weights/q_table_agent_0.npy"
            agents[0].save_weights(weights_filepath)
            print(f"Saved Q-table for agent 0 to {weights_filepath}")

        return pd.DataFrame(self.results)


    def run_episode(self, environment, agents, max_steps):
        overall_done = False # Overall episode termination flag
        episode_data = {
            "total_reward": 0,
            "steps": 0,
            "food_collected": 0,
            "pheromone_trail_usage": 0,
            "early_stop_all_food_collected": False # Initialize new metric
        }

        food_found_flags = [False] * len(agents)
        pheromone_usage_count = 0
        agent_total_rewards = np.zeros(len(agents)) # Store cumulative reward per agent for this episode

        current_states = environment.get_state()
        # Initial action selection for all agents
        current_actions = [agent.choose_action(state) for agent, state in zip(agents, current_states)]

        for step in range(max_steps):
            if self._state == "paused":
                time.sleep(0.1)
                continue
            if self._state == "stopped":
                break

            # Execute current actions
            next_states, rewards, overall_done = environment.step(current_actions)
            # environment.render() # Old direct render call, replaced by capture_frame_for_streamlit

            # If visualization is enabled in config, capture the frame
            if self.config.get("visualize", False):
                environment.capture_frame_for_streamlit()

            # Choose next actions for SARSA update (and for the next step)
            next_actions = [agent.choose_action(next_state) for agent, next_state in zip(agents, next_states)]

            # Learn step for each agent
            for i, agent in enumerate(agents):
                state_i = current_states[i]
                action_i = current_actions[i]
                reward_i = rewards[i]
                next_state_i = next_states[i]
                next_action_i = next_actions[i] # For SARSA

                # Pass the overall_done flag to each agent.
                # If environment provides per-agent done, that would be more accurate.
                agent.learn(state_i, action_i, reward_i, next_state_i, overall_done, next_action_i)

                agent_total_rewards[i] += reward_i

                # Pheromone trail usage (example metric, adapt if action representation changes)
                if action_i == 1: # Assuming action '1' means follow pheromone
                    pheromone_usage_count += 1

            current_states = next_states
            current_actions = next_actions # Crucial for SARSA: next action becomes current action

            episode_data["steps"] += 1
            if overall_done:
                break

        # Episode summary metrics
        if overall_done and environment.check_if_done(): # check_if_done() confirms it was due to food
            episode_data["early_stop_all_food_collected"] = True

        episode_data["food_collected"] = environment.ant_swarm.food_collected
        # Sum of rewards from all agents (or top N as before, adjust as needed)
        # For consistency, let's use the same top_indices logic if desired, or sum all.
        # Here, summing all agent rewards for simplicity in this change.
        episode_data["total_reward"] = sum(agent_total_rewards)
        episode_data["pheromone_trail_usage"] = pheromone_usage_count / (episode_data["steps"] * len(agents)) if episode_data["steps"] > 0 and len(agents) > 0 else 0
        
        return episode_data

    def analyze_results(self, results, idx):
        # Determine window size for moving average
        window_size = 10  # Adjust the window size as needed

        # Apply moving average to the data series
        if results.empty or window_size <= 0 or window_size > len(results):
            print("Not enough data to generate plots or invalid window size.")
            return []

        # Ensure 'plots' directory exists
        if not os.path.exists("plots"):
            os.makedirs("plots")

        plot_paths = []

        smoothed_total_reward = moving_average(results["total_reward"], window_size)
        smoothed_pheromone_trail_usage = moving_average(
            results["pheromone_trail_usage"], window_size
        )
        smoothed_avg_reward_per_step = moving_average(
            np.array(results["total_reward"]) / np.array(results["steps"]), window_size
        )
        smoothed_food_collected = moving_average(results["food_collected"], window_size)
        # Total reward trend with smoothing
        plt.figure()
        plt.plot(smoothed_total_reward, label="Total Reward per Episode", color="blue")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Total Reward per Episode")
        plt.legend()
        plot_path_total_reward = f"plots/Total_reward_{idx}.png"
        plt.savefig(plot_path_total_reward)
        plot_paths.append(plot_path_total_reward)
        # Total food collected trend with smoothing
        plt.figure()
        plt.plot(
            smoothed_food_collected, label="Food collected per Episode", color="red"
        )
        plt.xlabel("Episode")
        plt.ylabel("food collected")
        plt.title("Food collected per Episode")
        plt.legend()
        plot_path_food_collected = f"plots/Food_Collected_{idx}.png"
        plt.savefig(plot_path_food_collected)
        plot_paths.append(plot_path_food_collected)
        # Pheromone Trail Usage with smoothing
        plt.figure()
        plt.plot(
            smoothed_pheromone_trail_usage,
            label="Pheromone Trail Usage",
            color="orange",
        )
        plt.xlabel("Episode")
        plt.ylabel("Usage Ratio")
        plt.title("Pheromone Trail Usage per Episode")
        plt.legend()
        plot_path_pheromone = f"plots/Pheromone_{idx}.png"
        plt.savefig(plot_path_pheromone)
        plot_paths.append(plot_path_pheromone)

        # Average Reward Per Step with smoothing
        plt.figure()
        plt.plot(
            smoothed_avg_reward_per_step, label="Average Reward Per Step", color="green"
        )
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.title("Average Reward Per Step per Episode")
        plt.legend()
        plot_path_avg_reward = f"plots/Average_reward_{idx}.png"
        plt.savefig(plot_path_avg_reward)
        plot_paths.append(plot_path_avg_reward)
        plt.close("all") # Close all figures to free memory

        return plot_paths

    def load_and_run_inference(self, config, num_episodes, weights_filepath):
        """
        Loads agent weights and runs the simulation for inference.
        """
        print(f"Starting inference for {num_episodes} episodes...")
        self.config = config
        self.episodes = num_episodes
        self.current_episode = 0
        inference_results = []
        self._state = "running"

        # Create the environment and agent(s)
        self.environment = self.environment_class(**self.config)

        # Assuming saving/loading weights for a single agent or the first one for simplicity
        agent_type_str_inference = config.get("agent_type", "qlearning").lower()
        if agent_type_str_inference == "sarsa":
            from rl_agent import SARSAAgent
            InferenceAgentClass = SARSAAgent
        else:
            from rl_agent import QLearningAgent
            InferenceAgentClass = QLearningAgent

        num_agents_to_load = 1 # Or determine from config
        agents = [InferenceAgentClass(self.environment.num_states, self.environment.num_actions) for _ in range(num_agents_to_load)]

        if not agents:
            print("Error: No agents created for inference.")
            return pd.DataFrame(inference_results)

        # Load weights for the agent(s)
        # For simplicity, loading the same weights for all inference agents if multiple are instantiated
        # Or, adjust to load specific weights if they were saved per agent
        for agent in agents:
            try:
                agent.load_weights(weights_filepath)
                agent.epsilon = 0.01 # Set epsilon low for exploitation during inference
                print(f"Weights loaded for agent from {weights_filepath}. Epsilon set to {agent.epsilon}.")
            except FileNotFoundError:
                print(f"Error: Weights file not found at {weights_filepath}. Cannot run inference.")
                self._state = "stopped"
                return pd.DataFrame(inference_results)
            except Exception as e:
                print(f"Error loading weights for agent: {e}")
                self._state = "stopped"
                return pd.DataFrame(inference_results)


        while self.current_episode < num_episodes:
            if self._state == "stopped":
                print("Inference stopped prematurely.")
                break

            self.environment.running_state = self._state # Allow for pause/resume if needed
            if self._state == "paused":
                time.sleep(0.1)
                continue

            # Run the episode for inference (no learning)
            episode_data = self._run_inference_episode(self.environment, agents, max_steps=config.get("max_steps", 50))
            inference_results.append(episode_data)
            print(f"Inference Episode {self.current_episode + 1}/{num_episodes} complete. Reward: {episode_data['total_reward']}")
            self.current_episode += 1

        self._state = "stopped"
        print("Inference completed.")
        return pd.DataFrame(inference_results)

    def _run_inference_episode(self, environment, agents, max_steps):
        """
        Runs a single episode for inference (no learning).
        """
        done = False
        episode_data = {
            "total_reward": 0,
            "steps": 0,
            "food_collected": 0,
            # Add other metrics as needed
        }

        states = environment.get_state()
        agent_rewards = np.zeros(len(agents))

        for step in range(max_steps):
            if self._state == "paused":
                time.sleep(0.1)
                continue

            states = environment.get_state() # Get current states
            if done or self._state == "stopped":
                break

            actions = []
            for agent_idx, agent in enumerate(agents):
                # Ensure state for the agent is correctly indexed if states is a list of states
                current_agent_state = states[agent_idx] if isinstance(states, list) and len(states) == len(agents) else states
                actions.append(agent.choose_action(current_agent_state))

            next_states, rewards, overall_done = environment.step(actions) # Changed 'done' to 'overall_done' for clarity
            # environment.render() # Old direct render call

            # If visualization is enabled in config, capture the frame (for inference)
            if self.config.get("visualize", False): # Assuming self.config is accessible here, or pass config to this method
                environment.capture_frame_for_streamlit()

            for i, reward in enumerate(rewards):
                agent_rewards[i] += reward

            # Update overall_done for the episode based on this step
            # In inference, we typically run until max_steps or an explicit done signal.
            # The 'done' variable used in the loop condition should be 'overall_done'.
            if overall_done: # If environment signals episode end
                done = True # Update the loop control variable

            episode_data["steps"] += 1

        episode_data["food_collected"] = environment.ant_swarm.food_collected # Example metric
        episode_data["total_reward"] = sum(agent_rewards) # Sum of rewards from all agents

        return episode_data

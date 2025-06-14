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
        self.results = []
        self._state = "running"

        # Create the environment and agents
        self.environment = self.environment_class(**self.config)

        # Agent selection and instantiation
        agent_type_str = self.config.get("agent_type", "qlearning").lower()
        is_ppo = agent_type_str == "ppo"
        
        agents = []
        if is_ppo:
            from rl_agent import PPOAgent
            agent_params = {
                "n_states": self.config["num_states"],
                "n_actions": self.config["num_actions"],
                "lr_actor": self.config.get("lr_actor", 0.0003),
                "lr_critic": self.config.get("lr_critic", 0.001),
                "gamma": self.config.get("gamma", 0.99),
                "K_epochs": self.config.get("K_epochs", 4),
                "eps_clip": self.config.get("eps_clip", 0.2),
            }
            # Note: PPO implementation currently supports a single agent
            agents.append(PPOAgent(**agent_params))
            print("Using PPOAgent")
        else:
            AgentClass = None
            if agent_type_str == "sarsa":
                from rl_agent import SARSAAgent
                AgentClass = SARSAAgent
                print("Using SARSAAgent")
            else: # Default to QLearningAgent
                from rl_agent import QLearningAgent
                AgentClass = QLearningAgent
                print("Using QLearningAgent")
            
            agent_params = {
                "n_states": self.environment.num_states,
                "n_actions": self.environment.num_actions,
                "learning_rate": self.config.get("learning_rate", 0.1),
                "gamma": self.config.get("gamma", 0.95),
                "epsilon": self.config.get("epsilon", 0.5),
            }
            agents = [AgentClass(**agent_params) for _ in range(config["num_agents"])]

        while self.current_episode < episodes:
            if self._state == "stopped":
                if not is_ppo:
                    plot_top_agents_q_tables(agents)
                return pd.DataFrame(self.results)
            
            self.environment.running_state = self._state
            if self._state == "paused":
                time.sleep(0.1)  # Sleep briefly to reduce CPU usage while paused
                
                continue  # Skip the rest of the loop and check the state again

            # Run the episode and collect data
            max_steps_per_episode = self.config.get("max_steps_per_episode", 50) # Get max_steps from config
            
            if is_ppo:
                episode_data = self.run_ppo_episode(self.environment, agents, max_steps=max_steps_per_episode)
            else:
                episode_data = self.run_episode(self.environment, agents, max_steps=max_steps_per_episode)

            self.results.append(episode_data)
            print(f"Episode {self.current_episode + 1}/{episodes} complete. Total Reward: {episode_data['total_reward']}")
            self.current_episode += 1

            # Update the epsilon value for agents
            if not is_ppo:
                for agent in agents:
                    agent.update_epsilon(self.current_episode)

        self._state = "stopped"
        if not is_ppo:
            plot_top_agents_q_tables(agents)

        # Save the Q-table of the first agent
        if agents:
            if not os.path.exists("weights"):
                os.makedirs("weights")
            weights_filepath = "weights/q_table_agent_0.npy"
            agents[0].save_weights(weights_filepath)
            print(f"Saved Q-table for agent 0 to {weights_filepath}")

        return pd.DataFrame(self.results)


    def run_ppo_episode(self, environment, agents, max_steps):
        # This method is specifically for PPO's data collection and update cycle.
        # For simplicity, we'll handle one agent. Multi-agent PPO is more complex.
        agent = agents[0]
        
        episode_data = {
            "total_reward": 0,
            "steps": 0,
            "loss": 0,
        }
        
        state = environment.reset() # Assuming reset returns a single state object
        
        for step in range(max_steps):
            action = agent.choose_action(state[0]) # state is a list of states
            next_state, reward, done = environment.step([action])

            # Saving reward and is_terminals:
            agent.buffer.rewards.append(reward[0])
            agent.buffer.dones.append(done)
            
            state = next_state
            episode_data["total_reward"] += reward[0]
            episode_data["steps"] += 1
            if done:
                break
        
        # Update policy
        loss = agent.update()
        episode_data["loss"] = loss

        return episode_data


    def run_episode(self, environment, agents, max_steps):
        overall_done = False # Overall episode termination flag
        episode_data = {
            "total_reward": 0,
            "steps": 0,
            "food_collected": 0,
            "pheromone_trail_usage": 0,
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
        episode_data["food_collected"] = environment.ant_swarm.food_collected
        # Sum of rewards from all agents (or top N as before, adjust as needed)
        # For consistency, let's use the same top_indices logic if desired, or sum all.
        # Here, summing all agent rewards for simplicity in this change.
        episode_data["total_reward"] = sum(agent_total_rewards)
        episode_data["pheromone_trail_usage"] = pheromone_usage_count / (episode_data["steps"] * len(agents)) if episode_data["steps"] > 0 and len(agents) > 0 else 0
        
        return episode_data

    def analyze_results(self, results_df, idx):
        if results_df.empty:
            print("Results are empty, skipping analysis.")
            return []

        agent_type = self.config.get("agent_type", "qlearning")
        
        # Create a directory for plots if it doesn't exist
        if not os.path.exists("plots"):
            os.makedirs("plots")
            
        figures = {}

        # Plot 1: Cumulative Reward per Episode
        fig_reward = go.Figure()
        fig_reward.add_trace(go.Scatter(y=results_df['total_reward'], mode='lines', name='Total Reward'))
        # Add moving average
        if len(results_df['total_reward']) > 50:
            moving_avg = results_df['total_reward'].rolling(window=50).mean()
            fig_reward.add_trace(go.Scatter(y=moving_avg, mode='lines', name='50-episode MA', line=dict(dash='dash')))
        fig_reward.update_layout(title=f'[{agent_type.upper()}] Cumulative Reward per Episode - {idx}',
                                 xaxis_title='Episode', yaxis_title='Total Reward')
        figures['reward_plot'] = fig_reward

        # Plot 2: Steps per Episode
        fig_steps = go.Figure()
        fig_steps.add_trace(go.Scatter(y=results_df['steps'], mode='lines', name='Steps'))
        fig_steps.update_layout(title=f'[{agent_type.upper()}] Steps per Episode - {idx}',
                                xaxis_title='Episode', yaxis_title='Steps')
        figures['steps_plot'] = fig_steps
        
        # Algorithm-specific plots
        if agent_type == 'ppo' and 'loss' in results_df.columns:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=results_df['loss'], mode='lines', name='Loss'))
            fig_loss.update_layout(title=f'[PPO] Training Loss per Episode - {idx}',
                                     xaxis_title='Episode', yaxis_title='Loss')
            figures['loss_plot'] = fig_loss

        # Note: Saving plots to files is removed, the dashboard will use the figure objects directly.
        
        # Keep Q-table plot for non-PPO agents if needed, but return figures instead.
        if agent_type != 'ppo':
            # The heatmap can be generated here as a figure object if desired.
            pass

        return figures

    def load_and_run_inference(self, config, num_episodes, weights_filepath):
        """
        Runs inference using pre-trained weights.
        """
        self.config = config
        self.episodes = num_episodes
        self.current_episode = 0
        self.results = []
        self._state = "running"

        # Initialize environment and agent for inference
        self.environment = self.environment_class(**self.config)
        agent = self.agent_class(self.environment.num_states, self.environment.num_actions)
        agent.load_weights(weights_filepath)
        agents = [agent]  # Single agent for inference

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
            self.results.append(episode_data)
            print(f"Inference Episode {self.current_episode + 1}/{num_episodes} complete. Reward: {episode_data['total_reward']}")
            self.current_episode += 1

        self._state = "stopped"
        print("Inference completed.")
        return pd.DataFrame(self.results)

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

from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import seaborn as sns
from plotly import graph_objects as go
import plotly.figure_factory as ff


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
        agents = [self.agent_class(self.environment.num_states, self.environment.num_actions) for _ in range(config["num_agents"])]

        while self.current_episode < episodes:
            if self._state == "stopped":
                plot_top_agents_q_tables(agents)
                return pd.DataFrame(self.results)
            
            self.environment.running_state = self._state
            if self._state == "paused":
                time.sleep(0.1)  # Sleep briefly to reduce CPU usage while paused
                
                continue  # Skip the rest of the loop and check the state again

            # Run the episode and collect data
            episode_data = self.run_episode(self.environment, agents, max_steps=50)
            self.results.append(episode_data)
            print(f"Episode {self.current_episode + 1}/{episodes} complete")
            self.current_episode += 1

            # Update the epsilon value for agents
            for agent in agents:
                agent.update_epsilon(self.current_episode)

        self._state = "stopped"
        plot_top_agents_q_tables(agents)
        return pd.DataFrame(self.results)


    def run_episode(self, environment, agents, max_steps):
        done = False
        episode_data = {
            "total_reward": 0,
            "steps": 0,
            "food_collected": 0,
            "pheromone_trail_usage": 0,
        }

        states = environment.get_state()
        food_found_flags = [False] * len(
            agents
        )  # Flags to indicate if food was found by an agent
        food_finding_times = (
            []
        )  # List to store the time taken by each agent to find food
        pheromone_usage_count = 0  # Counter to keep track of pheromone trail usage
        agent_rewards = np.zeros(len(agents))
        for step in range(max_steps):
            # print(self._state)
            while self._state == "paused":
                        time.sleep(0.1)  # Sleep to reduce CPU usage while paused
                        continue  # Remain in pause loop until state changes


            states = environment.get_state()
            if done or self._state == "stopped":
                self._state = "stopped"
                break

            actions = [
                agent.choose_action(state) for agent, state in zip(agents, states)
            ]
            next_states, rewards, done = environment.step(actions)
            environment.render()

            for i, (agent, state, action, reward, next_state) in enumerate(
                zip(agents, states, actions, rewards, next_states)
            ):
                agent.learn(state, action, reward, next_state)
                # episode_data["total_reward"] += reward
                agent_rewards[i] += reward

                # Time to find food
                if not food_found_flags[i] and state[2]:  # state[2] is 'has_food'
                    food_finding_times.append(step)
                    food_found_flags[i] = True

                # Pheromone trail usage
                if action == 1:  # Follow pheromone trail
                    pheromone_usage_count += 1

            episode_data["steps"] += 1

        # Final calculations for the episode
        top_indices = np.argsort(agent_rewards)[-5:]

        episode_data["food_collected"] = environment.ant_swarm.food_collected
        episode_data["total_reward"] = sum(agent_rewards[top_indices])
        top_pheromone_usage_count = sum(
            (action == 1 and i in top_indices) for i, action in enumerate(actions)
        )
        episode_data["pheromone_trail_usage"] = pheromone_usage_count / (
            max_steps * len(agents)
        )
        
        return episode_data

    def analyze_results(self, results, idx):
        # Determine window size for moving average
        window_size = 10  # Adjust the window size as needed

        # Apply moving average to the data series
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
        plt.savefig(f"plots/Total_reward_{idx}.png")
        # Total food collected trend with smoothing
        plt.figure()
        plt.plot(
            smoothed_food_collected, label="Food collected per Episode", color="red"
        )
        plt.xlabel("Episode")
        plt.ylabel("food collected")
        plt.title("Food collected per Episode")
        plt.legend()
        plt.savefig(f"plots/Food_Collected_{idx}.png")
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
        plt.savefig(f"plots/Pheromone_{idx}.png")

        # Average Reward Per Step with smoothing
        plt.figure()
        plt.plot(
            smoothed_avg_reward_per_step, label="Average Reward Per Step", color="green"
        )
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.title("Average Reward Per Step per Episode")
        plt.legend()
        plt.savefig(f"plots/Average_reward_{idx}.png")
        plt.close("all")

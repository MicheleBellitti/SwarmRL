import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

class RLTrainer:
    def __init__(self, environment_class, agent_class):
        self.environment_class = environment_class
        self.agent_class = agent_class

    def train(self, config, episodes):
        environment = self.environment_class(**config)
        agents = [
            self.agent_class(environment.num_states, environment.num_actions)
            for _ in range(config["num_agents"])
        ]
        experiment_results = []
        for episode in range(episodes):
            # Run the episode and collect data
            episode_data = self.run_episode(environment, agents, max_steps=20)
            experiment_results.append(episode_data)
            print(f"Episode {episode + 1}/{episodes} complete")
            # Update the epsilon value
            for agent in agents:
                agent.update_epsilon(episode)
        
        
        return pd.DataFrame(experiment_results)

    def run_episode(self, environment, agents, max_steps):
        done = False
        episode_data = {
            "total_reward": 0, 
            "steps": 0, 
            "food_collected": 0,
            "time_to_find_food": 0,
            "pheromone_trail_usage": 0
        }

        states = environment.get_state()
        food_found_flags = [False] * len(agents)  # Flags to indicate if food was found by an agent
        food_finding_times = []  # List to store the time taken by each agent to find food
        pheromone_usage_count = 0  # Counter to keep track of pheromone trail usage
        for step in range(max_steps):
            if done:
                break

            actions = [agent.choose_action(state) for agent, state in zip(agents, states)]
            next_states, rewards, done = environment.step(actions)
            environment.render()

            for i, (agent, state, action, reward, next_state) in enumerate(
                zip(agents, states, actions, rewards, next_states)):

                agent.learn(state, action, reward, next_state)
                episode_data["total_reward"] += reward

                # Time to find food
                if not food_found_flags[i] and state[2]:  # state[2] is 'has_food'
                    food_finding_times.append(step)
                    food_found_flags[i] = True

                # Pheromone trail usage
                if action == 1:  # Follow pheromone trail
                    pheromone_usage_count += 1

            episode_data["steps"] += 1

        # Final calculations for the episode
        episode_data["food_collected"] = environment.ant_swarm.food_collected
        episode_data["time_to_find_food"] = np.mean(food_finding_times) if food_finding_times else np.nan
        episode_data["pheromone_trail_usage"] = pheromone_usage_count / (episode_data["steps"] * len(agents))

        return episode_data


    def analyze_results(self, results):
        # Analyze the results and generate plots
        # Time to Find Food
        plt.figure()
        plt.plot(results['time_to_find_food'], label='Time to Find Food')
        plt.xlabel('Episode')
        plt.ylabel('Average Time Steps')
        plt.title('Time to Find Food per Episode')
        plt.legend()
        plt.show() 

        # Pheromone Trail Usage
        plt.figure()
        plt.plot(results['pheromone_trail_usage'], label='Pheromone Trail Usage')
        plt.xlabel('Episode')
        plt.ylabel('Usage Ratio')
        plt.title('Pheromone Trail Usage per Episode')
        plt.legend()
        plt.show()

        # Average Reward Per Step
        plt.figure()
        plt.plot(results['total_reward'] / results['steps'], label='Average Reward Per Step')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('Average Reward Per Step per Episode')
        plt.legend()
        plt.show()

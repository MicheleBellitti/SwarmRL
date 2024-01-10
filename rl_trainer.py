import pandas as pd
import matplotlib.pyplot as plt


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
            episode_data = self.run_episode(environment, agents, max_steps=100)
            experiment_results.append(episode_data)
            print(f"Episode {episode + 1}/{episodes} complete")

        return pd.DataFrame(experiment_results)

    def run_episode(self, environment, agents, max_steps):
        done = False
        episode_data = {"total_reward": 0, "steps": 0, "food_collected": 0}
        states = environment.get_state()
        for _ in range(max_steps):
            if done:
                break
            states = environment.get_state()
            actions = [
                agent.choose_action(state) for agent, state in zip(agents, states)
            ]
            next_states, rewards, done = environment.step(actions)
            environment.render()

            for agent, state, action, reward, next_state in zip(
                agents, states, actions, rewards, next_states
            ):
                agent.learn(state, action, reward, next_state)

                if reward > 0:
                    print(reward)
                episode_data["total_reward"] += reward

            episode_data["steps"] += 1
            episode_data["food_collected"] = environment.ant_swarm.food_collected

        return episode_data

    def analyze_results(self, results):
        # Analyze the results and generate plots
        results.plot(x="steps", y="total_reward")
        plt.title("Total Reward over Steps")
        plt.show()

        results.plot(x="steps", y="food_collected")
        plt.title("Food Collected over Steps")
        plt.show()

import numpy as np
from abc import ABC, abstractmethod
# import ray # Removed as PPO is not currently supported due to dependency issues
# from ray.rllib.algorithms.ppo import PPOConfig # Removed

# # Initialize Ray once # Removed
# ray.init(ignore_reinit_error=True)


class RLAgent(ABC):
    """
    Abstract base class for reinforcement learning agents.
    """

    @abstractmethod
    def choose_action(self, observation):
        """
        Choose an action based on the given observation.

        Parameters:
        observation: The current observation of the agent.

        Returns:
        int: The action chosen by the agent.
        """
        pass

    @abstractmethod
    def learn(self, state, action, reward, next_state, done, next_action=None): # Added next_action
        """
        Update the agent's knowledge based on the experience.

        Parameters:
        state: The initial state.
        action: The action taken.
        reward: The reward received.
        next_state: The state reached after the action.
        done: Whether the episode has terminated.
        next_action: The action taken in the next_state (for SARSA).
        """
        pass

    @abstractmethod
    def save_weights(self, filepath):
        pass

    @abstractmethod
    def load_weights(self, filepath):
        pass


class PPOAgent(RLAgent):
    def __init__(self, n_states=None, n_actions=None, agent_specific_config=None):
        # agent_specific_config might contain RLLib specific settings,
        # e.g., {"env": "CartPole-v1"}
        # n_states and n_actions are part of RLAgent interface, might not be directly used by PPO
        # if the environment is registered with RLLib and has its own space definitions.

        if agent_specific_config is None:
            # Default to CartPole-v1 if no specific config provided
            # This is primarily for initial setup and testing Ray/RLLib integration
            self.env_name = "CartPole-v1"
            print(f"PPOAgent: agent_specific_config not provided, defaulting to env: {self.env_name}")
        else:
            self.env_name = agent_specific_config.get("env", "CartPole-v1")
            print(f"PPOAgent: Initializing with env: {self.env_name}")

        # Basic PPO configuration
        config = (
            PPOConfig()
            .environment(env=self.env_name)
            .framework("torch") # Or "tf"
            .rollouts(num_rollout_workers=1) # Adjust as needed
        )
        
        # For discrete action spaces like CartPole, n_actions might be inferred.
        # For continuous, or more complex custom envs, action space setup is critical.
        # self.n_actions = n_actions # Store if needed for other logic

        try:
            self.agent = config.build()
            print(f"PPOAgent for {self.env_name} initialized successfully.")
        except Exception as e:
            print(f"Error initializing PPOAgent: {e}")
            # Fallback or error handling - for now, let's ensure self.agent exists for other methods
            # This is a simplistic fallback. A real scenario might re-raise or handle differently.
            print("PPOAgent initialization failed. Agent will not be functional.")
            self.agent = None


    def choose_action(self, observation):
        if self.agent is None:
            print("PPOAgent not initialized, cannot choose action.")
            # Return a default action or raise an error, depending on desired handling
            return 0 # Assuming 0 is a valid action for CartPole as a fallback

        # RLLib's compute_single_action is typically used for inference/exploitation
        return self.agent.compute_single_action(observation)

    def learn(self, state, action, reward, next_state, done, next_action=None): # Added next_action
        # PPO typically learns from batches of experiences collected by rollout workers.
        # The RLLib PPOTrainer.train() method triggers a round of experience collection and learning.
        # Directly feeding single transitions (s, a, r, s', done) is not the standard RLLib PPO workflow.
        # For this subtask, we'll call .train() and acknowledge this is a temporary simplification.
        # Proper integration would involve adapting RLTrainer or using RLLib's own training utilities.
        if self.agent is None:
            print("PPOAgent not initialized, cannot learn.")
            return {}

        print("PPOAgent.learn() called. Triggering self.agent.train(). Note: This is a simplified integration.")
        # The result of train() is a dictionary of metrics.
        try:
            result = self.agent.train()
            return result
        except Exception as e:
            print(f"Error during PPOAgent.train(): {e}")
            return {}

    def save_weights(self, filepath):
        if self.agent is None:
            print("PPOAgent not initialized, cannot save weights.")
            return None
        try:
            # RLLib's save creates a checkpoint directory.
            # The 'filepath' should ideally be a directory path.
            checkpoint_dir = self.agent.save(checkpoint_dir=filepath) # filepath is expected to be a dir
            print(f"PPOAgent weights saved to checkpoint directory: {checkpoint_dir}")
            return checkpoint_dir
        except Exception as e:
            print(f"Error saving PPOAgent weights: {e}")
            return None

    def load_weights(self, filepath):
        if self.agent is None:
            print("PPOAgent not initialized, cannot load weights.")
            return
        try:
            # 'filepath' should be the path to the checkpoint directory/file created by save().
            self.agent.restore(filepath)
            print(f"PPOAgent weights loaded from: {filepath}")
        except Exception as e:
            print(f"Error loading PPOAgent weights: {e}")

class QLearningAgent(RLAgent):
    """
    An agent that implements the Q-learning algorithm.
    """
    def __init__(
        self, n_states, n_actions, learning_rate=0.1, gamma=0.95, epsilon=0.5, seed=42
    ):
        # Set the seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Initialize Q-table with random values
        self.q_table = np.random.rand(n_states, n_actions)
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        state_index = self.state_to_index(state)

        if np.random.uniform(0, 1) < self.epsilon:
            # Exploration: Random action
            action = np.random.choice(self.n_actions)
        else:
            # Exploitation: Best known action
            action = np.argmax(self.q_table[state_index])

        return action
    def update_epsilon(self, episode):
        self.epsilon = max(0.1, self.epsilon * (0.9 ** episode))  # Exponential decay
    def state_to_index(self, state):
        distance_to_food, pheromone_level, has_food = state

        # Discretizing distance to food
        distance_index = int(distance_to_food % 10)

        # Discretizing pheromone level
        if pheromone_level < 0.33:
            pheromone_index = 0  # Low
        elif pheromone_level < 0.66:
            pheromone_index = 1  # Medium
        else:
            pheromone_index = 2  # High

        # Combining has_food boolean
        has_food_index = 1 if has_food else 0

        # Calculating total index and returning it
        return distance_index * 3 * 2 + pheromone_index * 2 + has_food_index
        

    def learn(self, state, action, reward, next_state, done, next_action=None): # Added next_action
        # next_action is ignored by QLearningAgent
        state_index = self.state_to_index(state)
        next_state_index = self.state_to_index(next_state)

        # Update the Q-table
        current_q = self.q_table[state_index, action]
        
        if done:
            target_q = reward # If done, the future reward is 0
        else:
            max_future_q = np.max(self.q_table[next_state_index]) # Q-learning uses max Q value for next state
            target_q = reward + self.gamma * max_future_q

        new_q = current_q + self.lr * (target_q - current_q)
        self.q_table[state_index, action] = new_q

    def save_weights(self, filepath):
        """
        Saves the Q-table to a file.
        """
        np.save(filepath, self.q_table)
        print(f"Q-table saved to {filepath}")

    def load_weights(self, filepath):
        """
        Loads the Q-table from a file.
        """
        self.q_table = np.load(filepath)
        print(f"Q-table loaded from {filepath}")


class SARSAAgent(RLAgent):
    """
    An agent that implements the SARSA algorithm.
    """
    def __init__(
        self, n_states, n_actions, learning_rate=0.1, gamma=0.95, epsilon=0.5, seed=42
    ):
        if seed is not None:
            np.random.seed(seed)

        self.q_table = np.random.rand(n_states, n_actions)
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        state_index = self.state_to_index(state)
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.n_actions)
        else:
            action = np.argmax(self.q_table[state_index])
        return action

    def update_epsilon(self, episode):
        self.epsilon = max(0.1, self.epsilon * (0.9 ** episode))

    def state_to_index(self, state):
        # This state_to_index function is specific to the AntEnvironment's state representation.
        # It might need to be generalized or passed in if other environments are used.
        distance_to_food, pheromone_level, has_food = state
        distance_index = int(distance_to_food % 10)
        if pheromone_level < 0.33:
            pheromone_index = 0
        elif pheromone_level < 0.66:
            pheromone_index = 1
        else:
            pheromone_index = 2
        has_food_index = 1 if has_food else 0
        return distance_index * 3 * 2 + pheromone_index * 2 + has_food_index

    def learn(self, state, action, reward, next_state, done, next_action): # next_action is crucial for SARSA
        state_index = self.state_to_index(state)
        next_state_index = self.state_to_index(next_state)

        current_q = self.q_table[state_index, action]

        if done:
            target_q = reward # If done, the future reward is 0, so Q(s',a') is effectively 0.
        else:
            # SARSA uses the Q-value of the actual next action taken in the next state.
            next_q_value = self.q_table[next_state_index, next_action]
            target_q = reward + self.gamma * next_q_value
        
        new_q = current_q + self.lr * (target_q - current_q)
        self.q_table[state_index, action] = new_q

    def save_weights(self, filepath):
        np.save(filepath, self.q_table)
        print(f"SARSA Q-table saved to {filepath}")

    def load_weights(self, filepath):
        self.q_table = np.load(filepath)
        print(f"SARSA Q-table loaded from {filepath}")

import numpy as np
from abc import ABC, abstractmethod
# from ray.rllib.agents import ppo

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
    def learn(self, state, action, reward, next_state):
        """
        Update the agent's knowledge based on the experience.

        Parameters:
        state: The initial state.
        action: The action taken.
        reward: The reward received.
        next_state: The state reached after the action.
        """
        pass


''' class PPOAgent(RLAgent):
    def __init__(self, env_config):
        self.agent = ppo.PPOTrainer(env=env_config)
        
    def train(self):
        return self.agent.train()
    def learn(self, state, action, reward, next_state):
        self.agent.train()
        # self.agent.learn(state, action, reward, next_state)
    def get_policy(self):
        return self.agent.get_policy()
    
    def compute_action(self, observation):
        return self.agent.compute_action(observation)
    
    def choose_action(self, observation):
        return self.compute_action(observation)
    '''
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
        

    def learn(self, state, action, reward, next_state):
        # Convert state and next_state to indices
        state_index = self.state_to_index(state)
        next_state_index = self.state_to_index(next_state)

        # Update the Q-table
        current_q = self.q_table[state_index, action]
        
        max_future_q = np.max(self.q_table[next_state_index])
        new_q = (1 - self.lr) * current_q - self.lr * (
            reward + self.gamma * max_future_q
        )
        #print(f"Before q_val: {self.q_table[state_index, action]}")
        self.q_table[state_index, action] = new_q
        # print(f"After q_val: {self.q_table[state_index, action]}")
        # print(f"Learning: S={state}, A={action}, R={reward}, S'={next_state}")
        

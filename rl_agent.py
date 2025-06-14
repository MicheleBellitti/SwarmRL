import numpy as np
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.distributions import Categorical
import os
# import ray # Removed as PPO is not currently supported due to dependency issues
# from ray.rllib.algorithms.ppo import PPOConfig # Removed

# # Initialize Ray once # Removed
# ray.init(ignore_reinit_error=True)


# Buffer for PPO
class PPOBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.state_values = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]
        del self.state_values[:]

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


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        raise NotImplementedError

    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)
        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy


class PPOAgent(RLAgent):
    def __init__(self, n_states, n_actions, lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=4, eps_clip=0.2, **kwargs):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = PPOBuffer()

        self.policy = ActorCritic(n_states, n_actions)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(n_states, n_actions)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def state_to_tensor(self, state):
        # This is a basic conversion. A more robust solution would involve environment-specific observation spaces.
        distance_to_food, pheromone_level, has_food = state
        norm_dist = distance_to_food / 50.0 # Normalize by max possible distance (grid diagonal)
        norm_phero = pheromone_level / 30.0 # Normalize by an estimated max pheromone level
        has_food_float = 1.0 if has_food else 0.0
        return torch.FloatTensor([norm_dist, norm_phero, has_food_float])

    def choose_action(self, state):
        with torch.no_grad():
            state_tensor = self.state_to_tensor(state)
            action, action_logprob, state_val = self.policy_old.act(state_tensor.unsqueeze(0))
        
        # Store transition in buffer
        self.buffer.states.append(state_tensor)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.item()
    
    def learn(self, state, action, reward, next_state, done, next_action=None):
        # This method is not used by PPO, which learns in batches.
        # The `update` method should be called from the trainer instead.
        pass

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach()
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach()

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        
        return loss.mean().item()

    def save_weights(self, filepath):
        torch.save(self.policy.state_dict(), filepath)

    def load_weights(self, filepath):
        self.policy.load_state_dict(torch.load(filepath))
        self.policy_old.load_state_dict(torch.load(filepath))


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
        self.epsilon = max(0.1, self.epsilon * (0.9 ** episode))  # Exponential decay

    def state_to_index(self, state):
        # This state_to_index function is specific to the AntEnvironment's state representation.
        # It might need to be generalized or passed in if other environments are used.
        distance_to_food, pheromone_level, has_food = state
        distance_index = int(distance_to_food % 10)
        pheromone_index = 0
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
            target_q = reward
        else:
            next_q = self.q_table[next_state_index, next_action] # SARSA uses the Q-value of the next action
            target_q = reward + self.gamma * next_q

        new_q = current_q + self.lr * (target_q - current_q)
        self.q_table[state_index, action] = new_q

    def save_weights(self, filepath):
        np.save(filepath, self.q_table)
        print(f"SARSA Q-table saved to {filepath}")

    def load_weights(self, filepath):
        self.q_table = np.load(filepath)
        print(f"SARSA Q-table loaded from {filepath}")

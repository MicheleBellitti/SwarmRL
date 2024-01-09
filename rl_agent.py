import numpy as np
import pprint
class RLAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.01, gamma=0.95, epsilon=0.1):
        self.q_table = np.zeros((n_states, n_actions))
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
    
    def state_to_index(self, state):
        # Unpack the state
        position, has_food = state
        
        
        grid_size = 50
        # Convert the 2D position to a single index
        position_index = position[0] * grid_size + position[1]

        # Adjust the index based on the food-carrying status
        # We allocate the first half of indices for 'not carrying food' 
        # and the second half for 'carrying food'.
        if has_food:
            # Assuming that 'has_food' is a boolean
            return position_index + (grid_size * grid_size)
        else:
            return position_index

    def learn(self, state, action, reward, next_state):
        # pprint.pprint(self.q_table)
        # Convert state and next_state to indices
        state_index = self.state_to_index(state)
        next_state_index = self.state_to_index(next_state)

        # Update the Q-table
        current_q = self.q_table[state_index, action]
        max_future_q = np.max(self.q_table[next_state_index])
        new_q = (1 - self.lr) * current_q + self.lr * (reward + self.gamma * max_future_q)
        self.q_table[state_index, action] = new_q

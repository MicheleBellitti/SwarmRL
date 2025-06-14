config1 = {
    "episodes": 1000,  # A moderate number of episodes to ensure learning but not excessive training time.
    "num_actions": 4,
    "num_states": 10 * 5 * 2,
    "grid_size": 30,  # Environment: Standard
    "num_ants": 25,
    "num_food_sources": 3,
    "max_food_per_source": 150,
    # Tuned Q-Learning Hyperparameters (Set 1)
    "learning_rate": 0.05,
    "gamma": 0.95,
    "epsilon": 0.9,  # Initial Epsilon
    "epsilon_decay": 0.995, # Standard decay
    "num_agents": 25,
    "agent_type": "qlearning", # Explicitly Q-Learning
    "max_steps_per_episode": 100,
    "early_stopping_enabled": True,
    "early_stopping_consecutive_episodes": 10 # Example: stop if all food collected for 10 episodes
}

config2 = {
    "episodes": 1000,
    "num_actions": 4,
    "num_states": 10 * 5 * 2,
    "grid_size": 20,  # Environment: Smaller grid, more ants
    "num_ants": 30,
    "num_food_sources": 3,
    "max_food_per_source": 100,
    # Tuned Q-Learning Hyperparameters (Set 2 - same as Set 1 for now, can be varied)
    "learning_rate": 0.05,
    "gamma": 0.95,
    "epsilon": 0.9,  # Initial Epsilon
    "epsilon_decay": 0.995, # Standard decay (can be tuned too)
    "num_agents": 30,
    "agent_type": "qlearning", # Explicitly Q-Learning
    "max_steps_per_episode": 100,
    "early_stopping_enabled": True,
    "early_stopping_consecutive_episodes": 10
}
config3 = {
    "episodes": 1000,
    "num_actions": 4,
    "num_states": 10 * 5 * 2,
    "grid_size": 25,  # Environment: Medium
    "num_ants": 30,
    "num_food_sources": 4,
    "max_food_per_source": 100,
    # Tuned SARSA Hyperparameters (Set 1)
    "learning_rate": 0.05,
    "gamma": 0.95,
    "epsilon": 0.9,  # Initial Epsilon
    "epsilon_decay": 0.995, # Standard decay
    "num_agents": 30,
    "agent_type": "sarsa", # Explicitly SARSA
    "max_steps_per_episode": 100,
    "early_stopping_enabled": True,
    "early_stopping_consecutive_episodes": 10
}

config4 = {
    "episodes": 1000,
    "num_actions": 4,
    "num_states": 10 * 5 * 2,
    "grid_size": 60,  # Environment: Larger scale
    "num_ants": 200,
    "num_food_sources": 6,
    "max_food_per_source": 100,
    # Tuned SARSA Hyperparameters (Set 2 - same as Set 1 for now, can be varied)
    "learning_rate": 0.05,
    "gamma": 0.95,
    "epsilon": 0.9,  # Initial Epsilon
    "epsilon_decay": 0.995, # Standard decay
    "num_agents": 200, # Corrected num_agents to match num_ants for consistency
    "agent_type": "sarsa", # Explicitly SARSA
    "max_steps_per_episode": 150,
    "early_stopping_enabled": True,
    "early_stopping_consecutive_episodes": 5 # More aggressive for this larger config
}

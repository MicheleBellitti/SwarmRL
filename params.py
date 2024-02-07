config1 = {
    "episodes": 60,  # A moderate number of episodes to ensure learning but not excessive training time.
    "num_actions": 4,
    "num_states": 10*5*2,
    "grid_size": 30,  # Intermediate size to provide room for exploration but not too large for excessive travel time.
    "num_ants": 25,  # A moderate number of ants to balance competition and efficiency.
    "num_food_sources": 3,  # Enough food sources to provide learning opportunities without overcrowding.
    "max_food_per_source": 50,  # Moderate to ensure ants need to explore and exploit.
    "learning_rate": 0.05,  # A moderate learning rate to ensure steady but significant updates to Q-values.
    "gamma": 0.9,  # A higher discount factor to value future rewards but not as high as to cause delayed convergence.
    "epsilon": 0.98,  # Start with full exploration.
    "epsilon_decay": 0.995,  # Apply decay to epsilon to shift towards exploitation over time.
    "num_agents": 25,  # Matching the number of ants.
}

config2 = {
    "episodes": 25,
    "num_actions": 4,
    "num_states": 10*5*2,
    "grid_size": 20,
    "num_ants": 30,
    "num_food_sources": 3,
    "max_food_per_source": 25,
    "learning_rate": 0.05,
    "gamma": 0.95,
    "epsilon": 0.1,
    "num_agents": 30,
}
config3 = {
    "episodes": 40,
    "num_actions": 4,
    "num_states": 10*5*2,
    "grid_size": 25,
    "num_ants": 30,
    "num_food_sources": 5,
    "max_food_per_source": 100,
    "learning_rate": 0.01,
    "gamma": 0.5,
    "epsilon": 0.3,
    "num_agents": 30,
}

config4 = {
    "episodes": 100,
    "num_actions": 4,
    "num_states": 10*5*2,
    "grid_size": 60,
    "num_ants": 200,
    "num_food_sources": 4,
    "max_food_per_source": 100,
    "learning_rate": 0.001,
    "gamma": 0.95,
    "epsilon": 0.995,
    "num_agents": 100
}

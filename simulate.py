from ant_environment import AntEnvironment
from rl_agent import RLAgent
from rl_trainer import RLTrainer

# keep track of time of completion
import time
start_time = time.time()

# Initialize the ant environment
ant_env = AntEnvironment(num_actions=4, num_states=50*50*2,grid_size=50, num_ants=20, num_food_sources=5)
ant_env.render()
time_no_RL = time.time() - start_time
# print("Time without RL: ", time_no_RL)
# Get the number of states and actions
n_states = ant_env.num_states
n_actions = ant_env.num_actions


# Create RL agents
agents = [RLAgent(n_states, n_actions) for _ in range(10)]

# Create and run the RL trainer
trainer = RLTrainer(ant_env, agents)
start_time = time.time()
trainer.train(episodes=1000)
time_with_RL = time.time() - start_time
# print("Time with RL: ", time_with_RL)
# print(f"q_table: {agents[0].q_table}")



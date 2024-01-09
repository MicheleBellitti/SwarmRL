class RLTrainer:
    def __init__(self, environment, agents):
        self.env = environment
        self.agents = agents

    def train(self, episodes):
        for _ in range(episodes):
            states = self.env.reset()
            done = False
            while not done:
                actions = [agent.choose_action(state) for agent, state in zip(self.agents, states)]
                next_states, rewards, done = self.env.step(actions)
                for agent, state, action, reward, next_state in zip(self.agents, states, actions, rewards, next_states):
                    agent.learn(state, action, reward, next_state)
                try:
                    self.env.render()
                except Exception as e:
                    print(e)
                    self.env.close()
                    break
                
            self.env.close()

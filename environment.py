from abc import ABC, abstractmethod


class SwarmEnvironment(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, actions):
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def close(self):
        pass


class Swarm(ABC):
    def __init__(self, num_agents, environment):
        self.agents = [self.create_agent() for _ in range(num_agents)]
        self.environment = environment

    @abstractmethod
    def create_agent(self):
        pass

    def step(self):
        actions = [
            agent.get_action(self.environment.get_state()) for agent in self.agents
        ]
        self.environment.step(actions)

    def reset(self):
        self.environment.reset()
        for agent in self.agents:
            agent.reset()

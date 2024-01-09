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

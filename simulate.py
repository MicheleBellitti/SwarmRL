from ant_environment import AntEnvironment
from rl_agent import QLearning
from rl_trainer import RLTrainer

# keep track of time of completion
import time
import pygame
import pygame_gui
from params import *


class SimulationInterface:
    def __init__(self, screen_size=(800, 600)):
        pygame.init()
        self.window_surface = pygame.display.set_mode(screen_size)
        self.manager = pygame_gui.UIManager(screen_size)
        self.clock = pygame.time.Clock()
        self.create_ui_elements()

    def create_ui_elements(self):
        self.start_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((350, 275), (100, 50)),
            text="Start",
            manager=self.manager,
        )
        # Additional buttons and sliders can be added similarly

    def run(self):
        running = True
        while running:
            time_delta = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                self.manager.process_events(event)

                if event.type == pygame.USEREVENT:
                    if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                        if event.ui_element == self.start_button:
                            print("Start Simulation")

            self.manager.update(time_delta)
            self.window_surface.fill((255, 255, 255))
            self.manager.draw_ui(self.window_surface)

            pygame.display.update()

        pygame.quit()


import pandas as pd

# To run the interface
if __name__ == "__main__":
    start_time = time.time()

    # Initialize the ant environment
    ant_env = AntEnvironment(
        num_actions=4,
        num_states=50 * 50 * 2,
        grid_size=50,
        num_ants=20,
        num_food_sources=5,
    )
    # ant_env.render()
    time_no_RL = time.time() - start_time
    # print("Time without RL: ", time_no_RL)
    # Get the number of states and actions
    n_states = ant_env.num_states
    n_actions = ant_env.num_actions

    # Create RL agents
    agents = [QLearning(n_states, n_actions) for _ in range(ant_env.ant_swarm.num_ants)]

    # Create and run the RL trainer
    trainer = RLTrainer(AntEnvironment, QLearning)
    configs = [config1, config2, config3, config4, config5]
    for config in configs:
        start_time = time.time()
        results = trainer.train(config, episodes=20)
        end = time.time() - start_time
        print("Experiment time: ", end)

    # print("Time with RL: ", time_with_RL)
    trainer.analyze_results(results)
    for i, agent in enumerate(agents):
        print(f"Agent{i}: {agent.q_table}")

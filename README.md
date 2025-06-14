# SwarmRL: Reinforcement Learning in Swarm Intelligence

SwarmRL is a simulation project that explores the application of reinforcement learning algorithms within swarm intelligence scenarios, aiming at assessing the effectiveness of RL on multi-agent systems (MARL). It has been developed as the final project of the postgraduate’s course of Distributed Artificial Intelligence.

<img src="media/simulation-screen.png" alt="Swarm Simulation Example" width="40%" height="40%">

## Features

- **Interactive Dashboard**: Control simulations, select configurations (including agent type), view live agent behavior, and analyze results using a Streamlit-based dashboard.
- **Algorithm Variety**: Includes Q-Learning and SARSA reinforcement learning algorithms.
- **Agent Persistence**: Save and load trained Q-Learning and SARSA agent models.
- **Advanced Analytics**: Comprehensive dashboard for analyzing the behavior of agents within the simulation, including performance metrics and visualizations of results post-training.
- **Extensible Framework**: Designed with modularity in mind, enabling easy integration of new algorithms and simulation scenarios.

## Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed on your system. This project relies on several key Python libraries, including:
- `numpy` for numerical operations.
- `pandas` for data manipulation (results analysis).
- `matplotlib` and `seaborn` for plotting results.
- `pygame` for the underlying simulation visualization.
- `streamlit` for the interactive dashboard.
- `Pillow` for image processing for visualization.

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```
(Note: Ensure `requirements.txt` is up-to-date with all necessary packages like `streamlit`, `Pillow`, etc.)

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/MicheleBellitti/SwarmRL.git
    cd SwarmRL
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Simulation and Dashboard

### Running the Streamlit Dashboard
To use the interactive dashboard for running simulations, visualizing agent behavior, and viewing results:

1.  Ensure all dependencies are installed:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the dashboard:
    ```bash
    streamlit run dashboard.py
    ```
3.  Use the sidebar to select a simulation configuration (which includes agent type, learning parameters, etc.) and toggle live visualization.
4.  Control the simulation (start, pause, stop) using the buttons in the main panel.
5.  Results (data tables and plots) will be displayed on the dashboard after training completes.

### Command-Line Usage (Primarily for Inference)
While the Streamlit dashboard is recommended for training and interaction, `simulate.py` can still be used from the command line, primarily for running inference with pre-trained models:

```bash
python simulate.py --mode infer --weights_file <path_to_weights.npy> --config_index <idx>
```
-   `--mode infer`: Specifies inference mode.
-   `--weights_file <path_to_weights.npy>`: Path to the saved agent model (e.g., `weights/q_table_Config 1_qlearning.npy`).
-   `--config_index <idx>`: Index of the configuration (0-3) from `params.py` to use. This is important to ensure the environment and agent architecture match the saved model.

For training and more detailed interaction, please use the Streamlit dashboard.

## Agent Configuration

The type of agent (Q-Learning or SARSA) and its parameters (learning rate, epsilon, gamma, etc.) are defined within configuration dictionaries in the `params.py` file.

Each configuration (e.g., `config1`, `config2`) includes an `"agent_type"` key which can be set to `"qlearning"` or `"sarsa"`. Other parameters like `"learning_rate"`, `"epsilon"`, `"gamma"`, and `"max_steps_per_episode"` are also specified here.

To run a simulation with a specific agent algorithm and parameter set, select the corresponding configuration in the Streamlit dashboard.

## Saving and Loading Agent Weights

Trained agent models (Q-tables for Q-Learning and SARSA) are automatically saved to the `weights/` directory after a training session initiated from the Streamlit dashboard. The filename typically includes the configuration name used for training (e.g., `weights/q_table_agent_0.npy` - note: the filename was previously generic, but ideally should reflect the config name or agent type more directly if multiple models are saved).

*Self-correction: The current saving mechanism in `RLTrainer` saves to a generic `weights/q_table_agent_0.npy`. This should be enhanced in the future to include config name or agent type for clarity if multiple models from different configs are to be preserved.*

To run inference using a saved model, you can use the command-line interface:
```bash
python simulate.py --mode infer --weights_file weights/q_table_agent_0.npy --config_index 0
```
Ensure the `--config_index` corresponds to a configuration in `params.py` that matches the architecture of the saved agent (e.g., number of states and actions).

## Future Enhancements

-   **Algorithm Integration**: Adding support for more reinforcement learning algorithms. (Note: PPO integration was attempted but deferred due to environment limitations regarding disk space for large dependencies like PyTorch/Ray RLLib).
-   **Improved Visualization**: Enhancing the dashboard with more interactive elements and detailed analytics.
-   **Refined Model Naming**: Improve the naming convention for saved agent weights to include configuration details and agent type for better organization.
-   **Community Contributions**: Opening the platform for community-driven scenarios, algorithms, and improvements.

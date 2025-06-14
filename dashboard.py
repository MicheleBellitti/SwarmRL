import streamlit as st
from simulate import SimulationManager # Assuming simulate.py is in the same directory or PYTHONPATH
from ant_environment import AntEnvironment
from rl_agent import QLearningAgent
from params import config1, config2, config3, config4 # Base config
import pandas as pd # For creating DataFrames for charts
import threading
import time

# Page configuration
st.set_page_config(layout="wide", page_title="SwarmRL Simulation Dashboard")

# Initialize SimulationManager in session state if it doesn't exist
if 'sim_manager' not in st.session_state:
    st.session_state.sim_manager = SimulationManager()
    st.session_state.simulation_running = False
    st.session_state.training_thread = None
    st.session_state.results_df = None
    st.session_state.plot_paths = None
    st.session_state.active_simulation_config_name = None

    # Default values for new controls
    st.session_state.selected_agent_type = "Q-Learning"
    st.session_state.num_episodes = 100 # Shorter default for dashboard interaction
    st.session_state.learning_rate = 0.1
    st.session_state.gamma = 0.9
    st.session_state.initial_epsilon = 0.95
    st.session_state.show_visualization_cb = True
    st.session_state.early_stopping_enabled_cb = True
    st.session_state.early_stopping_consecutive_episodes_ni = 5


    # Placeholders for live charts
    st.session_state.live_reward_chart_placeholder = None
    st.session_state.live_food_chart_placeholder = None
    st.session_state.live_steps_chart_placeholder = None


st.title("🐜 SwarmRL Simulation Dashboard")

# --- Helper function to stop simulation ---
def stop_simulation():
    if st.session_state.sim_manager.state != "stopped":
        st.session_state.sim_manager.quit()
    if st.session_state.training_thread and st.session_state.training_thread.is_alive():
        # Waiting for thread to stop might be complex if sim_manager.quit() doesn't make it exit quickly.
        # For now, we rely on sim_manager.quit() to signal the thread's main loop to terminate.
        # A more robust solution might involve thread.join() with a timeout,
        # but that can still block the Streamlit app if not handled carefully.
        pass # Let the thread finish based on sim_manager's state
    st.session_state.simulation_running = False
    st.session_state.training_thread = None # Clear the thread object
    # Keep results and plots, don't clear them here
    st.session_state.active_simulation_config_name = None

# --- Configuration Selection ---
st.sidebar.header("Agent & Training Parameters")

# Agent Algorithm
st.session_state.selected_agent_type = st.sidebar.selectbox(
    "Select Agent Algorithm",
    ["Q-Learning", "SARSA"],
    index=["Q-Learning", "SARSA"].index(st.session_state.selected_agent_type)
)

# Number of Episodes
st.session_state.num_episodes = st.sidebar.number_input(
    "Number of Episodes",
    min_value=1,
    value=st.session_state.num_episodes,
    step=100
)

# Learning Rate
st.session_state.learning_rate = st.sidebar.number_input(
    "Learning Rate (alpha)",
    min_value=0.001,
    max_value=1.0,
    value=st.session_state.learning_rate,
    format="%.3f",
    step=0.01
)

# Discount Factor
st.session_state.gamma = st.sidebar.number_input(
    "Discount Factor (gamma)",
    min_value=0.0,
    max_value=1.0,
    value=st.session_state.gamma,
    format="%.2f",
    step=0.05
)

# Initial Epsilon
st.session_state.initial_epsilon = st.sidebar.number_input(
    "Initial Epsilon",
    min_value=0.0,
    max_value=1.0,
    value=st.session_state.initial_epsilon,
    format="%.2f",
    step=0.05
)

# Visualization Checkbox
st.sidebar.header("Visualization & Stopping")
st.session_state.show_visualization_cb = st.sidebar.checkbox(
    "Show Live Simulation Visual",
    value=st.session_state.show_visualization_cb,
    key="show_visualization_checkbox"
)

# Early Stopping Controls
st.session_state.early_stopping_enabled_cb = st.sidebar.checkbox(
    "Enable Early Stopping (All Food)",
    value=st.session_state.early_stopping_enabled_cb,
    key="early_stopping_enabled_checkbox"
)
st.session_state.early_stopping_consecutive_episodes_ni = st.sidebar.number_input(
    "Consecutive Episodes for Early Stop",
    min_value=1,
    value=st.session_state.early_stopping_consecutive_episodes_ni,
    step=1,
    key="early_stopping_consecutive_episodes_number_input",
    disabled=not st.session_state.early_stopping_enabled_cb
)

# --- Simulation Controls ---
st.header("Simulation Controls")
col1, col2, col3, col4, col5 = st.columns(5)


if col1.button("🚀 Start Training", disabled=st.session_state.simulation_running, use_container_width=True):
    if st.session_state.sim_manager.state == "stopped":
        # Assemble the current configuration
        current_config = config1.copy() # Start with base environmental params from params.config1
        current_config["agent_type"] = st.session_state.selected_agent_type.lower()
        current_config["episodes"] = st.session_state.num_episodes
        current_config["learning_rate"] = st.session_state.learning_rate
        current_config["gamma"] = st.session_state.gamma
        current_config["epsilon"] = st.session_state.initial_epsilon
        current_config["visualize"] = st.session_state.show_visualization_cb
        current_config["early_stopping_enabled"] = st.session_state.early_stopping_enabled_cb
        current_config["early_stopping_consecutive_episodes"] = st.session_state.early_stopping_consecutive_episodes_ni

        # Generate a descriptive name for this run
        config_name = (
            f"{st.session_state.selected_agent_type}_"
            f"Ep{st.session_state.num_episodes}_"
            f"LR{st.session_state.learning_rate}_"
            f"G{st.session_state.gamma}_"
            f"E{st.session_state.initial_epsilon}"
        )
        current_config["name"] = config_name
        st.session_state.active_simulation_config_name = config_name

        st.session_state.results_df = None
        st.session_state.plot_paths = None

        # Initialize placeholders for charts and image
        if 'live_reward_chart_placeholder' not in st.session_state or st.session_state.live_reward_chart_placeholder is None:
            # This section might need to be moved to where columns are defined if they are dynamic
            # For now, assume placeholders are created once, then updated.
            # If they are inside columns that get recreated, this logic might need adjustment.
            pass # Placeholders are defined globally now based on session state check

        if current_config["visualize"]:
            if 'image_placeholder' not in st.session_state or st.session_state.image_placeholder is None:
                 # Placeholder creation is handled in the status display section
                pass

        # Run simulation in a thread
        st.session_state.training_thread = threading.Thread(
            target=st.session_state.sim_manager.start,
            args=(current_config,)
        )
        st.session_state.training_thread.start()
        st.session_state.simulation_running = True
        st.success(f"Training started for {st.session_state.active_simulation_config_name} in the background...")
        st.experimental_rerun()
    else:
        st.warning("Simulation is already running or paused. Please stop it first.")

if col2.button("⏸️ Pause", disabled=not st.session_state.simulation_running or st.session_state.sim_manager.state != "running", use_container_width=True):
    st.session_state.sim_manager.pause()
    # simulation_running remains true, but sim_manager.state will be 'paused'
    st.info("Simulation paused.")
    st.experimental_rerun()

if col3.button("▶️ Resume", disabled=not st.session_state.simulation_running or st.session_state.sim_manager.state != "paused", use_container_width=True):
    st.session_state.sim_manager.resume()
    st.info("Simulation resumed.")
    st.experimental_rerun()

if col4.button("⏹️ Stop", disabled=st.session_state.sim_manager.state == "stopped", use_container_width=True):
    stop_simulation()
    st.warning("Simulation stopped.")
    st.experimental_rerun()

if col5.button("🔄 Refresh Status", use_container_width=True):
    st.experimental_rerun()

# --- Simulation Status & Live Visualization/Charts ---
st.header("Live Dashboard")
status_col, vis_col = st.columns([1,2]) # Column for status text, column for visualization

with status_col:
    st.subheader("Simulation Status")
    status_placeholder = st.empty()

with vis_col:
    st.subheader("Live Simulation View")
    if 'image_placeholder' not in st.session_state or st.session_state.image_placeholder is None:
        st.session_state.image_placeholder = st.empty()

# Placeholders for live charts below status/visualization
st.markdown("---") #Separator
st.subheader("Live Training Metrics")
lc_col1, lc_col2, lc_col3 = st.columns(3)
with lc_col1:
    if 'live_reward_chart_placeholder' not in st.session_state or st.session_state.live_reward_chart_placeholder is None:
        st.session_state.live_reward_chart_placeholder = st.empty()
with lc_col2:
    if 'live_food_chart_placeholder' not in st.session_state or st.session_state.live_food_chart_placeholder is None:
        st.session_state.live_food_chart_placeholder = st.empty()
with lc_col3:
    if 'live_steps_chart_placeholder' not in st.session_state or st.session_state.live_steps_chart_placeholder is None:
        st.session_state.live_steps_chart_placeholder = st.empty()


def update_status_display():
    # Update textual status
    status_data = st.session_state.sim_manager.get_state()
    sim_state = status_data.get("simulation_manager_state", "N/A").capitalize()
    trainer_info = status_data.get("trainer_state", {})

    status_md = f"**Simulation Manager:** {sim_state}"
    if st.session_state.get('active_simulation_config_name') and st.session_state.sim_manager.state != "stopped":
        status_md += f" (Config: *{st.session_state.active_simulation_config_name}*)"

    if isinstance(trainer_info, dict):
        trainer_state = trainer_info.get("state", "N/A").capitalize()
        current_ep = trainer_info.get("current_episode", 0)
        total_ep = trainer_info.get("total_episodes", 0)
        status_md += f"\n\n**Trainer:** {trainer_state}"
        if total_ep > 0 and st.session_state.sim_manager.state == "running":
            status_md += f"\n\nEpisode: {current_ep}/{total_ep} ({current_ep/total_ep:.0%})"
            # status_placeholder.progress(current_ep / total_ep) # This creates a new progress bar each time
    else:
        status_md += f"\n\n**Trainer:** {str(trainer_info)}"

    status_placeholder.markdown(status_md)

    # Progress bar for episodes
    if isinstance(trainer_info, dict) and trainer_info.get("total_episodes", 0) > 0 and st.session_state.sim_manager.state == "running":
        status_placeholder.progress(trainer_info.get("current_episode", 0) / trainer_info.get("total_episodes", 0))

    # Update visualization
    if st.session_state.show_visualization_cb and \
       st.session_state.sim_manager.state == "running" and \
       hasattr(st.session_state.sim_manager, 'trainer') and \
       st.session_state.sim_manager.trainer is not None and \
       hasattr(st.session_state.sim_manager.trainer, 'environment'):

        current_frame = st.session_state.sim_manager.trainer.environment.latest_frame_image
        if current_frame:
            st.session_state.image_placeholder.image(current_frame, caption="Live Simulation", use_column_width=True)
        elif st.session_state.sim_manager.state == "running":
             st.session_state.image_placeholder.caption("Waiting for first frame...")
    elif not st.session_state.show_visualization_cb or st.session_state.sim_manager.state == "stopped":
        if st.session_state.image_placeholder: st.session_state.image_placeholder.empty()

    # Update Live Charts
    if st.session_state.sim_manager.state == "running" and \
       hasattr(st.session_state.sim_manager, 'trainer') and \
       hasattr(st.session_state.sim_manager.trainer, 'live_metrics'):

        live_metrics = st.session_state.sim_manager.trainer.live_metrics
        if live_metrics["rewards"]:
            st.session_state.live_reward_chart_placeholder.line_chart(pd.DataFrame(live_metrics["rewards"], columns=['Reward']))
        if live_metrics["food_collected"]:
            st.session_state.live_food_chart_placeholder.line_chart(pd.DataFrame(live_metrics["food_collected"], columns=['Food Collected']))
        if live_metrics["steps"]:
            st.session_state.live_steps_chart_placeholder.line_chart(pd.DataFrame(live_metrics["steps"], columns=['Steps per Episode']))

    # Clear live charts when simulation is stopped and not just paused
    if st.session_state.sim_manager.state == "stopped":
        if st.session_state.live_reward_chart_placeholder: st.session_state.live_reward_chart_placeholder.empty()
        if st.session_state.live_food_chart_placeholder: st.session_state.live_food_chart_placeholder.empty()
        if st.session_state.live_steps_chart_placeholder: st.session_state.live_steps_chart_placeholder.empty()


update_status_display()


# --- Handle Thread Completion & Display Results ---
if st.session_state.training_thread and not st.session_state.training_thread.is_alive():
    active_config_name_at_finish = st.session_state.get('active_simulation_config_name', "N/A")
    st.success(f"Training for {active_config_name_at_finish} finished!")

    st.session_state.results_df = st.session_state.sim_manager.latest_results_df
    st.session_state.plot_paths = st.session_state.sim_manager.latest_plot_paths

    st.session_state.training_thread = None
    st.session_state.simulation_running = False
    # active_simulation_config_name is kept to label the results
    st.experimental_rerun()

# --- Results & Visualization (Post-run) ---
# Display results if they exist, using the config name active when training finished
results_config_name = st.session_state.get('active_simulation_config_name', "N/A")
if st.session_state.results_df is not None:
    st.header(f"📊 Results for {results_config_name}")
    st.dataframe(st.session_state.results_df)

if st.session_state.plot_paths:
    st.header(f"📈 Plots for {results_config_name}")
    for plot_path in st.session_state.plot_paths:
        try:
            st.image(plot_path)
        except Exception as e:
            st.error(f"Error loading plot {plot_path}: {e}")
else:
    # Show this message only if a simulation was attempted and finished (i.e., active_simulation_config_name is set)
    # and it's not currently running.
    if st.session_state.sim_manager.state == "stopped" and \
       st.session_state.get('active_simulation_config_name') is not None and \
       not st.session_state.simulation_running:
        st.info("No plots generated or training was stopped before plots could be generated.")


# Auto-refresh loop while training (optional, use with caution)
# This part is tricky with Streamlit's execution model.
# A dedicated refresh button is safer.
if st.session_state.simulation_running and st.session_state.training_thread and st.session_state.training_thread.is_alive():
    time.sleep(2) # Refresh interval
    st.experimental_rerun()


# To run this dashboard:
# 1. Ensure you are in the correct directory (where simulate.py, params.py etc. are).
# 2. Run `streamlit run dashboard.py` in your terminal.

import streamlit as st
from simulate import SimulationManager # Assuming simulate.py is in the same directory or PYTHONPATH
from ant_environment import AntEnvironment
from rl_agent import QLearningAgent
from params import config1, config2, config3, config4, config5
import threading
import time # For potential auto-refresh logic later

# Page configuration
st.set_page_config(layout="wide", page_title="SwarmRL Simulation Dashboard")

# Initialize SimulationManager in session state if it doesn't exist
if 'sim_manager' not in st.session_state:
    st.session_state.sim_manager = SimulationManager()
    st.session_state.simulation_running = False # True if training thread is active
    st.session_state.current_config_name = "Config 1"
    st.session_state.active_simulation_config_name = None # Config name of the currently/last run sim
    st.session_state.training_thread = None
    st.session_state.results_df = None
    st.session_state.plot_figures = None # Changed from plot_paths

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
st.sidebar.header("Simulation Configuration")
config_options = {
    "Config 1": config1,
    "Config 2": config2,
    "Config 3": config3,
    "Config 4": config4,
    "Config 5": config5,
}
# Use the session state to keep track of the selected config name
# Ensure current_config_name is initialized before this line if not already
if 'current_config_name' not in st.session_state:
    st.session_state.current_config_name = list(config_options.keys())[0]

selected_config_name_from_selectbox = st.sidebar.selectbox(
    "Choose Configuration",
    list(config_options.keys()),
    index=list(config_options.keys()).index(st.session_state.current_config_name)
)
# Update session state only if the selection changes
if selected_config_name_from_selectbox != st.session_state.current_config_name:
    st.session_state.current_config_name = selected_config_name_from_selectbox
    # Clear previous results when config changes if a simulation is not running
    if not st.session_state.simulation_running:
        st.session_state.results_df = None
        st.session_state.plot_figures = None # Changed from plot_paths

selected_config_obj_original = config_options[st.session_state.current_config_name]

# Display selected configuration details (optional)
with st.sidebar.expander("View Selected Configuration Details"):
    st.json(selected_config_obj_original)

st.sidebar.header("Visualization")
st.session_state.show_visualization_cb = st.sidebar.checkbox(
    "Show Live Simulation Visual",
    value=st.session_state.get("show_visualization_cb", True), # Default to True, maintain state
    key="show_visualization_checkbox" # Explicit key for robust state
)

# --- Simulation Controls ---
st.header("Simulation Controls")
col1, col2, col3, col4, col5 = st.columns(5)

# Prepare config with its name and visualization setting for use in SimulationManager
selected_config_to_run = selected_config_obj_original.copy()
selected_config_to_run["name"] = st.session_state.current_config_name
selected_config_to_run["visualize"] = st.session_state.show_visualization_cb


if col1.button("🚀 Start Training", disabled=st.session_state.simulation_running, use_container_width=True):
    if st.session_state.sim_manager.state == "stopped":
        st.session_state.results_df = None # Clear previous results
        st.session_state.plot_figures = None # Clear previous plots
        st.session_state.active_simulation_config_name = st.session_state.current_config_name

        # Initialize placeholder for image if visualization is on
        if selected_config_to_run["visualize"]:
            if 'image_placeholder' not in st.session_state or st.session_state.image_placeholder is None:
                # Create the placeholder in the main body, below status
                # This will be filled during status updates.
                pass # Placeholder will be created in the status update section if needed.

        # Run simulation in a thread
        st.session_state.training_thread = threading.Thread(
            target=st.session_state.sim_manager.start,
            args=(selected_config_to_run,)
        )
        st.session_state.training_thread.start()
        st.session_state.simulation_running = True
        st.success(f"Training started for {st.session_state.active_simulation_config_name} in the background...")
        st.rerun() # Fixed: use st.rerun() instead of st.experimental_rerun()
    else:
        st.warning("Simulation is already running or paused. Please stop it first.")


if col2.button("⏸️ Pause", disabled=not st.session_state.simulation_running or st.session_state.sim_manager.state != "running", use_container_width=True):
    st.session_state.sim_manager.pause()
    # simulation_running remains true, but sim_manager.state will be 'paused'
    st.info("Simulation paused.")
    st.rerun() # Fixed

if col3.button("▶️ Resume", disabled=not st.session_state.simulation_running or st.session_state.sim_manager.state != "paused", use_container_width=True):
    st.session_state.sim_manager.resume()
    st.info("Simulation resumed.")
    st.rerun() # Fixed

if col4.button("⏹️ Stop", disabled=st.session_state.sim_manager.state == "stopped", use_container_width=True):
    stop_simulation()
    st.warning("Simulation stopped.")
    st.rerun() # Fixed

if col5.button("🔄 Refresh Status", use_container_width=True):
    st.rerun() # Fixed

# --- Simulation Status (dynamically updates) ---
st.header("Current Status")
status_col, vis_col = st.columns([2,3]) # Create columns for status and visualization

with status_col:
    status_placeholder = st.empty()

if 'image_placeholder' not in st.session_state:
    with vis_col:
        st.session_state.image_placeholder = st.empty()


def update_status_display():
    # Update textual status
    status_data = st.session_state.sim_manager.get_state()
    sim_state = status_data.get("simulation_manager_state", "N/A").capitalize()
    trainer_info = status_data.get("trainer_state", {})

    status_md = f"**Simulation Manager:** {sim_state}"
    if st.session_state.active_simulation_config_name and st.session_state.sim_manager.state != "stopped":
        status_md += f" (Running: *{st.session_state.active_simulation_config_name}*)"

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

    # Progress bar for episodes - outside markdown for better control
    if isinstance(trainer_info, dict) and trainer_info.get("total_episodes", 0) > 0 and st.session_state.sim_manager.state == "running":
        # This progress bar might be better placed directly under the markdown in its column
        st.progress(trainer_info.get("current_episode", 0) / trainer_info.get("total_episodes", 0))

    # Update visualization
    if st.session_state.show_visualization_cb and \
       st.session_state.sim_manager.state == "running" and \
       hasattr(st.session_state.sim_manager, 'trainer') and \
       st.session_state.sim_manager.trainer is not None and \
       hasattr(st.session_state.sim_manager.trainer, 'environment'):

        current_frame = st.session_state.sim_manager.trainer.environment.latest_frame_image
        if current_frame:
            st.session_state.image_placeholder.image(current_frame, caption="Live Simulation", use_column_width=True)
        elif st.session_state.sim_manager.state == "running": # no frame yet, but running
             st.session_state.image_placeholder.caption("Waiting for first frame...")
    elif not st.session_state.show_visualization_cb or st.session_state.sim_manager.state == "stopped":
        st.session_state.image_placeholder.empty() # Clear the placeholder if viz is off or sim stopped

update_status_display()


# --- Handle Thread Completion & Display Results ---
if st.session_state.training_thread and not st.session_state.training_thread.is_alive():
    st.success(f"Training for {st.session_state.active_simulation_config_name} finished!")
    # Retrieve final results once thread is done
    st.session_state.results_df = st.session_state.sim_manager.latest_results_df
    st.session_state.plot_figures = st.session_state.sim_manager.latest_plot_paths # This now holds figures

    st.session_state.training_thread = None
    st.session_state.simulation_running = False
    st.rerun() # Fixed: Rerun to show final results and update button states

# --- Results and Plots ---
st.header("Results and Analysis")
if st.session_state.results_df is not None and not st.session_state.results_df.empty:
    st.info(f"Showing results for the last completed run: **{st.session_state.active_simulation_config_name}**")
    
    # Display plots
    if st.session_state.plot_figures:
        for plot_name, fig in st.session_state.plot_figures.items():
            st.plotly_chart(fig, use_container_width=True)

    # Display dataframe
    with st.expander("View Raw Results Data"):
        st.dataframe(st.session_state.results_df)
else:
    st.info("No results to display. Run a simulation to see the output.")


# Auto-refresh loop while training (optional, use with caution)
# This part is tricky with Streamlit's execution model.
# A dedicated refresh button is safer.
if st.session_state.simulation_running and st.session_state.training_thread and st.session_state.training_thread.is_alive():
    time.sleep(2) # Refresh interval
    st.rerun() # Fixed

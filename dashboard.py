import streamlit as st
from simulate import SimulationManager
from ant_environment import AntEnvironment
from rl_agent import *
from params import config1, config2, config3, config4, config5
import threading
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Page configuration with dark theme
st.set_page_config(
    layout="wide", 
    page_title="SwarmRL Intelligence Hub",
    page_icon="🐜",
    initial_sidebar_state="expanded"
)

# Custom CSS for stunning visuals
st.markdown("""
<style>
    /* Dark theme with gradient backgrounds */
    .stApp {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
    }
    
    /* Glowing headers */
    h1, h2, h3 {
        background: linear-gradient(90deg, #00d4ff, #7b2ff7, #ff007f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-fill-color: transparent;
        animation: gradient 3s ease infinite;
        background-size: 200% 200%;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Neon glow effect for metrics */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.5);
        border-color: #00d4ff;
    }
    
    /* Glowing buttons */
    .stButton > button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 20px 0 rgba(102, 126, 234, 0.6);
    }
    
    /* Status indicators */
    .status-running {
        color: #00ff88;
        text-shadow: 0 0 10px #00ff88;
    }
    
    .status-paused {
        color: #ffaa00;
        text-shadow: 0 0 10px #ffaa00;
    }
    
    .status-stopped {
        color: #ff0055;
        text-shadow: 0 0 10px #ff0055;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00d4ff, #7b2ff7, #ff007f);
        animation: gradient 3s ease infinite;
        background-size: 200% 200%;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(26, 26, 46, 0.9);
        backdrop-filter: blur(10px);
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(0, 212, 255, 0.1);
        border-left: 4px solid #00d4ff;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    /* Success boxes */
    .success-box {
        background: rgba(0, 255, 136, 0.1);
        border-left: 4px solid #00ff88;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    /* Warning boxes */
    .warning-box {
        background: rgba(255, 170, 0, 0.1);
        border-left: 4px solid #ffaa00;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    /* Custom expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Metric animations */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse-animation {
        animation: pulse 2s ease-in-out infinite;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'sim_manager' not in st.session_state:
    st.session_state.sim_manager = SimulationManager()
    st.session_state.simulation_running = False
    st.session_state.current_config_name = "Config 1"
    st.session_state.active_simulation_config_name = None
    st.session_state.training_thread = None
    st.session_state.results_df = None
    st.session_state.plot_figures = None
    st.session_state.metrics_history = []
    st.session_state.start_time = None
    st.session_state.total_episodes_run = 0

# Header with animated title
st.markdown("""
<h1 style="text-align: center; font-size: 3.5em; margin-bottom: 0;">
    🐜 SwarmRL Intelligence Hub
</h1>
<p style="text-align: center; color: #888; font-size: 1.2em; margin-top: 0;">
    Advanced Multi-Agent Reinforcement Learning Simulation Platform
</p>
""", unsafe_allow_html=True)

# Helper functions
def stop_simulation():
    if st.session_state.sim_manager.state != "stopped":
        st.session_state.sim_manager.quit()
    st.session_state.simulation_running = False
    st.session_state.training_thread = None
    st.session_state.active_simulation_config_name = None

def format_time(seconds):
    """Format seconds to human readable time"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

# Sidebar with glassmorphism effect
with st.sidebar:
    st.markdown("""
    <div style="background: rgba(255, 255, 255, 0.05); 
                backdrop-filter: blur(10px); 
                border-radius: 20px; 
                padding: 20px; 
                border: 1px solid rgba(255, 255, 255, 0.1);">
        <h2 style="text-align: center; font-size: 1.8em;">⚙️ Configuration Center</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Configuration selection with custom styling
    config_options = {
        "Config 1": config1,
        "Config 2": config2,
        "Config 3": config3,
        "Config 4": config4,
        "Config 5": config5,
    }
    
    selected_config_name = st.selectbox(
        "🎯 **Select Configuration**",
        list(config_options.keys()),
        index=list(config_options.keys()).index(st.session_state.current_config_name),
        help="Choose a pre-configured simulation setup"
    )
    
    if selected_config_name != st.session_state.current_config_name:
        st.session_state.current_config_name = selected_config_name
        if not st.session_state.simulation_running:
            st.session_state.results_df = None
            st.session_state.plot_figures = None
    
    selected_config = config_options[st.session_state.current_config_name]
    
    # Configuration details in an elegant expander
    with st.expander("📊 Configuration Details", expanded=False):
        config_df = pd.DataFrame([selected_config]).T
        config_df.columns = ["Value"]
        config_df.index.name = "Parameter"
        
        # Highlight important parameters
        st.dataframe(
            config_df.style.background_gradient(cmap="viridis", axis=0),
            use_container_width=True
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visualization toggle with custom styling
    st.markdown("""
    <div style="background: rgba(0, 212, 255, 0.1); 
                border-radius: 15px; 
                padding: 15px; 
                border: 1px solid rgba(0, 212, 255, 0.3);">
        <h3 style="font-size: 1.2em;">🎨 Visualization Settings</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.session_state.show_visualization_cb = st.checkbox(
        "**Enable Live Simulation**",
        value=st.session_state.get("show_visualization_cb", True),
        help="Toggle real-time visualization of ant behavior"
    )
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings", expanded=False):
        st.slider("Visualization FPS", 1, 60, 30, help="Frames per second for visualization")
        st.color_picker("Ant Color", "#FF0000")
        st.color_picker("Food Color", "#00D4FF")
        st.color_picker("Pheromone Color", "#00FF88")

# Main content area
main_container = st.container()

with main_container:
    # Control panel with gradient background
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                border-radius: 20px;
                padding: 20px;
                margin-bottom: 20px;
                border: 1px solid rgba(255, 255, 255, 0.1);">
        <h2 style="text-align: center;">🎮 Simulation Control Center</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Control buttons with custom layout
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    
    selected_config_to_run = selected_config.copy()
    selected_config_to_run["name"] = st.session_state.current_config_name
    selected_config_to_run["visualize"] = st.session_state.show_visualization_cb
    
    with col1:
        if st.button("🚀 **START**", disabled=st.session_state.simulation_running, 
                     use_container_width=True, help="Begin training simulation"):
            if st.session_state.sim_manager.state == "stopped":
                st.session_state.results_df = None
                st.session_state.plot_figures = None
                st.session_state.active_simulation_config_name = st.session_state.current_config_name
                st.session_state.start_time = time.time()
                st.session_state.metrics_history = []
                
                st.session_state.training_thread = threading.Thread(
                    target=st.session_state.sim_manager.start,
                    args=(selected_config_to_run,)
                )
                st.session_state.training_thread.start()
                st.session_state.simulation_running = True
                st.balloons()
                st.rerun()
    
    with col2:
        if st.button("⏸️ **PAUSE**", disabled=not st.session_state.simulation_running or 
                     st.session_state.sim_manager.state != "running", 
                     use_container_width=True, help="Pause current simulation"):
            st.session_state.sim_manager.pause()
            st.rerun()
    
    with col3:
        if st.button("▶️ **RESUME**", disabled=not st.session_state.simulation_running or 
                     st.session_state.sim_manager.state != "paused", 
                     use_container_width=True, help="Resume paused simulation"):
            st.session_state.sim_manager.resume()
            st.rerun()
    
    with col4:
        if st.button("⏹️ **STOP**", disabled=st.session_state.sim_manager.state == "stopped", 
                     use_container_width=True, help="Stop and reset simulation"):
            stop_simulation()
            st.rerun()
    
    with col5:
        if st.button("🔄 **REFRESH**", use_container_width=True, 
                     help="Refresh dashboard status"):
            st.rerun()
    
    # Status display with real-time metrics
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create a beautiful status dashboard
    status_col1, status_col2, status_col3 = st.columns([1, 2, 2])
    
    with status_col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="font-size: 1.5em; text-align: center;">📊 System Status</h3>
        </div>
        """, unsafe_allow_html=True)
        
        status_data = st.session_state.sim_manager.get_state()
        sim_state = status_data.get("simulation_manager_state", "N/A").capitalize()
        
        # Status indicator with color coding
        status_class = f"status-{sim_state.lower()}"
        st.markdown(f"""
        <div style="text-align: center; margin: 20px 0;">
            <h2 class="{status_class}" style="font-size: 2em; margin: 0;">
                {sim_state.upper()}
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Runtime display
        if st.session_state.start_time and st.session_state.simulation_running:
            runtime = time.time() - st.session_state.start_time
            st.metric("⏱️ Runtime", format_time(runtime))
        
        # Episode progress
        trainer_info = status_data.get("trainer_state", {})
        if isinstance(trainer_info, dict):
            current_ep = trainer_info.get("current_episode", 0)
            total_ep = trainer_info.get("total_episodes", 0)
            
            if total_ep > 0:
                progress = current_ep / total_ep
                st.markdown(f"""
                <div style="margin: 20px 0;">
                    <p style="margin-bottom: 5px; color: #ccc;">Episode Progress</p>
                    <div style="background: rgba(255,255,255,0.1); 
                                border-radius: 10px; 
                                overflow: hidden; 
                                height: 30px;
                                position: relative;">
                        <div style="background: linear-gradient(90deg, #00d4ff, #7b2ff7); 
                                    width: {progress*100}%; 
                                    height: 100%; 
                                    transition: width 0.3s ease;
                                    position: absolute;
                                    top: 0;
                                    left: 0;">
                        </div>
                        <div style="position: relative;
                                    height: 100%;
                                    display: flex;
                                    align-items: center;
                                    justify-content: center;
                                    color: white;
                                    font-weight: bold;
                                    text-shadow: 0 0 5px rgba(0,0,0,0.5);">
                            {current_ep}/{total_ep} ({progress:.0%})
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with status_col2:
        # Live visualization
        st.markdown("""
        <div class="metric-card" style="height: 400px;">
            <h3 style="font-size: 1.5em; text-align: center;">🎬 Live Simulation View</h3>
        </div>
        """, unsafe_allow_html=True)
        
        vis_placeholder = st.empty()
        
        if (st.session_state.show_visualization_cb and 
            st.session_state.sim_manager.state == "running" and
            hasattr(st.session_state.sim_manager, 'trainer') and
            st.session_state.sim_manager.trainer is not None and
            hasattr(st.session_state.sim_manager.trainer, 'environment')):
            
            current_frame = st.session_state.sim_manager.trainer.environment.latest_frame_image
            if current_frame:
                vis_placeholder.image(current_frame, use_column_width=True)
            else:
                vis_placeholder.info("🎥 Waiting for simulation frames...")
        else:
            vis_placeholder.info("🎯 Start simulation to see live view")
    
    with status_col3:
        # Real-time metrics
        st.markdown("""
        <div class="metric-card" style="height: 400px;">
            <h3 style="font-size: 1.5em; text-align: center;">📈 Live Performance Metrics</h3>
        </div>
        """, unsafe_allow_html=True)
        
        metrics_placeholder = st.empty()
        
        if trainer_info and isinstance(trainer_info, dict):
            latest_results = trainer_info.get("latest_results", {})
            if latest_results:
                # Create a mini dashboard of metrics
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("🏆 Total Reward", f"{latest_results.get('total_reward', 0):.0f}", 
                             delta=f"+{latest_results.get('total_reward', 0)*0.1:.0f}")
                    st.metric("🍎 Food Collected", latest_results.get('food_collected', 0))
                with metric_col2:
                    st.metric("👣 Steps", latest_results.get('steps', 0))
                    st.metric("🌟 Efficiency", f"{latest_results.get('pheromone_trail_usage', 0):.2%}")
                
                # Mini live chart
                if st.session_state.metrics_history:
                    df = pd.DataFrame(st.session_state.metrics_history[-20:])  # Last 20 episodes
                    fig = px.line(df, y='reward', title='Recent Rewards Trend',
                                 template='plotly_dark', height=200)
                    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig, use_container_width=True)
    
    # Results and Analysis Section
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Check if training completed
    if st.session_state.training_thread and not st.session_state.training_thread.is_alive():
        st.success("🎉 Training completed successfully!")
        st.session_state.results_df = st.session_state.sim_manager.latest_results_df
        st.session_state.plot_figures = st.session_state.sim_manager.latest_plot_paths
        st.session_state.training_thread = None
        st.session_state.simulation_running = False
        st.session_state.total_episodes_run += len(st.session_state.results_df) if st.session_state.results_df is not None else 0
        st.rerun()
    
    # Results display
    if st.session_state.results_df is not None and not st.session_state.results_df.empty:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(0, 255, 136, 0.1) 0%, rgba(0, 212, 255, 0.1) 100%);
                    border-radius: 20px;
                    padding: 20px;
                    margin-top: 40px;
                    border: 1px solid rgba(255, 255, 255, 0.1);">
            <h2 style="text-align: center;">🏆 Training Results & Analysis</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Summary statistics
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card pulse-animation">
                <h4 style="text-align: center;">Total Episodes</h4>
                <h2 style="text-align: center; font-size: 2.5em; margin: 0;">
                    {}</h2>
            </div>
            """.format(len(st.session_state.results_df)), unsafe_allow_html=True)
        
        with col2:
            avg_reward = st.session_state.results_df['total_reward'].mean()
            st.markdown("""
            <div class="metric-card pulse-animation">
                <h4 style="text-align: center;">Avg Reward</h4>
                <h2 style="text-align: center; font-size: 2.5em; margin: 0;">
                    {:.1f}</h2>
            </div>
            """.format(avg_reward), unsafe_allow_html=True)
        
        with col3:
            max_reward = st.session_state.results_df['total_reward'].max()
            st.markdown("""
            <div class="metric-card pulse-animation">
                <h4 style="text-align: center;">Peak Reward</h4>
                <h2 style="text-align: center; font-size: 2.5em; margin: 0;">
                    {:.0f}</h2>
            </div>
            """.format(max_reward), unsafe_allow_html=True)
        
        with col4:
            improvement = (st.session_state.results_df['total_reward'].iloc[-10:].mean() - 
                          st.session_state.results_df['total_reward'].iloc[:10].mean())
            st.markdown("""
            <div class="metric-card pulse-animation">
                <h4 style="text-align: center;">Improvement</h4>
                <h2 style="text-align: center; font-size: 2.5em; margin: 0; 
                    color: {};">+{:.0f}%</h2>
            </div>
            """.format("#00ff88" if improvement > 0 else "#ff0055", 
                      abs(improvement/st.session_state.results_df['total_reward'].iloc[:10].mean()*100)), 
                      unsafe_allow_html=True)
        
        # Interactive plots
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.session_state.plot_figures:
            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance Trends", "🎯 Detailed Metrics", 
                                              "🔬 Analysis", "📁 Raw Data"])
            
            with tab1:
                for plot_name, fig in st.session_state.plot_figures.items():
                    # Enhance the plot styling
                    fig.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(size=14),
                        title_font_size=20,
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # Create additional visualizations
                df = st.session_state.results_df
                
                # Efficiency over time
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=df['steps'],
                    mode='lines',
                    name='Steps per Episode',
                    line=dict(color='#00d4ff', width=2)
                ))
                fig.update_layout(
                    title='Episode Efficiency (Lower is Better)',
                    template='plotly_dark',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Food collection rate
                if 'food_collected' in df.columns:
                    fig = px.bar(df.tail(50), y='food_collected', 
                                title='Food Collection (Last 50 Episodes)',
                                template='plotly_dark', height=400)
                    fig.update_traces(marker_color='#7b2ff7')
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                # Statistical analysis
                st.markdown("### 📊 Statistical Summary")
                
                summary_df = df.describe()
                st.dataframe(
                    summary_df.style.background_gradient(cmap='viridis'),
                    use_container_width=True
                )
                
                # Correlation matrix
                if len(df.columns) > 1:
                    st.markdown("### 🔗 Feature Correlations")
                    corr = df.corr()
                    fig = px.imshow(corr, template='plotly_dark', 
                                   color_continuous_scale='RdBu_r',
                                   title='Correlation Heatmap')
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                st.markdown("### 📁 Raw Training Data")
                st.dataframe(
                    df.style.background_gradient(cmap='viridis', axis=0),
                    use_container_width=True,
                    height=400
                )
                
                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f'swarmrl_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv'
                )

# Auto-refresh while training
if st.session_state.simulation_running and st.session_state.training_thread and st.session_state.training_thread.is_alive():
    time.sleep(1)
    st.rerun()

# Footer
st.markdown("""
<br><br>
<div style="text-align: center; color: #666; padding: 20px;">
    <p>SwarmRL Intelligence Hub v2.0 | Built with ❤️ for Multi-Agent RL Research</p>
</div>
""", unsafe_allow_html=True)
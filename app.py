import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import threading
import queue
import json
import os
import logging

from enhanced_ekf_tracker import EnhancedEKFTracker
from validation_framework import ValidationFramework
from space_weather import SpaceWeatherData
from utils import parse_tle, format_time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Page configuration
st.set_page_config(
    page_title="Enhanced Orbital Determination System",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'tracker' not in st.session_state:
    st.session_state.tracker = None
if 'tracking_active' not in st.session_state:
    st.session_state.tracking_active = False
if 'tracking_data' not in st.session_state:
    st.session_state.tracking_data = []
if 'validation_results' not in st.session_state:
    st.session_state.validation_results = None
if 'space_weather' not in st.session_state:
    st.session_state.space_weather = SpaceWeatherData()
if 'update_placeholder' not in st.session_state:
    st.session_state.update_placeholder = None

def main():
    st.title("🛰️ Enhanced Orbital Determination System")
    st.markdown("*Advanced EKF with sub-1km accuracy targeting*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("System Configuration")
        
        # Satellite selection
        st.subheader("Satellite Selection")
        satellite_type = st.selectbox(
            "Select Satellite",
            ["ISS (International Space Station)", "Custom TLE"]
        )
        
        if satellite_type == "Custom TLE":
            tle_line1 = st.text_input("TLE Line 1")
            tle_line2 = st.text_input("TLE Line 2")
        else:
            # Default ISS TLE (recent)
            tle_line1 = "1 25544U 98067A   25257.50000000  .00002182  00000+0  40864-4 0  9990"
            tle_line2 = "2 25544  51.6461 216.5824 0001234  85.3421 274.8071 15.48919103123456"
        
        # EKF Configuration
        st.subheader("EKF Parameters")
        drag_coeff = st.slider("Drag Coefficient (CdA)", 0.5, 5.0, 2.2, 0.1)
        srp_coeff = st.slider("SRP Coefficient (Cr)", 0.5, 2.5, 1.3, 0.1)
        process_noise = st.slider("Process Noise Scale", 0.1, 10.0, 1.0, 0.1)
        
        # Force Model Configuration
        st.subheader("Force Models")
        use_j2_j6 = st.checkbox("J2-J6 Harmonics", value=True)
        use_drag = st.checkbox("Atmospheric Drag", value=True)
        use_srp = st.checkbox("Solar Radiation Pressure", value=True)
        use_nrlmsise = st.checkbox("NRLMSISE-00 Atmosphere", value=True)
        
        # Advanced Features
        st.subheader("Advanced Features")
        use_batch_estimation = st.checkbox("Batch Parameter Estimation", value=True)
        use_rts_smoother = st.checkbox("RTS Smoother", value=True)
        use_adaptive_filtering = st.checkbox("Adaptive Q/R Matrices", value=True)
        use_ml_corrector = st.checkbox("ML Residual Corrector", value=False)
        
        # Space Weather
        st.subheader("Space Weather")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Update Space Weather"):
                with st.spinner("Fetching space weather data..."):
                    st.session_state.space_weather.update()
                st.success("Space weather updated!")
        
        with col2:
            sw_data = st.session_state.space_weather.get_current_data()
            st.metric("F10.7", f"{sw_data['f107']:.1f}")
            st.metric("Kp", f"{sw_data['kp']:.1f}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Tracking Control")
        
        col_start, col_stop, col_validate = st.columns(3)
        
        with col_start:
            if st.button("Start Tracking", disabled=st.session_state.tracking_active):
                if tle_line1 and tle_line2:
                    try:
                        # Initialize tracker
                        config = {
                            'drag_coeff': drag_coeff,
                            'srp_coeff': srp_coeff,
                            'process_noise_scale': process_noise,
                            'use_j2_j6': use_j2_j6,
                            'use_drag': use_drag,
                            'use_srp': use_srp,
                            'use_nrlmsise': use_nrlmsise,
                            'use_batch_estimation': use_batch_estimation,
                            'use_rts_smoother': use_rts_smoother,
                            'use_adaptive_filtering': use_adaptive_filtering,
                            'use_ml_corrector': use_ml_corrector,
                            'satellite_mass': 464291.0  # ISS mass
                        }
                        
                        st.session_state.tracker = EnhancedEKFTracker(
                            tle_line1, tle_line2, config
                        )
                        st.session_state.tracking_active = True
                        st.session_state.tracking_data = []
                        st.success("Tracking started!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Failed to start tracking: {str(e)}")
                else:
                    st.error("Please provide valid TLE data")
        
        with col_stop:
            if st.button("Stop Tracking", disabled=not st.session_state.tracking_active):
                st.session_state.tracking_active = False
                st.session_state.tracker = None
                st.info("Tracking stopped")
                st.rerun()
        
        with col_validate:
            if st.button("Run Validation"):
                if len(st.session_state.tracking_data) > 0:
                    with st.spinner("Running validation against OEM data..."):
                        validator = ValidationFramework()
                        if validator.load_oem_data("data/iss_oem_data.txt"):
                            results = validator.validate_tracking_data(
                                st.session_state.tracking_data
                            )
                            st.session_state.validation_results = results
                            st.success("Validation complete!")
                        else:
                            st.warning("Using synthetic validation - OEM data not available")
                            results = validator.validate_tracking_data(
                                st.session_state.tracking_data
                            )
                            st.session_state.validation_results = results
                else:
                    st.warning("No tracking data available for validation")
    
    with col2:
        st.subheader("System Status")
        
        # Real-time status
        if st.session_state.tracking_active:
            st.success("🟢 Tracking Active")
            if st.session_state.tracker:
                status = st.session_state.tracker.get_status()
                st.metric("Filter Iterations", status.get('iterations', 0))
                st.metric("Last Update", status.get('last_update', 'Never'))
                st.metric("Covariance Trace", f"{status.get('cov_trace', 0):.2e}")
                st.metric("Divergence Count", status.get('divergence_count', 0))
        else:
            st.info("🔴 Tracking Inactive")
        
        # Data collection status
        st.metric("Data Points", len(st.session_state.tracking_data))
        
        # Validation status
        if st.session_state.validation_results:
            results = st.session_state.validation_results
            if 'metrics' in results:
                metrics = results['metrics']
                st.metric("RMS Error (m)", f"{metrics.get('rms_error', 0):.1f}")
                st.metric("P95 Error (m)", f"{metrics.get('p95_error', 0):.1f}")
                st.metric("% < 1km", f"{metrics.get('percent_under_1km', 0):.1f}%")
    
    # Real-time tracking simulation
    if st.session_state.tracking_active and st.session_state.tracker:
        # Create placeholders for real-time updates
        if st.session_state.update_placeholder is None:
            st.session_state.update_placeholder = st.empty()
        
        # Simulate periodic updates
        current_time = datetime.utcnow()
        tracking_result = st.session_state.tracker.update(current_time)
        
        if tracking_result:
            st.session_state.tracking_data.append(tracking_result)
            
            # Keep only last 1000 points for display
            if len(st.session_state.tracking_data) > 1000:
                st.session_state.tracking_data = st.session_state.tracking_data[-1000:]
            
            # Display current results
            with st.session_state.update_placeholder.container():
                display_tracking_results(tracking_result)
                display_tracking_plots()
        
        # Auto-refresh every 10 seconds when tracking is active
        time.sleep(1)
        st.rerun()
    
    elif len(st.session_state.tracking_data) > 0:
        # Display historical data
        display_tracking_plots()
    
    # Validation results display
    if st.session_state.validation_results:
        display_validation_results()

def display_tracking_results(result):
    """Display current tracking results"""
    st.subheader("Current State Estimate")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Altitude (km)", f"{result['altitude']:.2f}")
        st.metric("Velocity (km/s)", f"{result['velocity_magnitude']:.3f}")
    
    with col2:
        st.metric("Latitude (°)", f"{result['latitude']:.4f}")
        st.metric("Longitude (°)", f"{result['longitude']:.4f}")
    
    with col3:
        st.metric("Drag Coeff", f"{result['drag_coeff']:.3f}")
        st.metric("SRP Coeff", f"{result['srp_coeff']:.3f}")

def display_tracking_plots():
    """Display tracking visualization plots"""
    if len(st.session_state.tracking_data) == 0:
        return
    
    df = pd.DataFrame(st.session_state.tracking_data)
    
    st.subheader("Tracking Visualization")
    
    # Tabs for different plots
    tab1, tab2, tab3, tab4 = st.tabs(["Orbit Plot", "Error Analysis", "Parameters", "Innovation"])
    
    with tab1:
        # 3D orbit plot
        fig = go.Figure()
        
        if len(df) > 0:
            positions = np.array(df['position_eci'].tolist())
            fig.add_trace(go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode='lines+markers',
                name='EKF Estimate',
                line=dict(color='blue', width=3),
                marker=dict(size=2)
            ))
        
        # Add Earth sphere
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        earth_radius = 6371000  # m
        x_earth = earth_radius * np.outer(np.cos(u), np.sin(v))
        y_earth = earth_radius * np.outer(np.sin(u), np.sin(v))
        z_earth = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig.add_trace(go.Surface(
            x=x_earth, y=y_earth, z=z_earth,
            colorscale='Blues',
            showscale=False,
            opacity=0.3,
            name='Earth'
        ))
        
        fig.update_layout(
            title="Orbital Trajectory",
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)",
                aspectmode='cube'
            ),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Error analysis plots
        if 'position_error' in df.columns:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['position_error'],
                mode='lines',
                name='Position Error',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title="Position Error Over Time",
                xaxis_title="Time",
                yaxis_title="Error (m)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Error data not available (requires validation)")
    
    with tab3:
        # Parameter evolution
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['drag_coeff'],
            mode='lines',
            name='Drag Coefficient',
            line=dict(color='green')
        ))
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['srp_coeff'],
            mode='lines',
            name='SRP Coefficient',
            line=dict(color='orange'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="Parameter Evolution",
            xaxis_title="Time",
            yaxis_title="Drag Coefficient",
            yaxis2=dict(
                title="SRP Coefficient",
                overlaying='y',
                side='right'
            ),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Innovation analysis
        if 'innovation_norm' in df.columns:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['innovation_norm'],
                mode='lines',
                name='Innovation Norm',
                line=dict(color='purple')
            ))
            
            # Add theoretical bounds
            dof = 6  # Position and velocity
            theoretical_mean = dof
            theoretical_std = np.sqrt(2 * dof)
            
            fig.add_hline(y=theoretical_mean, line_dash="dash", 
                         annotation_text="Theoretical Mean")
            fig.add_hline(y=theoretical_mean + 2*theoretical_std, line_dash="dot",
                         annotation_text="2σ Upper Bound")
            
            fig.update_layout(
                title="Innovation Statistics",
                xaxis_title="Time",
                yaxis_title="Normalized Innovation",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Innovation data not available")

def display_validation_results():
    """Display comprehensive validation results"""
    st.subheader("Validation Results")
    
    results = st.session_state.validation_results
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    if 'metrics' in results:
        metrics = results['metrics']
        
        with col1:
            rms_error = metrics.get('rms_error', 0)
            delta_text = "Target: <500m" if rms_error > 500 else "✓ Target met"
            st.metric(
                "RMS Error", 
                f"{rms_error:.1f} m",
                delta=delta_text
            )
        
        with col2:
            p95_error = metrics.get('p95_error', 0)
            delta_text = "Target: <1000m" if p95_error > 1000 else "✓ Target met"
            st.metric(
                "P95 Error", 
                f"{p95_error:.1f} m",
                delta=delta_text
            )
        
        with col3:
            percent_1km = metrics.get('percent_under_1km', 0)
            delta_text = "Target: >90%" if percent_1km < 90 else "✓ Target met"
            st.metric(
                "% Under 1km", 
                f"{percent_1km:.1f}%",
                delta=delta_text
            )
        
        with col4:
            max_error = metrics.get('max_error', 0)
            st.metric(
                "Max Error", 
                f"{max_error:.1f} m"
            )
    
    # Detailed plots
    if 'analysis' in results and 'error_time_series' in results['analysis']:
        error_data = results['analysis']['error_time_series']
        if 'errors' in error_data and len(error_data['errors']) > 0:
            fig = go.Figure()
            
            timestamps = error_data.get('timestamps', [])
            errors = error_data['errors']
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=errors,
                mode='lines',
                name='Position Error',
                line=dict(color='red')
            ))
            
            # Add 1km threshold line
            fig.add_hline(y=1000, line_dash="dash", line_color="orange",
                         annotation_text="1km Threshold")
            
            fig.update_layout(
                title="Validation Error Time Series",
                xaxis_title="Time",
                yaxis_title="Position Error (m)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Error histogram
    if 'analysis' in results and 'error_histogram' in results['analysis']:
        hist_data = results['analysis']['error_histogram']
        if 'errors' in hist_data and len(hist_data['errors']) > 0:
            fig = px.histogram(
                x=hist_data['errors'],
                bins=50,
                title="Error Distribution",
                labels={'x': 'Position Error (m)', 'y': 'Frequency'}
            )
            
            fig.add_vline(x=1000, line_dash="dash", line_color="orange",
                         annotation_text="1km Target")
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Performance assessment
    if 'analysis' in results and 'overall_performance' in results['analysis']:
        performance = results['analysis']['overall_performance']
        
        st.subheader("Performance Assessment")
        
        grade = performance.get('overall_grade', 'Unknown')
        if grade == 'Excellent':
            st.success(f"Overall Grade: {grade} 🌟")
        elif grade == 'Good':
            st.success(f"Overall Grade: {grade} ✅")
        elif grade == 'Acceptable':
            st.warning(f"Overall Grade: {grade} ⚠️")
        else:
            st.error(f"Overall Grade: {grade} ❌")
        
        # Show target compliance
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if performance.get('rms_meets_target', False):
                st.success("RMS Target: ✅ Met")
            else:
                st.error("RMS Target: ❌ Not Met")
        
        with col2:
            if performance.get('p95_meets_target', False):
                st.success("P95 Target: ✅ Met") 
            else:
                st.error("P95 Target: ❌ Not Met")
        
        with col3:
            if performance.get('accuracy_meets_target', False):
                st.success("Accuracy Target: ✅ Met")
            else:
                st.error("Accuracy Target: ❌ Not Met")
    
    # Recommendations
    if 'analysis' in results and 'recommendations' in results['analysis']:
        recommendations = results['analysis']['recommendations']
        if recommendations:
            st.subheader("Recommendations")
            for rec in recommendations:
                st.info(f"💡 {rec}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
True EKF Satellite Tracker Dashboard
===================================
Interactive Streamlit dashboard for visualizing EKF vs SGP4 vs NASA OEM trajectories
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import json
import threading
import queue
import warnings
warnings.filterwarnings('ignore')
from skyfield.api import utc

from true_ekf_tracker import TrueEKFTracker
from satellite_data import SatelliteDataManager


# Page configuration
st.set_page_config(
    page_title="True EKF Satellite Tracker",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'tracker' not in st.session_state:
    st.session_state.tracker = None
if 'tracking_data' not in st.session_state:
    st.session_state.tracking_data = []
if 'is_tracking' not in st.session_state:
    st.session_state.is_tracking = False
if 'tracking_thread' not in st.session_state:
    st.session_state.tracking_thread = None
if 'data_queue' not in st.session_state:
    st.session_state.data_queue = queue.Queue()
if 'stop_flag' not in st.session_state:
    st.session_state.stop_flag = {'running': False}


def load_satellite_info():
    """Load satellite information from JSON"""
    try:
        with open('selected_satellite.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def initialize_tracker():
    """Initialize the EKF tracker"""
    if st.session_state.tracker is None:
        st.session_state.tracker = TrueEKFTracker()
        return st.session_state.tracker.initialize_ekf()
    return True


def tracking_worker(tracker, duration_minutes, update_interval, data_queue, stop_flag):
    """Background tracking worker"""
    start_time = datetime.now(utc)
    end_time = start_time + timedelta(minutes=duration_minutes)
    
    current_time = start_time
    
    while current_time < end_time and stop_flag['running']:
        # Perform tracking step
        result = tracker.track_step(current_time)
        
        if result:
            # Put result in queue for main thread
            data_queue.put(result)
        
        # Wait for next update
        time.sleep(update_interval)
        current_time = datetime.now(utc)
    
    # Signal completion
    data_queue.put(None)


def create_trajectory_map(tracking_data):
    """Create trajectory map visualization"""
    if not tracking_data:
        return None
    
    df = pd.DataFrame(tracking_data)
    
    # Create map
    fig = go.Figure()
    
    # EKF trajectory
    ekf_lats = [d['ekf_position']['latitude'] for d in tracking_data]
    ekf_lons = [d['ekf_position']['longitude'] for d in tracking_data]
    ekf_alts = [d['ekf_position']['altitude'] for d in tracking_data]
    
    fig.add_trace(go.Scattergeo(
        lat=ekf_lats,
        lon=ekf_lons,
        mode='markers+lines',
        name='EKF Trajectory',
        line=dict(color='blue', width=3),
        marker=dict(size=6, color='blue'),
        hovertemplate='<b>EKF</b><br>' +
                     'Lat: %{lat:.4f}°<br>' +
                     'Lon: %{lon:.4f}°<br>' +
                     'Alt: %{text:.1f}km<extra></extra>',
        text=ekf_alts
    ))
    
    # SGP4 trajectory
    sgp4_lats = [d['sgp4_position']['latitude'] for d in tracking_data]
    sgp4_lons = [d['sgp4_position']['longitude'] for d in tracking_data]
    sgp4_alts = [d['sgp4_position']['altitude'] for d in tracking_data]
    
    fig.add_trace(go.Scattergeo(
        lat=sgp4_lats,
        lon=sgp4_lons,
        mode='markers+lines',
        name='SGP4 Reference',
        line=dict(color='red', width=2, dash='dash'),
        marker=dict(size=4, color='red'),
        hovertemplate='<b>SGP4</b><br>' +
                     'Lat: %{lat:.4f}°<br>' +
                     'Lon: %{lon:.4f}°<br>' +
                     'Alt: %{text:.1f}km<extra></extra>',
        text=sgp4_alts
    ))
    
    # Current position (latest point)
    if tracking_data:
        latest = tracking_data[-1]
        fig.add_trace(go.Scattergeo(
            lat=[latest['ekf_position']['latitude']],
            lon=[latest['ekf_position']['longitude']],
            mode='markers',
            name='Current Position',
            marker=dict(size=12, color='green', symbol='star'),
            hovertemplate='<b>Current EKF Position</b><br>' +
                         f"Lat: {latest['ekf_position']['latitude']:.4f}°<br>" +
                         f"Lon: {latest['ekf_position']['longitude']:.4f}°<br>" +
                         f"Alt: {latest['ekf_position']['altitude']:.1f}km<extra></extra>"
        ))
    
    fig.update_layout(
        title="Satellite Trajectory: EKF vs SGP4",
        geo=dict(
            showland=True,
            landcolor='lightgray',
            coastlinecolor='white',
            showocean=True,
            oceancolor='lightblue',
            projection_type='natural earth'
        ),
        height=500
    )
    
    return fig


def create_accuracy_plot(tracking_data):
    """Create accuracy comparison plot"""
    if not tracking_data:
        return None
    
    timestamps = [d['timestamp'] for d in tracking_data]
    ekf_uncertainty = [d['ekf_uncertainty_m'] for d in tracking_data]
    ekf_vs_sgp4_error = [d['ekf_vs_sgp4_error_m'] for d in tracking_data]
    
    fig = go.Figure()
    
    # EKF uncertainty
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=ekf_uncertainty,
        mode='lines+markers',
        name='EKF Uncertainty (3σ)',
        line=dict(color='blue'),
        marker=dict(size=4)
    ))
    
    # EKF vs SGP4 error
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=ekf_vs_sgp4_error,
        mode='lines+markers',
        name='EKF vs SGP4 Error',
        line=dict(color='red', dash='dash'),
        marker=dict(size=4)
    ))
    
    # Add OEM validation if available
    oem_errors = [d['oem_validation']['error_m'] if d['oem_validation'] else None for d in tracking_data]
    if any(e is not None for e in oem_errors):
        # Filter out None values
        oem_timestamps = [timestamps[i] for i, e in enumerate(oem_errors) if e is not None]
        oem_values = [e for e in oem_errors if e is not None]
        
        fig.add_trace(go.Scatter(
            x=oem_timestamps,
            y=oem_values,
            mode='markers',
            name='EKF vs NASA OEM Error',
            marker=dict(color='green', size=6, symbol='diamond')
        ))
    
    fig.update_layout(
        title="Tracking Accuracy Over Time",
        xaxis_title="Time (UTC)",
        yaxis_title="Error (meters)",
        yaxis_type="log",
        height=400
    )
    
    return fig


def create_performance_metrics(tracking_data):
    """Create performance metrics display"""
    if not tracking_data or not st.session_state.tracker:
        return None
    
    perf = st.session_state.tracker.get_performance_summary()
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Predictions Made", perf.get('predictions_made', 0))
        st.metric("Measurements Taken", perf.get('measurements_processed', 0))
    
    with col2:
        if 'ekf_mean_uncertainty_m' in perf:
            st.metric("Mean EKF Uncertainty", f"{perf['ekf_mean_uncertainty_m']:.0f}m")
        if 'ekf_sub_1km_rate' in perf:
            st.metric("Sub-1km Rate", f"{perf['ekf_sub_1km_rate']:.1f}%")
    
    with col3:
        if 'ekf_vs_sgp4_mean_error_m' in perf:
            st.metric("Mean EKF vs SGP4 Error", f"{perf['ekf_vs_sgp4_mean_error_m']:.0f}m")
        if 'measurement_rate' in perf:
            st.metric("Measurement Rate", f"{perf['measurement_rate']:.1f}%")
    
    with col4:
        if 'oem_validation_points' in perf:
            st.metric("OEM Validation Points", perf['oem_validation_points'])
        if 'oem_mean_error_m' in perf:
            st.metric("Mean EKF vs OEM Error", f"{perf['oem_mean_error_m']:.0f}m")
    
    return perf


# Main dashboard
def main():
    st.title("🛰️ True Extended Kalman Filter Satellite Tracker")
    st.markdown("---")
    
    # Load satellite info
    sat_info = load_satellite_info()
    if not sat_info:
        st.error("❌ No satellite data found. Please ensure selected_satellite.json exists.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("📡 Satellite Information")
        st.markdown(f"**Name:** {sat_info['name']}")
        st.markdown(f"**NORAD ID:** {sat_info['norad_id']}")
        st.markdown(f"**Altitude:** {sat_info['altitude_km']:.1f} km")
        st.markdown(f"**TLE Age:** {sat_info['tle_age_hours']:.1f} hours")
        st.markdown(f"**Expected Accuracy:** ±{sat_info['expected_accuracy_m']}m")
        
        st.markdown("---")
        st.header("⚙️ Tracking Configuration")
        
        duration_minutes = st.slider("Duration (minutes)", 5, 60, 15)
        update_interval = st.slider("Update Interval (seconds)", 1, 30, 5)
        measurement_interval = st.slider("Measurement Interval (seconds)", 60, 600, 300)
        
        if st.button("🔧 Initialize Tracker"):
            with st.spinner("Initializing EKF tracker..."):
                if initialize_tracker():
                    st.session_state.tracker.measurement_interval = measurement_interval
                    st.success("✅ Tracker initialized successfully!")
                    
                    # Setup OEM validation
                    with st.spinner("Setting up NASA OEM validation..."):
                        if st.session_state.tracker.setup_oem_validation(duration_minutes/60):
                            st.success("✅ OEM validation ready!")
                        else:
                            st.warning("⚠️ OEM validation unavailable, using synthetic data")
                else:
                    st.error("❌ Failed to initialize tracker")
        
        st.markdown("---")
        
        # Tracking controls
        if st.session_state.tracker:
            if not st.session_state.is_tracking:
                if st.button("🚀 Start Tracking"):
                    st.session_state.is_tracking = True
                    st.session_state.tracking_data = []
                    st.session_state.stop_flag['running'] = True
                    
                    # Start background tracking
                    st.session_state.tracking_thread = threading.Thread(
                        target=tracking_worker,
                        args=(st.session_state.tracker, duration_minutes, update_interval, st.session_state.data_queue, st.session_state.stop_flag)
                    )
                    st.session_state.tracking_thread.start()
                    st.rerun()
            else:
                if st.button("⏹️ Stop Tracking"):
                    st.session_state.is_tracking = False
                    st.session_state.stop_flag['running'] = False
                    st.rerun()
    
    # Main content
    if st.session_state.is_tracking:
        st.success("🔄 Live tracking active...")
        
        # Process data from background thread
        while not st.session_state.data_queue.empty():
            data = st.session_state.data_queue.get()
            if data is None:  # End of tracking
                st.session_state.is_tracking = False
                st.rerun()
            else:
                st.session_state.tracking_data.append(data)
        
        # Auto-refresh every few seconds
        time.sleep(2)
        st.rerun()
    
    # Display results if we have data
    if st.session_state.tracking_data:
        st.header("📊 Real-Time Results")
        
        # Performance metrics
        create_performance_metrics(st.session_state.tracking_data)
        
        st.markdown("---")
        
        # Two columns for visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🗺️ Trajectory Visualization")
            trajectory_fig = create_trajectory_map(st.session_state.tracking_data)
            if trajectory_fig:
                st.plotly_chart(trajectory_fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Accuracy Analysis")
            accuracy_fig = create_accuracy_plot(st.session_state.tracking_data)
            if accuracy_fig:
                st.plotly_chart(accuracy_fig, use_container_width=True)
        
        # Latest tracking result
        if st.session_state.tracking_data:
            st.markdown("---")
            st.subheader("📍 Latest Position")
            
            latest = st.session_state.tracking_data[-1]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**EKF Position:**")
                st.write(f"Lat: {latest['ekf_position']['latitude']:.6f}°")
                st.write(f"Lon: {latest['ekf_position']['longitude']:.6f}°")
                st.write(f"Alt: {latest['ekf_position']['altitude']:.3f} km")
                st.write(f"Uncertainty: ±{latest['ekf_uncertainty_m']:.0f}m")
            
            with col2:
                st.markdown("**SGP4 Position:**")
                st.write(f"Lat: {latest['sgp4_position']['latitude']:.6f}°")
                st.write(f"Lon: {latest['sgp4_position']['longitude']:.6f}°")
                st.write(f"Alt: {latest['sgp4_position']['altitude']:.3f} km")
                st.write(f"EKF vs SGP4: {latest['ekf_vs_sgp4_error_m']:.0f}m")
            
            with col3:
                st.markdown("**Tracking Status:**")
                st.write(f"Predictions: {latest['predictions_made']}")
                st.write(f"Measurements: {latest['measurements_processed']}")
                if latest['measurement_taken']:
                    st.success("📡 Measurement taken")
                else:
                    st.info("🔮 Pure prediction")
                
                if latest['oem_validation']:
                    oem = latest['oem_validation']
                    st.write(f"OEM Error: {oem['error_m']:.0f}m")
                    if oem['within_tolerance']:
                        st.success("✅ OEM validated")
                    else:
                        st.warning("⚠️ OEM deviation")
    
    elif st.session_state.tracker:
        st.info("🎯 Tracker initialized and ready. Click 'Start Tracking' to begin.")
    else:
        st.info("🔧 Please initialize the tracker first.")
    
    # Information panel
    st.markdown("---")
    with st.expander("ℹ️ About True EKF Tracking"):
        st.markdown("""
        ### 🔬 How This Works
        
        This is a **True Extended Kalman Filter** that:
        
        1. **🚀 Independent Prediction**: Uses real orbital mechanics (2-body + J2 perturbations + atmospheric drag) to predict satellite positions
        
        2. **📡 Occasional Measurements**: Takes SGP4 measurements every 5 minutes (configurable), not every timestep
        
        3. **🔄 Pure EKF Operation**: Between measurements, relies entirely on orbital mechanics predictions
        
        4. **🌍 Proper Coordinates**: Performs correct ECI → ECEF → Geodetic transformations
        
        5. **✅ NASA Validation**: Compares results against NASA OEM ephemerides when available
        
        ### 📊 What You're Seeing
        
        - **Blue Line**: EKF predictions using orbital mechanics
        - **Red Dashed Line**: SGP4 reference trajectory  
        - **Green Diamonds**: NASA OEM validation points
        - **Uncertainty Bands**: 3-sigma confidence intervals
        
        ### 🎯 Expected Performance
        
        - **Sub-kilometer accuracy** over extended periods
        - **Independent operation** between measurements
        - **Realistic uncertainty estimates** based on orbital mechanics
        """)


if __name__ == "__main__":
    main()

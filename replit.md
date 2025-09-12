# True EKF Satellite Tracker

## Overview

This is a Python-based satellite tracking system that implements an Extended Kalman Filter (EKF) for real-time satellite trajectory prediction. The system uses Streamlit for an interactive web dashboard and compares EKF predictions against SGP4 model calculations and NASA OEM (Orbital Ephemeris Message) data for accuracy validation.

**Current State**: Fully configured and running in Replit environment with all dependencies installed and working correctly.

## User Preferences

**Preferred communication style**: Simple, everyday language for non-technical users.

## Recent Changes

**Date**: September 12, 2025
- **Dependencies Installed**: All Python packages installed via pyproject.toml (numpy, pandas, plotly, requests, scipy, skyfield, streamlit)
- **Streamlit Configuration**: Configured for Replit environment with proper host settings (port 5000, 0.0.0.0 binding)
- **Workflow Setup**: Created streamlit_app workflow for continuous running
- **Deployment Ready**: Configured for autoscale deployment with proper production settings

## Project Architecture

### Frontend Architecture
- **Framework**: Streamlit web dashboard
- **Port**: 5000 (configured for Replit environment)
- **Host**: 0.0.0.0 with CORS disabled for proxy compatibility
- **Features**: Interactive trajectory visualization, real-time tracking dashboard, accuracy comparison charts

### Core Components
1. **app.py** - Main Streamlit application with dashboard interface
2. **true_ekf_tracker.py** - Extended Kalman Filter implementation with orbital mechanics
3. **orbital_mechanics.py** - Pure orbital physics calculations (2-body + J2 perturbations + drag)
4. **coordinate_transforms.py** - ECI/ECEF/Geodetic coordinate system transformations
5. **satellite_data.py** - TLE data management and SGP4 calculations
6. **nasa_oem_validator.py** - NASA ephemeris data validation system

### External Dependencies
- **Scientific Libraries**: numpy, scipy, pandas for mathematical operations
- **Visualization**: plotly for interactive charts and trajectory maps
- **Astronomical**: skyfield for TLE parsing and astronomical calculations
- **Web Framework**: streamlit for dashboard interface
- **APIs**: NASA JPL Horizons API for validation data, Celestrak for TLE feeds

### Data Sources
- **ISS_OEM_J2K.txt** - NASA ISS ephemeris data (5,405 validation points)
- **selected_satellite.json** - ISS configuration and metadata
- **Real-time TLE feeds** from Celestrak for current orbital elements

## Deployment Configuration

- **Type**: Autoscale (stateless web application)
- **Command**: `streamlit run app.py --server.port=5000 --server.address=0.0.0.0`
- **Environment**: Production-ready with proper Replit proxy configuration

## Workflow Configuration

- **Name**: streamlit_app
- **Command**: `streamlit run app.py`
- **Port**: 5000
- **Status**: Running and accessible via Replit web preview

## Technical Features

### Extended Kalman Filter Implementation
- **True EKF**: Independent orbital mechanics prediction between measurements
- **Measurement Interval**: Configurable (default 5 minutes)
- **Prediction Steps**: 30-second intervals using pure orbital mechanics
- **Uncertainty Estimation**: Proper covariance propagation

### Accuracy Validation
- **SGP4 Comparison**: Real-time comparison against standard orbital model
- **NASA OEM Validation**: Validation against authoritative NASA ephemeris data
- **Performance Metrics**: Sub-kilometer accuracy over extended periods

### Coordinate Systems
- **ECI (Earth-Centered Inertial)**: For orbital mechanics calculations
- **ECEF (Earth-Centered Earth-Fixed)**: For ground tracking
- **Geodetic**: For latitude/longitude/altitude display

## Operating Status

- ✅ All dependencies installed and working
- ✅ Streamlit application running on port 5000
- ✅ Replit environment properly configured
- ✅ Deployment configuration ready for production
- ✅ All core modules functional and tested
# Enhanced Orbital Determination System

## Overview
This is an advanced satellite orbital determination system imported from GitHub that uses an Enhanced Extended Kalman Filter (EKF) to achieve sub-1km accuracy for satellite tracking. The system processes Two-Line Element (TLE) data and validates results against NASA's Orbital Ephemeris Message (OEM) reference data.

## Recent Changes
- **September 20, 2025**: Successfully imported and configured for Replit environment
  - Installed Python dependencies using UV package manager
  - Configured Streamlit to run on port 5000 with proper host settings (0.0.0.0)
  - Set up deployment configuration for autoscale deployment
  - Verified application runs successfully with web interface

## User Preferences
- Communication style: Simple, everyday language
- Prefers step-by-step explanations for technical concepts

## Project Architecture

### Frontend
- **Technology**: Streamlit web application
- **Port**: 5000 (configured for Replit proxy environment)
- **Host**: 0.0.0.0 (allows all hosts for Replit iframe access)
- **Configuration**: Located in `.streamlit/config.toml`

### Core Components
- **Enhanced EKF Tracker** (`enhanced_ekf_tracker.py`): 9-dimensional state vector tracking
- **Force Models** (`force_models.py`): High-fidelity orbital mechanics (J2-J6 harmonics, atmospheric drag, SRP)
- **Validation Framework** (`validation_framework.py`): Accuracy assessment using NASA OEM data
- **Adaptive Filtering** (`adaptive_filtering.py`): Innovation-based covariance matching
- **ML Integration** (`ml_residual_corrector.py`): Neural network bias correction

### Dependencies
- **Package Manager**: UV (uv.lock, pyproject.toml)
- **Python Version**: 3.11+
- **Key Libraries**: streamlit, numpy, pandas, plotly, scikit-learn, scipy, sgp4, skyfield
- **Data Sources**: NASA OEM data, NOAA Space Weather data

### Data Files
- `data/`: Contains ISS orbital ephemeris data and TLE noise mappings
- `attached_assets/`: Additional ISS orbital data files

### Deployment
- **Target**: Autoscale (stateless web application)
- **Command**: `streamlit run app.py --server.address=0.0.0.0 --server.port=5000`
- **Environment**: Configured for Replit cloud environment

## Features
- Real-time satellite tracking with sub-1km accuracy targeting
- Interactive 3D orbital visualization
- Parameter estimation for drag and solar radiation pressure coefficients
- Comprehensive validation against NASA reference data
- Advanced filtering with adaptive algorithms
- Machine learning residual correction
- Space weather integration for atmospheric density modeling

## Usage
The application provides a web-based dashboard for:
1. Satellite selection (ISS or custom TLE)
2. EKF parameter configuration
3. Force model selection
4. Real-time tracking control
5. Validation result analysis
6. Performance visualization

## Status
✅ Fully configured and operational in Replit environment
✅ All dependencies installed and working
✅ Web interface accessible and functional
✅ Deployment configuration complete
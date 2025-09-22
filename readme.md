# Enhanced Orbital Determination System

## Overview

This is an advanced satellite orbital determination system that uses an Enhanced Extended Kalman Filter (EKF) to achieve sub-1km accuracy for satellite tracking. The system processes Two-Line Element (TLE) data and validates results against NASA's Orbital Ephemeris Message (OEM) reference data. It incorporates sophisticated force modeling, adaptive filtering, machine learning residual correction, and comprehensive validation frameworks to provide high-precision orbital state estimation for Low Earth Orbit (LEO) satellites, with particular focus on the International Space Station (ISS).

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Core Tracking Engine
The system is built around an Enhanced EKF tracker (`enhanced_ekf_tracker.py`) that maintains a 9-dimensional state vector including position, velocity, drag coefficient (CdA), solar radiation pressure coefficient (Cr), and empirical along-track acceleration. This augmented state approach allows for real-time parameter estimation alongside orbital state determination.

### Force Modeling Framework
The force models (`force_models.py`) implement high-fidelity orbital mechanics including:
- J2-J6 zonal harmonics for Earth's gravitational field
- Atmospheric drag using NRLMSISE-00 density model with real-time space weather data
- Solar radiation pressure with shadow modeling
- Third-body perturbations from Sun and Moon

The atmospheric modeling (`atmospheric_models.py`) integrates space weather data (`space_weather.py`) to provide accurate density estimates that are crucial for drag modeling in LEO.

### Adaptive and Robust Filtering
The adaptive filtering system (`adaptive_filtering.py`) implements innovation-based covariance matching and divergence detection to automatically adjust process and measurement noise parameters. This prevents filter divergence and maintains optimal performance across varying conditions.

### Parameter Estimation Strategy
A sliding-window batch estimator (`batch_estimator.py`) uses Levenberg-Marquardt optimization to robustly identify drag and SRP coefficients from measurement residuals. This addresses the weak observability problem inherent in TLE-only tracking.

### Post-Processing Enhancement
The Rauch-Tung-Striebel smoother (`rts_smoother.py`) provides fixed-lag smoothing to improve state estimates at specific timestamps, particularly for validation against OEM data.

### Machine Learning Integration
An ML residual corrector (`ml_residual_corrector.py`) uses neural networks to identify and correct systematic biases in the EKF predictions, learning from historical tracking errors to improve future performance.

### Measurement Modeling
The TLE measurement model (`tle_measurement_model.py`) provides empirical age-to-noise mapping based on historical TLE vs OEM residual analysis. This ensures proper weighting of TLE observations based on their age and expected accuracy.

### Validation Framework
A comprehensive validation system (`validation_framework.py`) provides rigorous accuracy assessment using NASA OEM data as ground truth. The validator (`nasa_oem_validator.py`) handles comparison at OEM-only timestamps to avoid interpolation artifacts.

### User Interface
The Streamlit web application (`app.py`) provides an interactive dashboard for system configuration, real-time tracking visualization, and validation result analysis.

## External Dependencies

### Space Data Sources
- **NASA OEM Data**: Official orbital ephemeris from NASA JSC for validation ground truth
- **NOAA Space Weather**: Real-time F10.7 solar flux and Kp geomagnetic indices from SWPC
- **Celestrak APIs**: Alternative space weather data sources

### Scientific Computing Stack
- **NumPy/SciPy**: Core numerical computing and optimization algorithms
- **Pandas**: Data manipulation and time series handling
- **Scikit-learn**: Machine learning models for residual correction

### Visualization and UI
- **Streamlit**: Web-based interactive dashboard
- **Plotly**: Interactive plotting for orbital visualization and analysis
- **Plotly Express**: Simplified plotting interface

### Data Persistence
- **JSON**: Configuration and empirical mapping storage
- **Joblib**: ML model serialization and persistence

### Coordinate Systems and Time
- **Datetime**: Standard Python time handling
- **Custom coordinate transforms**: ECI/ECEF/geodetic transformations with proper Earth rotation modeling

The system architecture prioritizes modularity, allowing individual components to be tested and validated independently while maintaining clean interfaces between subsystems. The design emphasizes robustness through multiple layers of validation, adaptive algorithms, and fallback mechanisms to ensure reliable sub-kilometer accuracy for operational satellite tracking.

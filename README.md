# Overview

True EKF Satellite Tracker is a Python-based satellite tracking system that implements an Extended Kalman Filter (EKF) for real-time satellite trajectory prediction. The system compares EKF predictions against SGP4 model calculations and NASA OEM (Orbital Ephemeris Message) data to provide accuracy validation. It features an interactive Streamlit dashboard for visualizing satellite trajectories, orbital mechanics calculations with J2 perturbations, and coordinate transformations between different reference frames (ECI, ECEF, Geodetic).

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
The system uses Streamlit as the web framework to create an interactive dashboard. The main application (`app.py`) serves as the entry point, providing real-time visualization of satellite tracking data through Plotly charts and graphs. The interface allows users to start/stop tracking, view accuracy comparisons between different prediction models, and monitor tracking statistics.

## Core Tracking System
The architecture centers around the `TrueEKFTracker` class, which implements a proper Extended Kalman Filter for satellite state estimation. Unlike simple SGP4 wrappers, this system performs independent orbital mechanics calculations including:

- State prediction using numerical integration of orbital equations
- J2 perturbation modeling for Earth's oblateness effects
- Atmospheric drag calculations for low Earth orbit satellites
- Covariance propagation for uncertainty estimation

## Coordinate Systems
The `CoordinateTransforms` module handles transformations between multiple coordinate systems:
- Earth-Centered Inertial (ECI) for orbital calculations
- Earth-Centered Earth-Fixed (ECEF) for ground tracking
- Geodetic coordinates for latitude/longitude positions
- Julian date calculations for astronomical time systems

## Orbital Mechanics Engine
The `OrbitalMechanics` class implements pure orbital physics calculations including:
- Two-body gravitational acceleration
- J2 zonal harmonic perturbations
- Atmospheric drag modeling
- Solar radiation pressure effects
- Numerical integration using Runge-Kutta methods

## Data Management
The `SatelliteDataManager` handles TLE (Two-Line Element) data from multiple sources including Celestrak feeds. It manages satellite selection, TLE parsing, and maintains satellite metadata. The system stores selected satellite information in JSON format for persistence.

## Validation System
The `NASAOEMValidator` provides ground truth comparison by fetching official NASA ephemeris data through the JPL Horizons API. This allows for accuracy assessment of both EKF and SGP4 predictions against authoritative orbital data.

## Threading Architecture
The application uses Python threading for concurrent tracking operations, with a queue-based system for thread-safe data exchange between the tracking engine and the Streamlit interface. This prevents UI blocking during intensive orbital calculations.

# External Dependencies

## Core Scientific Libraries
- **NumPy**: Mathematical operations and array processing for orbital calculations
- **SciPy**: Numerical integration routines for orbital propagation
- **Skyfield**: Astronomical calculations and TLE parsing
- **Pandas**: Data structure management for tracking history

## Web Framework and Visualization
- **Streamlit**: Interactive web dashboard framework
- **Plotly**: Interactive plotting library for trajectory visualization
- **Plotly Express**: Simplified plotting interface for statistical charts

## External APIs and Data Sources
- **NASA JPL Horizons API**: Authoritative orbital ephemeris data for validation
- **Celestrak TLE Feeds**: Real-time Two-Line Element data sources
  - Space stations feed
  - Visual satellites feed  
  - Active satellites feed

## Data Formats and Standards
- **JSON**: Configuration and satellite metadata storage
- **TLE Format**: Standard orbital element format from NORAD/Space Force
- **OEM Format**: NASA Orbital Ephemeris Message standard for validation

The system is designed to operate independently with fallback mechanisms when external APIs are unavailable, using synthetic validation data based on SGP4 calculations when NASA OEM data cannot be retrieved.

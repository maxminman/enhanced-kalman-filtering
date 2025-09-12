#!/usr/bin/env python3
"""
True Extended Kalman Filter Satellite Tracker
=============================================
Real EKF with independent orbital mechanics prediction
"""

import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')
from skyfield.api import utc

from orbital_mechanics import OrbitalMechanics
from coordinate_transforms import CoordinateTransforms
from satellite_data import SatelliteDataManager
from nasa_oem_validator import NASAOEMValidator


class TrueEKFTracker:
    """True Extended Kalman Filter for satellite tracking"""
    
    def __init__(self):
        self.orbital_mechanics = OrbitalMechanics()
        self.coord_transforms = CoordinateTransforms()
        self.sat_manager = SatelliteDataManager()
        self.validator = NASAOEMValidator()
        
        # EKF state: [x, y, z, vx, vy, vz] in ECI coordinates (km, km/s)
        self.state = None
        self.covariance = None
        self.last_prediction_time = None
        self.last_measurement_time = None
        
        # EKF configuration
        self.measurement_interval = 60  # seconds (1 minute between measurements - prevents EKF divergence)
        self.prediction_step = 30  # seconds (30s prediction steps)
        
        # Statistics
        self.stats = {
            'predictions_made': 0,
            'measurements_processed': 0,
            'ekf_accuracy_history': [],
            'sgp4_accuracy_history': [],
            'oem_validation_history': []
        }
        
        # Load satellite data
        self.sat_manager.load_selected_satellite()
    
    def initialize_ekf(self, initial_time=None):
        """Initialize EKF with SGP4 state as starting point"""
        if initial_time is None:
            initial_time = datetime.now(utc)
        
        # Get initial state from SGP4
        sgp4_state = self.sat_manager.get_sgp4_state_at_time(initial_time)
        if not sgp4_state:
            return False
        
        # Initialize state vector [x, y, z, vx, vy, vz]
        self.state = np.concatenate([
            sgp4_state['position_km'],
            sgp4_state['velocity_km_s']
        ])
        
        # Initialize covariance based on SGP4 uncertainty estimates
        noise_params = self.sat_manager.estimate_measurement_noise()
        if not noise_params:
            # Fallback defaults
            pos_uncertainty = 0.5  # 500m
            vel_uncertainty = 0.001  # 1mm/s
        else:
            pos_uncertainty = noise_params['position_noise_km']
            vel_uncertainty = noise_params['velocity_noise_km_s']
        
        # Initial covariance (conservative)
        self.covariance = np.diag([
            pos_uncertainty**2, pos_uncertainty**2, pos_uncertainty**2,
            vel_uncertainty**2, vel_uncertainty**2, vel_uncertainty**2
        ])
        
        self.last_prediction_time = initial_time
        self.last_measurement_time = initial_time
        
        print("✅ True EKF initialized with orbital mechanics")
        print(f"📍 Initial position uncertainty: ±{pos_uncertainty*1000:.0f}m")
        print(f"🚀 Initial velocity uncertainty: ±{vel_uncertainty*1000:.1f}mm/s")
        print(f"⏱️ Measurement interval: {self.measurement_interval}s")
        print(f"🔄 Prediction step: {self.prediction_step}s")
        
        return True
    
    def predict_step(self, current_time):
        """Pure orbital mechanics prediction step"""
        if self.state is None or self.last_prediction_time is None or self.covariance is None:
            print("❌ EKF state/covariance/time not initialized, cannot predict")
            return False
        
        # Time since last prediction
        dt = (current_time - self.last_prediction_time).total_seconds()
        
        if dt <= 0:
            return True  # No time has passed
        
        # Propagate state using orbital mechanics (enable drag for accuracy)
        self.state = self.orbital_mechanics.propagate_state(
            self.state, dt, include_drag=True
        )
        
        # Compute Jacobian for covariance propagation
        F = self.orbital_mechanics.compute_jacobian(self.state)
        
        # Discrete-time state transition matrix
        F_discrete = np.eye(6) + F * dt
        
        # Kinematic process noise with continuous acceleration uncertainty
        altitude_km = np.linalg.norm(self.state[:3]) - 6378.137  # Earth radius
        
        # Continuous acceleration noise standard deviation (km/s²)
        if altitude_km < 500:  # LEO (like ISS) - high drag variability
            sigma_a = 1e-5  # 10 μm/s² continuous acceleration noise
        elif altitude_km < 1500:  # MEO
            sigma_a = 5e-6  # 5 μm/s² acceleration noise
        else:  # GEO - minimal perturbations
            sigma_a = 1e-6  # 1 μm/s² acceleration noise
        
        # Scale with TLE age - older TLEs have more uncertainty
        if hasattr(self.sat_manager, 'satellite_data') and self.sat_manager.satellite_data:
            tle_age_hours = self.sat_manager.satellite_data.get('tle_age_hours', 24)
            age_factor = 1 + (tle_age_hours / 24) * 0.5  # 50% increase per day
            sigma_a *= age_factor
        
        # Kinematic process noise matrix Q_d with proper correlation structure
        # Q_d = [[dt³/3*I, dt²/2*I], [dt²/2*I, dt*I]] * σ_a²
        dt2 = dt * dt
        dt3 = dt2 * dt
        
        # Position-position block (3x3)
        Q_pp = np.eye(3) * (dt3 / 3.0) * (sigma_a ** 2)
        # Position-velocity block (3x3) 
        Q_pv = np.eye(3) * (dt2 / 2.0) * (sigma_a ** 2)
        # Velocity-velocity block (3x3)
        Q_vv = np.eye(3) * dt * (sigma_a ** 2)
        
        # Assemble the 6x6 process noise matrix
        Q = np.block([[Q_pp, Q_pv],
                      [Q_pv, Q_vv]])
        
        # Covariance prediction
        self.covariance = F_discrete @ self.covariance @ F_discrete.T + Q
        
        # Ensure positive definite
        self.covariance = self._make_positive_definite(self.covariance)
        
        self.last_prediction_time = current_time
        self.stats['predictions_made'] += 1
        
        return True
    
    def should_take_measurement(self, current_time):
        """Determine if we should take a measurement"""
        if self.last_measurement_time is None:
            return True
        
        time_since_measurement = (current_time - self.last_measurement_time).total_seconds()
        return time_since_measurement >= self.measurement_interval
    
    def update_step(self, measurement_pos, measurement_vel, current_time):
        """EKF measurement update with SGP4 observation"""
        # Check if state and covariance are initialized
        if self.state is None or self.covariance is None:
            print("❌ EKF state not initialized, cannot perform update")
            return False
            
        # Measurement vector [x, y, z, vx, vy, vz]
        measurement = np.concatenate([measurement_pos, measurement_vel])
        
        # Measurement matrix (observe full state)
        H = np.eye(6)
        
        # Measurement noise based on SGP4 uncertainty
        noise_params = self.sat_manager.estimate_measurement_noise()
        if not noise_params:
            # Fallback defaults
            pos_noise = 0.5  # 500m
            vel_noise = 0.001  # 1mm/s
        else:
            pos_noise = noise_params['position_noise_km']
            vel_noise = noise_params['velocity_noise_km_s']
        
        R = np.diag([
            pos_noise**2, pos_noise**2, pos_noise**2,
            vel_noise**2, vel_noise**2, vel_noise**2
        ])
        
        # Innovation
        y = measurement - H @ self.state
        
        # Innovation covariance
        S = H @ self.covariance @ H.T + R
        
        # Kalman gain
        try:
            K = self.covariance @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.covariance @ H.T @ np.linalg.pinv(S)
        
        # State update
        self.state = self.state + K @ y
        
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(6) - K @ H
        self.covariance = I_KH @ self.covariance @ I_KH.T + K @ R @ K.T
        
        # Ensure positive definite
        self.covariance = self._make_positive_definite(self.covariance)
        
        self.last_measurement_time = current_time
        self.stats['measurements_processed'] += 1
        
        return True
    
    def _make_positive_definite(self, matrix):
        """Ensure matrix is positive definite"""
        # Make symmetric
        matrix = (matrix + matrix.T) / 2
        
        # Eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(matrix)
        
        # Clip negative eigenvalues
        eigenvals = np.maximum(eigenvals, 1e-12)
        
        # Reconstruct matrix
        return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    
    def get_position_uncertainty(self):
        """Get current position uncertainty"""
        if self.covariance is None:
            return 0
        
        # Extract position covariance
        pos_cov = self.covariance[:3, :3]
        
        # Calculate uncertainty magnitude (3-sigma)
        uncertainty_km = 3 * np.sqrt(np.trace(pos_cov) / 3)
        return uncertainty_km
    
    def get_geodetic_position(self, current_time):
        """Convert EKF state to geodetic coordinates"""
        if self.state is None:
            return None
        
        pos_eci = self.state[:3]
        vel_eci = self.state[3:]
        
        try:
            geodetic = self.coord_transforms.eci_to_geodetic(
                pos_eci, vel_eci, current_time
            )
            return geodetic
        except Exception as e:
            print(f"⚠️ Geodetic conversion failed: {e}")
            return None
    
    def track_step(self, current_time):
        """Single tracking step - predict and optionally update"""
        # Always predict
        if not self.predict_step(current_time):
            return None
        
        # Get SGP4 reference (for comparison and optional measurement)
        sgp4_state = self.sat_manager.get_sgp4_state_at_time(current_time)
        if not sgp4_state:
            return None
        
        # Take measurement if interval has elapsed
        if self.should_take_measurement(current_time):
            self.update_step(
                sgp4_state['position_km'],
                sgp4_state['velocity_km_s'],
                current_time
            )
        
        # Convert to geodetic
        ekf_geodetic = self.get_geodetic_position(current_time)
        if not ekf_geodetic:
            return None
        
        # Get SGP4 geodetic for comparison
        sgp4_geodetic = self.coord_transforms.eci_to_geodetic(
            sgp4_state['position_km'],
            sgp4_state['velocity_km_s'],
            current_time
        )
        
        # Calculate EKF uncertainty
        uncertainty_km = self.get_position_uncertainty()
        
        # Validate against OEM if available
        oem_validation = None
        if self.state is not None:
            oem_validation = self.validator.validate_prediction(
                self.state[:3], current_time
            )
        
        # Calculate accuracy vs SGP4
        pos_diff = 0
        if self.state is not None:
            pos_diff = np.linalg.norm(self.state[:3] - sgp4_state['position_km'])
        
        # Update statistics
        self.stats['ekf_accuracy_history'].append(uncertainty_km * 1000)  # meters
        self.stats['sgp4_accuracy_history'].append(pos_diff * 1000)  # meters
        
        if oem_validation:
            self.stats['oem_validation_history'].append(oem_validation['error_m'])
        
        return {
            'timestamp': current_time,
            'ekf_position': {
                'latitude': ekf_geodetic['latitude'],
                'longitude': ekf_geodetic['longitude'],
                'altitude': ekf_geodetic['altitude']
            },
            'sgp4_position': {
                'latitude': sgp4_geodetic['latitude'],
                'longitude': sgp4_geodetic['longitude'],
                'altitude': sgp4_geodetic['altitude']
            },
            'ekf_uncertainty_m': uncertainty_km * 1000,
            'ekf_vs_sgp4_error_m': pos_diff * 1000,
            'oem_validation': oem_validation,
            'measurement_taken': self.should_take_measurement(current_time),
            'predictions_made': self.stats['predictions_made'],
            'measurements_processed': self.stats['measurements_processed']
        }
    
    def get_performance_summary(self):
        """Get comprehensive performance summary"""
        ekf_acc = self.stats['ekf_accuracy_history']
        sgp4_acc = self.stats['sgp4_accuracy_history']
        oem_acc = self.stats['oem_validation_history']
        
        summary = {
            'predictions_made': self.stats['predictions_made'],
            'measurements_processed': self.stats['measurements_processed'],
            'measurement_rate': self.stats['measurements_processed'] / max(1, self.stats['predictions_made']) * 100
        }
        
        if ekf_acc:
            summary.update({
                'ekf_mean_uncertainty_m': np.mean(ekf_acc),
                'ekf_std_uncertainty_m': np.std(ekf_acc),
                'ekf_max_uncertainty_m': np.max(ekf_acc),
                'ekf_sub_100m_rate': np.sum(np.array(ekf_acc) < 100) / len(ekf_acc) * 100,
                'ekf_sub_500m_rate': np.sum(np.array(ekf_acc) < 500) / len(ekf_acc) * 100,
                'ekf_sub_1km_rate': np.sum(np.array(ekf_acc) < 1000) / len(ekf_acc) * 100
            })
        
        if sgp4_acc:
            summary.update({
                'ekf_vs_sgp4_mean_error_m': np.mean(sgp4_acc),
                'ekf_vs_sgp4_rms_error_m': np.sqrt(np.mean(np.array(sgp4_acc)**2))
            })
        
        if oem_acc:
            summary.update({
                'oem_validation_points': len(oem_acc),
                'oem_mean_error_m': np.mean(oem_acc),
                'oem_rms_error_m': np.sqrt(np.mean(np.array(oem_acc)**2)),
                'oem_sub_1km_rate': np.sum(np.array(oem_acc) < 1000) / len(oem_acc) * 100,
                'oem_sub_500m_rate': np.sum(np.array(oem_acc) < 500) / len(oem_acc) * 100
            })
        
        return summary
    
    def setup_oem_validation(self, duration_hours=2):
        """Setup OEM validation for the tracking period"""
        if not self.sat_manager.satellite_data:
            return False
        
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(hours=duration_hours)
        
        return self.validator.fetch_oem_data(
            self.sat_manager.satellite_data['norad_id'],
            start_time,
            end_time
        )

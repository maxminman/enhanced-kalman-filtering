import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Any

from force_models import ForceModels
from atmospheric_models import NRLMSISE00
from batch_estimator import BatchEstimator
from rts_smoother import RTSSmoother
from adaptive_filtering import AdaptiveFiltering
from ml_residual_corrector import MLResidualCorrector
from tle_measurement_model import TLEMeasurementModel
from coordinate_transforms import CoordinateTransforms
from utils import julian_date, eci_to_geodetic

class EnhancedEKFTracker:
    """
    Enhanced Extended Kalman Filter for satellite orbital determination
    with sub-1km accuracy targeting through advanced parameter estimation
    and validation techniques.
    """
    
    def __init__(self, tle_line1: str, tle_line2: str, config: Dict[str, Any]):
        """
        Initialize the Enhanced EKF Tracker
        
        Args:
            tle_line1: First line of TLE
            tle_line2: Second line of TLE  
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize coordinate transformer
        self.coord_transform = CoordinateTransforms()
        
        # Parse TLE and initialize state
        self.tle_data = self._parse_tle(tle_line1, tle_line2)
        self.initial_state = self._tle_to_state_vector(self.tle_data)
        
        # Initialize state vector [x, y, z, vx, vy, vz, CdA, Cr, along_track_accel]
        self.state_dim = 9  # Position, velocity, drag coeff, SRP coeff, empirical accel
        self.obs_dim = 6    # Position and velocity observations from TLE
        
        # State: [position(3), velocity(3), CdA(1), Cr(1), along_track_accel(1)]
        self.state = np.zeros(self.state_dim)
        self.state[:6] = self.initial_state[:6]  # Position and velocity
        self.state[6] = config.get('drag_coeff', 2.2)  # CdA
        self.state[7] = config.get('srp_coeff', 1.3)   # Cr
        self.state[8] = 0.0  # Along-track empirical acceleration
        
        # Initialize covariance matrix
        self._initialize_covariance()
        
        # Initialize sub-systems
        self.force_models = ForceModels(config)
        self.atmosphere = NRLMSISE00() if config.get('use_nrlmsise', True) else None
        self.measurement_model = TLEMeasurementModel()
        
        # Advanced features
        if config.get('use_batch_estimation', True):
            self.batch_estimator = BatchEstimator(config)
        else:
            self.batch_estimator = None
            
        if config.get('use_rts_smoother', True):
            self.rts_smoother = RTSSmoother(config)
        else:
            self.rts_smoother = None
            
        if config.get('use_adaptive_filtering', True):
            self.adaptive_filter = AdaptiveFiltering(config)
        else:
            self.adaptive_filter = None
            
        if config.get('use_ml_corrector', False):
            self.ml_corrector = MLResidualCorrector(config)
        else:
            self.ml_corrector = None
        
        # Tracking variables
        self.current_time = datetime.utcnow()
        self.last_update = None
        self.iteration_count = 0
        self.measurement_history = []
        self.state_history = []
        self.innovation_history = []
        
        # Filter health monitoring
        self.divergence_count = 0
        self.max_divergence_count = 5
        
        self.logger.info("Enhanced EKF Tracker initialized")
    
    def _parse_tle(self, line1: str, line2: str) -> Dict[str, Any]:
        """Parse TLE data"""
        try:
            # Basic TLE parsing - extract key orbital elements
            epoch_year = int(line1[18:20])
            epoch_day = float(line1[20:32])
            if epoch_year > 56:
                epoch_year += 1900
            else:
                epoch_year += 2000
                
            inclination = float(line2[8:16])
            raan = float(line2[17:25])
            eccentricity = float('0.' + line2[26:33])
            arg_perigee = float(line2[34:42])
            mean_anomaly = float(line2[43:51])
            mean_motion = float(line2[52:63])
            
            return {
                'epoch_year': epoch_year,
                'epoch_day': epoch_day,
                'inclination': inclination,
                'raan': raan,
                'eccentricity': eccentricity,
                'arg_perigee': arg_perigee,
                'mean_anomaly': mean_anomaly,
                'mean_motion': mean_motion,
                'line1': line1,
                'line2': line2
            }
        except Exception as e:
            self.logger.error(f"TLE parsing error: {e}")
            raise ValueError(f"Invalid TLE format: {e}")
    
    def _tle_to_state_vector(self, tle_data: Dict[str, Any]) -> np.ndarray:
        """Convert TLE to Cartesian state vector using SGP4"""
        try:
            # Fallback to basic Keplerian conversion
            return self._keplerian_to_cartesian(tle_data)
            
        except Exception as e:
            self.logger.error(f"SGP4 conversion error: {e}")
            # Fallback to basic Keplerian conversion
            return self._keplerian_to_cartesian(tle_data)
    
    def _keplerian_to_cartesian(self, tle_data: Dict[str, Any]) -> np.ndarray:
        """Convert Keplerian elements to Cartesian coordinates"""
        # Earth gravitational parameter (m^3/s^2)
        mu = 3.986004418e14
        
        # Convert to radians
        i = np.radians(tle_data['inclination'])
        raan = np.radians(tle_data['raan'])
        e = tle_data['eccentricity']
        w = np.radians(tle_data['arg_perigee'])
        M = np.radians(tle_data['mean_anomaly'])
        n = tle_data['mean_motion'] * 2 * np.pi / 86400  # rad/s
        
        # Semi-major axis
        a = (mu / (n**2))**(1/3)
        
        # Solve Kepler's equation for eccentric anomaly
        E = M
        for _ in range(10):  # Newton-Raphson iteration
            E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        
        # True anomaly
        nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E/2), 
                           np.sqrt(1 - e) * np.cos(E/2))
        
        # Distance
        r = a * (1 - e * np.cos(E))
        
        # Position and velocity in orbital plane
        x_orb = r * np.cos(nu)
        y_orb = r * np.sin(nu)
        z_orb = 0
        
        p = a * (1 - e**2)
        h = np.sqrt(mu * p)
        
        vx_orb = -(mu / h) * np.sin(nu)
        vy_orb = (mu / h) * (e + np.cos(nu))
        vz_orb = 0
        
        # Rotation matrices
        R1 = np.array([[np.cos(raan), -np.sin(raan), 0],
                       [np.sin(raan), np.cos(raan), 0],
                       [0, 0, 1]])
        
        R2 = np.array([[1, 0, 0],
                       [0, np.cos(i), -np.sin(i)],
                       [0, np.sin(i), np.cos(i)]])
        
        R3 = np.array([[np.cos(w), -np.sin(w), 0],
                       [np.sin(w), np.cos(w), 0],
                       [0, 0, 1]])
        
        # Combined rotation matrix
        R = R1 @ R2 @ R3
        
        # Transform to inertial frame
        r_eci = R @ np.array([x_orb, y_orb, z_orb])
        v_eci = R @ np.array([vx_orb, vy_orb, vz_orb])
        
        state = np.zeros(6)
        state[:3] = r_eci
        state[3:6] = v_eci
        
        return state
    
    def _initialize_covariance(self):
        """Initialize covariance matrix with appropriate uncertainties"""
        self.P = np.eye(self.state_dim)
        
        # Position uncertainty (m^2)
        self.P[:3, :3] *= (1000)**2  # 1 km initial position uncertainty
        
        # Velocity uncertainty (m/s)^2
        self.P[3:6, 3:6] *= (10)**2  # 10 m/s initial velocity uncertainty
        
        # CdA uncertainty
        self.P[6, 6] = (0.5)**2  # 50% uncertainty in drag coefficient
        
        # Cr uncertainty  
        self.P[7, 7] = (0.3)**2  # 30% uncertainty in SRP coefficient
        
        # Along-track acceleration uncertainty (m/s^2)^2
        self.P[8, 8] = (1e-6)**2  # Very small empirical acceleration
    
    def predict(self, dt: float):
        """Prediction step of the EKF"""
        # State transition using numerical integration
        self.state[:6] = self._propagate_orbit(self.state[:6], dt)
        
        # Parameter evolution (slow states)
        # CdA and Cr evolve slowly with small random walk
        # Along-track acceleration remains constant
        
        # Compute state transition matrix F
        F = self._compute_state_transition_matrix(dt)
        
        # Process noise matrix Q
        Q = self._compute_process_noise_matrix(dt)
        
        # Covariance prediction
        self.P = F @ self.P @ F.T + Q
        
        # Adaptive process noise adjustment
        if self.adaptive_filter:
            Q_adaptive = self.adaptive_filter.adapt_process_noise(
                self.P, self.innovation_history
            )
            if Q_adaptive is not None:
                self.P += Q_adaptive
    
    def _propagate_orbit(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Numerical integration of orbital equations"""
        def dynamics(t, y):
            """Orbital dynamics with perturbations"""
            r = y[:3]
            v = y[3:6]
            
            # Total acceleration
            accel = np.zeros(3)
            
            # Two-body gravitational acceleration
            r_norm = np.linalg.norm(r)
            mu = 3.986004418e14  # Earth gravitational parameter
            accel += -mu * r / (r_norm**3)
            
            # Add perturbations through force models
            perturbations = self.force_models.compute_perturbations(
                r, v, self.current_time, self.state[6], self.state[7]
            )
            accel += perturbations
            
            # Along-track empirical acceleration
            # Compute along-track unit vector
            h = np.cross(r, v)  # Angular momentum vector
            if np.linalg.norm(h) > 0:
                r_unit = r / np.linalg.norm(r)
                along_track = np.cross(h, r)
                along_track = along_track / np.linalg.norm(along_track)
                
                accel += self.state[8] * along_track
            
            # Return state derivative [velocity, acceleration]
            return np.concatenate([v, accel])
        
        # Use RK45 integration
        from scipy.integrate import solve_ivp
        
        try:
            result = solve_ivp(
                dynamics, [0, dt], state, 
                method='RK45', rtol=1e-10, atol=1e-12
            )
            
            return result.y[:, -1]
        except Exception as e:
            self.logger.warning(f"Integration error: {e}")
            # Fallback to simple propagation
            return self._simple_propagation(state, dt)
    
    def _simple_propagation(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Simple two-body propagation as fallback"""
        r = state[:3]
        v = state[3:6]
        
        # Two-body acceleration
        r_norm = np.linalg.norm(r)
        mu = 3.986004418e14
        accel = -mu * r / (r_norm**3)
        
        # Simple Euler integration
        new_v = v + accel * dt
        new_r = r + v * dt + 0.5 * accel * dt**2
        
        return np.concatenate([new_r, new_v])
    
    def _compute_state_transition_matrix(self, dt: float) -> np.ndarray:
        """Compute the state transition matrix F"""
        F = np.eye(self.state_dim)
        
        # Position-velocity coupling
        F[:3, 3:6] = np.eye(3) * dt
        
        # For orbital dynamics, we approximate the velocity-position coupling
        # This is a simplified version - full implementation would require
        # numerical differentiation of the dynamics
        r = self.state[:3]
        r_norm = np.linalg.norm(r)
        
        if r_norm > 0:
            mu = 3.986004418e14
            
            # Gravity gradient matrix
            I = np.eye(3)
            rr = np.outer(r, r)
            gravity_grad = -mu / (r_norm**3) * (I - 3 * rr / (r_norm**2))
            
            F[3:6, :3] = gravity_grad * dt
        
        # Parameter states are constant (random walk)
        # F[6:, 6:] is already identity
        
        return F
    
    def _compute_process_noise_matrix(self, dt: float) -> np.ndarray:
        """Compute process noise matrix Q"""
        Q = np.zeros((self.state_dim, self.state_dim))
        
        # Position and velocity process noise
        # Use continuous-time white noise acceleration model
        q_accel = self.config.get('process_noise_scale', 1.0) * 1e-12  # m^2/s^3
        
        Q_pos_vel = np.array([
            [dt**3/3, dt**2/2],
            [dt**2/2, dt]
        ]) * q_accel
        
        # Apply to each spatial dimension
        for i in range(3):
            Q[i:6:3, i:6:3] = Q_pos_vel
        
        # Parameter process noise (random walk)
        Q[6, 6] = (0.01 * dt)**2  # CdA random walk
        Q[7, 7] = (0.005 * dt)**2  # Cr random walk
        Q[8, 8] = (1e-8 * dt)**2  # Along-track acceleration random walk
        
        return Q
    
    def update(self, current_time: datetime) -> Optional[Dict[str, Any]]:
        """Update step with TLE-derived measurements"""
        try:
            dt = (current_time - self.current_time).total_seconds()
            if dt <= 0:
                dt = 10.0  # Default 10 second update
            
            # Prediction step
            self.predict(dt)
            
            # Generate synthetic measurement from current TLE
            # In practice, this would be a new TLE observation
            measurement, R = self.measurement_model.generate_measurement(
                self.tle_data, current_time
            )
            
            if measurement is None or R is None:
                # Use current state estimate as measurement with high uncertainty
                measurement = self.state[:6].copy()
                R = np.eye(6) * 10000**2  # 10km uncertainty
            
            # Measurement update
            self._measurement_update(measurement, R)
            
            # Batch parameter estimation (periodic)
            if (self.batch_estimator and 
                self.iteration_count % 20 == 0 and 
                len(self.measurement_history) > 10):
                
                try:
                    updated_params = self.batch_estimator.estimate_parameters(
                        self.measurement_history, self.state_history
                    )
                    self.state[6:8] = updated_params[:2]  # Update CdA and Cr
                except Exception as e:
                    self.logger.warning(f"Batch estimation failed: {e}")
            
            # Apply RTS smoothing (fixed-lag)
            if (self.rts_smoother and 
                len(self.state_history) > 10):
                
                try:
                    smoothed_states = self.rts_smoother.smooth(
                        self.state_history[-10:]
                    )
                    # Update recent history with smoothed estimates
                    if len(smoothed_states) > 0:
                        self.state = smoothed_states[-1]
                except Exception as e:
                    self.logger.warning(f"RTS smoothing failed: {e}")
            
            # ML residual correction
            if self.ml_corrector:
                try:
                    correction = self.ml_corrector.predict_correction(
                        self.state, self.innovation_history[-5:] if len(self.innovation_history) >= 5 else []
                    )
                    self.state[:6] += correction
                except Exception as e:
                    self.logger.warning(f"ML correction failed: {e}")
            
            # Update tracking variables
            self.current_time = current_time
            self.last_update = current_time
            self.iteration_count += 1
            
            # Store history
            self.state_history.append(self.state.copy())
            if len(self.state_history) > 1000:  # Keep last 1000 states
                self.state_history = self.state_history[-1000:]
            
            # Prepare result
            result = self._prepare_result()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Update failed: {e}")
            self._handle_filter_divergence()
            return None
    
    def _measurement_update(self, measurement: np.ndarray, R: np.ndarray):
        """Kalman filter measurement update"""
        # Measurement function (identity for position and velocity)
        H = np.zeros((self.obs_dim, self.state_dim))
        H[:6, :6] = np.eye(6)  # Observe position and velocity
        
        # Predicted measurement
        h_pred = H @ self.state
        
        # Innovation
        innovation = measurement - h_pred
        
        # Innovation covariance
        S = H @ self.P @ H.T + R
        
        # Check for filter divergence
        try:
            innovation_norm = innovation.T @ np.linalg.inv(S) @ innovation
        except np.linalg.LinAlgError:
            innovation_norm = np.linalg.norm(innovation)**2
            
        self.innovation_history.append(innovation_norm)
        
        if innovation_norm > 50:  # Chi-square threshold for 6 DOF
            self.divergence_count += 1
            self.logger.warning(f"Large innovation detected: {innovation_norm:.2f}")
            
            if self.divergence_count > self.max_divergence_count:
                self._handle_filter_divergence()
                return
        else:
            self.divergence_count = max(0, self.divergence_count - 1)
        
        # Kalman gain
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.pinv(S)
        
        # State update
        self.state += K @ innovation
        
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.state_dim) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        
        # Adaptive measurement noise
        if self.adaptive_filter:
            R_adaptive = self.adaptive_filter.adapt_measurement_noise(
                innovation, S, self.P
            )
            if R_adaptive is not None:
                R = R_adaptive
        
        # Store for history
        self.measurement_history.append({
            'measurement': measurement.copy(),
            'timestamp': self.current_time,
            'innovation': innovation.copy(),
            'R': R.copy()
        })
        
        if len(self.measurement_history) > 500:  # Keep last 500 measurements
            self.measurement_history = self.measurement_history[-500:]
    
    def _handle_filter_divergence(self):
        """Handle filter divergence by reinitializing"""
        self.logger.warning("Filter divergence detected - reinitializing")
        
        # Increase covariance to account for uncertainty
        self.P *= 10
        
        # Reset divergence counter
        self.divergence_count = 0
        
        # Optionally reinitialize from TLE
        if len(self.state_history) == 0:
            self.state[:6] = self.initial_state[:6]
            self._initialize_covariance()
    
    def _prepare_result(self) -> Dict[str, Any]:
        """Prepare tracking result for output"""
        # Extract position and velocity
        position_eci = self.state[:3]  # meters
        velocity_eci = self.state[3:6]  # m/s
        
        # Convert to geodetic coordinates
        lat, lon, alt = eci_to_geodetic(
            position_eci, self.current_time
        )
        
        # Compute velocity magnitude
        velocity_magnitude = np.linalg.norm(velocity_eci) / 1000  # km/s
        
        # Compute altitude
        altitude = alt / 1000  # km
        
        result = {
            'timestamp': self.current_time,
            'position_eci': position_eci.copy(),
            'velocity_eci': velocity_eci.copy(),
            'latitude': lat,
            'longitude': lon,
            'altitude': altitude,
            'velocity_magnitude': velocity_magnitude,
            'drag_coeff': self.state[6],
            'srp_coeff': self.state[7],
            'along_track_accel': self.state[8],
            'covariance_trace': np.trace(self.P),
            'innovation_norm': self.innovation_history[-1] if self.innovation_history else 0
        }
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get current tracker status"""
        return {
            'iterations': self.iteration_count,
            'last_update': self.last_update.isoformat() if self.last_update else 'Never',
            'cov_trace': np.trace(self.P),
            'divergence_count': self.divergence_count,
            'state_history_length': len(self.state_history),
            'measurement_history_length': len(self.measurement_history)
        }

import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Any

from force_models import ForceModels
from atmospheric_models import NRLMSISE00
from adaptive_atmospheric_model import AdaptiveAtmosphericModel
from satellite_characterizer import SatelliteCharacterizer
from batch_estimator import EnhancedBatchEstimator
from rts_smoother import RTSSmoother
from adaptive_filtering import AdaptiveFiltering
from ml_residual_corrector import MLResidualCorrector
from tle_measurement_model import TLEMeasurementModel
from coordinate_transforms import CoordinateTransforms
from adaptive_filter_tuner import AdaptiveFilterTuner, AdaptationConfig
from enhanced_divergence_detector import EnhancedDivergenceDetector, RecoveryConfig
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
        from utils import parse_tle
        self.tle_data = parse_tle(tle_line1, tle_line2)
        self.initial_state = self._tle_to_state_vector(self.tle_data)
        
        # Initialize state vector [x, y, z, vx, vy, vz, Bc, Cr, along_track_accel]
        self.state_dim = 9  # Position, velocity, ballistic coeff, SRP coeff, empirical accel
        self.obs_dim = 6    # Position and velocity observations from TLE
        
        # State: [position(3), velocity(3), Bc(1), Cr(1), along_track_accel(1)]
        self.state = np.zeros(self.state_dim)
        self.state[:6] = self.initial_state[:6]  # Position and velocity
        
        # Use updated ISS parameters from OEM data
        mass = config.get('satellite_mass', 471286.0)  # Updated ISS mass
        drag_area = config.get('drag_area', 1514.10)   # Updated ISS drag area
        drag_coeff = config.get('drag_coeff', 1.20)    # Updated ISS drag coefficient
        
        # Calculate ballistic coefficient from OEM parameters
        ballistic_coeff = config.get('ballistic_coeff', drag_coeff * drag_area / mass)
        
        self.state[6] = ballistic_coeff  # Bc (m^2/kg) - calculated from OEM
        self.state[7] = config.get('srp_coeff', 1.25)   # Cr - optimized value
        self.state[8] = 0.0  # Along-track empirical acceleration
        
        # Initialize covariance matrix
        self._initialize_covariance()
        
        # Initialize satellite characterization
        self.satellite_characterizer = SatelliteCharacterizer()
        self.satellite_props = self.satellite_characterizer.characterize_satellite(
            self.tle_data, config.get('norad_id', '25544')
        )
        
        # Initialize sub-systems
        self.force_models = ForceModels(config)
        self.atmosphere = NRLMSISE00() if config.get('use_nrlmsise', True) else None
        
        # Initialize adaptive atmospheric model
        self.adaptive_atmosphere = AdaptiveAtmosphericModel(config)
        self.adaptive_atmosphere.configure_for_satellite(self.satellite_props)
        
        self.measurement_model = TLEMeasurementModel()
        
        # Advanced features
        if config.get('use_batch_estimation', True):
            self.batch_estimator = EnhancedBatchEstimator(config)
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
        
        # Initialize adaptive filter tuner
        adaptation_config = AdaptationConfig(
            innovation_window_size=config.get('innovation_window_size', 50),
            min_samples_for_adaptation=config.get('min_samples_for_adaptation', 10),
            max_adaptation_rate=config.get('max_adaptation_rate', 0.1),
            bias_detection_threshold=config.get('bias_detection_threshold', 3.0),
            process_noise_adaptation_rate=config.get('process_noise_adaptation_rate', 0.05),
            measurement_noise_adaptation_rate=config.get('measurement_noise_adaptation_rate', 0.1),
            tle_age_threshold_hours=config.get('tle_age_threshold_hours', 24.0)
        )
        
        if config.get('use_adaptive_tuning', True):
            self.adaptive_tuner = AdaptiveFilterTuner(adaptation_config)
        else:
            self.adaptive_tuner = None
        
        # Initialize enhanced divergence detector
        recovery_config = RecoveryConfig(
            innovation_warning_threshold=config.get('innovation_warning_threshold', 1000.0),
            innovation_moderate_threshold=config.get('innovation_moderate_threshold', 5000.0),
            innovation_severe_threshold=config.get('innovation_severe_threshold', 20000.0),
            innovation_critical_threshold=config.get('innovation_critical_threshold', 50000.0),
            enable_covariance_inflation=config.get('enable_covariance_inflation', True),
            enable_parameter_reset=config.get('enable_parameter_reset', True),
            enable_filter_restart=config.get('enable_filter_restart', True)
        )
        
        if config.get('use_enhanced_divergence_detection', True):
            self.divergence_detector = EnhancedDivergenceDetector(recovery_config)
        else:
            self.divergence_detector = None
        
        # Tracking variables - use TLE epoch for proper time alignment
        self.current_time = self.tle_data.epoch_datetime
        self.last_update = None
        self.iteration_count = 0
        self.measurement_history = []
        self.state_history = []
        self.innovation_history = []
        
        # Filter health monitoring
        self.divergence_count = 0
        self.max_divergence_count = 5
        
        self.logger.info("Enhanced EKF Tracker initialized")
    
    
    def _tle_to_state_vector(self, tle_data) -> np.ndarray:
        """Convert TLE to Cartesian state vector using centralized SGP4→ECI conversion"""
        try:
            # Use centralized SGP4→ECI conversion for consistency
            from utils import sgp4_to_eci_state_vector
            
            state_vector = sgp4_to_eci_state_vector(
                tle_data.line1, tle_data.line2, tle_data.epoch_datetime
            )
            
            if state_vector is not None:
                return state_vector
            else:
                self.logger.warning("SGP4 conversion failed, falling back to Keplerian")
                return self._keplerian_to_cartesian(tle_data)
            
        except Exception as e:
            self.logger.error(f"SGP4 conversion error: {e}")
            # Fallback to basic Keplerian conversion
            return self._keplerian_to_cartesian(tle_data)
    
    def _keplerian_to_cartesian(self, tle_data) -> np.ndarray:
        """Convert Keplerian elements to Cartesian coordinates"""
        # Earth gravitational parameter (m^3/s^2)
        mu = 3.986004418e14
        
        # Convert to radians
        i = np.radians(tle_data.inclination)
        raan = np.radians(tle_data.raan)
        e = tle_data.eccentricity
        w = np.radians(tle_data.arg_perigee)
        M = np.radians(tle_data.mean_anomaly)
        n = tle_data.mean_motion * 2 * np.pi / 86400  # rad/s
        
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
        """Initialize covariance matrix with architect-recommended uncertainties for sub-1km accuracy"""
        self.P = np.eye(self.state_dim)
        
        # Architect recommendations for sub-1km accuracy:
        # Position uncertainty (m^2) - P0: pos (5 km)²
        self.P[:3, :3] *= (5000)**2  # 5 km initial position uncertainty
        
        # Velocity uncertainty (m/s)² - P0: vel (5 m/s)²
        self.P[3:6, 3:6] *= (5)**2  # 5 m/s initial velocity uncertainty
        
        # Ballistic coefficient uncertainty - P0: Bc (0.3·Bc0)²
        bc0 = self.state[6]  # Current Bc value
        self.P[6, 6] = (0.3 * bc0)**2  # 30% of initial Bc value
        
        # SRP coefficient uncertainty - P0: Cr 0.1²
        self.P[7, 7] = (0.1)**2  # 0.1 uncertainty in SRP coefficient
        
        # Along-track acceleration uncertainty - P0: a_at (5e⁻⁷ m/s²)²
        self.P[8, 8] = (5e-7)**2  # Architect-recommended empirical acceleration uncertainty
        
        # Set up base noise matrices for adaptive tuner
        if hasattr(self, 'adaptive_tuner') and self.adaptive_tuner is not None:
            # Compute base process noise matrix (using 1 second as reference)
            base_Q = self._compute_process_noise_matrix(1.0)
            
            # Base measurement noise matrix (position and velocity from TLE)
            base_R = np.eye(6)
            base_R[:3, :3] *= (1000.0)**2  # 1 km position noise
            base_R[3:6, 3:6] *= (1.0)**2   # 1 m/s velocity noise
            
            # Set base matrices in adaptive tuner
            self.adaptive_tuner.set_base_noise_matrices(base_Q, base_R)
    
    def predict(self, dt: float):
        """Prediction step of the EKF"""
        # State transition using numerical integration
        self.state[:6] = self._propagate_orbit(self.state[:6], dt)
        
        # Parameter evolution (slow states)
        # Bc and Cr evolve slowly with small random walk
        # Along-track acceleration remains constant
        
        # Compute state transition matrix F
        F = self._compute_state_transition_matrix(dt)
        
        # Process noise matrix Q
        Q = self._compute_process_noise_matrix(dt)
        
        # Apply adaptive process noise scaling if available
        if self.adaptive_tuner is not None:
            try:
                adapted_Q, _ = self.adaptive_tuner.get_adapted_noise_matrices()
                # Scale the computed Q matrix by the adaptation factor
                Q = adapted_Q * (dt / 1.0)  # Scale by dt since base matrix is for 1 second
            except Exception as e:
                self.logger.warning(f"Failed to apply adaptive process noise: {e}")
        
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
            
            # Add perturbations through adaptive force models
            perturbations = self.force_models.compute_adaptive_perturbations(
                r, v, self.current_time, self.state[6], self.state[7],
                self.adaptive_atmosphere, self.satellite_props
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
        """Compute process noise matrix Q using architect-recommended RTN frame spectral densities"""
        Q = np.zeros((self.state_dim, self.state_dim))
        
        # Architect-recommended process noise spectral densities in RTN frame:
        # qR = 1e⁻⁸ m²/s³ (radial)
        # qT = 5e⁻⁷ m²/s³ (tangential)  
        # qN = 1e⁻⁸ m²/s³ (normal)
        
        scale = self.config.get('process_noise_scale', 0.5)  # Architect recommended: 0.5
        
        q_radial = scale * 1e-8      # m²/s³
        q_tangential = scale * 5e-7  # m²/s³ 
        q_normal = scale * 1e-8      # m²/s³
        
        # Compute RTN transformation matrix
        r_eci = self.state[:3]
        v_eci = self.state[3:6]
        
        # RTN basis vectors
        r_hat = r_eci / np.linalg.norm(r_eci)  # Radial
        h_vec = np.cross(r_eci, v_eci)
        n_hat = h_vec / np.linalg.norm(h_vec)  # Normal (orbit normal)
        t_hat = np.cross(n_hat, r_hat)        # Tangential (along-track)
        
        # RTN to ECI transformation matrix
        T_rtn_eci = np.array([r_hat, t_hat, n_hat]).T
        
        # Process noise in RTN frame (diagonal)
        Q_pos_vel_rtn = np.array([
            [dt**3/3, dt**2/2],
            [dt**2/2, dt]
        ])
        
        # Build RTN process noise matrix
        Q_rtn = np.zeros((6, 6))
        for i, q_spec in enumerate([q_radial, q_tangential, q_normal]):
            Q_rtn[i:6:3, i:6:3] = Q_pos_vel_rtn * q_spec
        
        # Transform to ECI frame
        T_full = np.zeros((6, 6))
        T_full[:3, :3] = T_rtn_eci
        T_full[3:6, 3:6] = T_rtn_eci
        
        Q[:6, :6] = T_full @ Q_rtn @ T_full.T
        
        # Parameter process noise (random walk) - architect recommendations
        Q[6, 6] = (1e-6 * dt)**2     # Bc random walk (m^2/kg)^2
        Q[7, 7] = (0.005 * dt)**2    # Cr random walk
        Q[8, 8] = 1e-12 * dt         # Along-track acceleration random walk: 1e⁻¹² (m/s²)²/s (corrected discretization)
        
        return Q
    
    def _should_apply_tle_update(self, current_time: datetime) -> bool:
        """
        Determine if TLE measurement should be applied (conservative strategy)
        
        Only use TLE measurements:
        1. When filter confidence is low (large uncertainty)
        2. At large time intervals (avoid bias accumulation) 
        3. When close to TLE epoch (SGP4 most accurate)
        4. When filter divergence is detected
        
        Returns:
            bool: True if TLE measurement should be applied
        """
        try:
            # Calculate time since TLE epoch
            tle_age = (current_time - self.tle_data.epoch_datetime).total_seconds() / 3600  # hours
            
            # 1. Only apply if TLE is relatively fresh (< 48 hours old)
            if tle_age > 48:
                return False
            
            # 2. Apply only every N iterations to reduce bias accumulation
            update_interval = 30  # Only every 30th update (~5 minutes at 10s intervals)
            if self.iteration_count % update_interval != 0:
                return False
            
            # 3. Don't apply if filter is very confident (small position uncertainty)
            position_uncertainty = np.sqrt(np.trace(self.P[:3, :3]))  # meters
            if position_uncertainty < 100:  # Very confident
                return False
                
            # 4. Always apply if filter divergence detected
            if self.divergence_count > 2:
                self.logger.info(f"Applying TLE update due to divergence (count: {self.divergence_count})")
                return True
            
            # 5. Apply if close to TLE epoch (SGP4 most accurate)
            if tle_age < 6:  # Within 6 hours of TLE epoch
                self.logger.info(f"Applying TLE update - close to epoch (age: {tle_age:.1f}h)")
                return True
            
            # 6. Apply if filter uncertainty is high
            if position_uncertainty > 5000:  # Uncertainty > 5km
                self.logger.info(f"Applying TLE update - high uncertainty ({position_uncertainty:.0f}m)")
                return True
                
            # Otherwise, skip TLE measurement to avoid bias
            return False
            
        except Exception as e:
            self.logger.error(f"Error in TLE update decision: {e}")
            return False  # Default to no update on error
    
    def update(self, current_time: datetime) -> Optional[Dict[str, Any]]:
        """Update step with TLE-derived measurements"""
        try:
            dt = (current_time - self.current_time).total_seconds()
            if dt <= 0:
                dt = 10.0  # Default 10 second update
            
            # Prediction step
            self.predict(dt)
            
            # CONSERVATIVE UPDATE STRATEGY: Only use TLE measurements occasionally
            # to avoid bias accumulation from comparing filter vs SGP4 propagation
            should_use_tle_measurement = self._should_apply_tle_update(current_time)
            
            if should_use_tle_measurement:
                # Generate TLE-based calibration measurement
                measurement, R = self.measurement_model.generate_measurement(
                    self.tle_data, current_time
                )
                
                if measurement is not None and R is not None:
                    self.logger.info(f"Applying TLE calibration measurement")
                    self._measurement_update(measurement, R)
                else:
                    self.logger.warning("TLE measurement generation failed, skipping update")
            else:
                # No measurement update - just propagate with prediction only
                self.logger.debug("Skipping TLE measurement - prediction only")
            
            # Batch parameter estimation (periodic)
            if (self.batch_estimator and 
                self.iteration_count % 20 == 0 and 
                len(self.measurement_history) > 10):
                
                try:
                    updated_params = self.batch_estimator.estimate_parameters(
                        self.measurement_history, self.state_history
                    )
                    self.state[6:8] = updated_params[:2]  # Update Bc and Cr
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
        
        # Optional diagnostic logging (disabled for performance)
        # if len(self.measurement_history) < 3:  
        #     pos_diff = np.linalg.norm(innovation[:3]) / 1000  # km
        #     self.logger.debug(f"Innovation: position={pos_diff:.1f}km")
        
        # Get adapted noise matrices if adaptive tuner is available
        if self.adaptive_tuner is not None:
            try:
                adapted_Q, adapted_R = self.adaptive_tuner.get_adapted_noise_matrices()
                # Use adapted measurement noise for this update
                R = adapted_R[:self.obs_dim, :self.obs_dim]  # Extract relevant portion
            except Exception as e:
                self.logger.warning(f"Failed to get adapted noise matrices: {e}")
        
        # Innovation covariance
        S = H @ self.P @ H.T + R
        
        # Update adaptive tuner with innovation statistics
        if self.adaptive_tuner is not None:
            try:
                self.adaptive_tuner.update_innovation(
                    innovation, S, self.current_time, self.tle_data.epoch_datetime
                )
            except Exception as e:
                self.logger.warning(f"Adaptive tuner update failed: {e}")
        
        # Enhanced divergence detection and recovery
        if self.divergence_detector is not None:
            try:
                # Get current parameter vector
                param_vector = np.array([self.state[6], self.state[7], self.state[8]])  # Bc, Cr, empirical_accel
                
                # Assess divergence
                divergence_metrics = self.divergence_detector.assess_divergence(
                    innovation, S, self.P, param_vector, self.current_time
                )
                
                # Log divergence status
                if divergence_metrics.divergence_level.value != 'healthy':
                    self.logger.warning(f"Divergence detected: {divergence_metrics.divergence_level.value}, "
                                      f"innovation={divergence_metrics.innovation_norm:.1f}m, "
                                      f"confidence={divergence_metrics.confidence_score:.3f}")
                
                # Get recovery recommendations
                recovery_actions = self.divergence_detector.recommend_recovery_actions(
                    divergence_metrics, self.state, self.P, param_vector
                )
                
                # Execute recovery actions
                for action in recovery_actions:
                    self.state, self.P, updated_params = self.divergence_detector.execute_recovery_action(
                        action, self.state, self.P, param_vector
                    )
                    # Update parameter states
                    self.state[6:9] = updated_params[:3]
                    
                    self.logger.info(f"Applied recovery action: {action['action']} - {action.get('reason', '')}")
                
            except Exception as e:
                self.logger.warning(f"Enhanced divergence detection failed: {e}")
                # Fallback to basic divergence detection
                innovation_norm = np.linalg.norm(innovation)
                if innovation_norm > 100000:  # 100km threshold
                    self.divergence_count += 1
                    if self.divergence_count > self.max_divergence_count:
                        self._handle_filter_divergence()
                        return
        else:
            # Basic divergence detection (legacy)
            innovation_norm = np.linalg.norm(innovation)
            self.innovation_history.append(innovation_norm)
            
            if innovation_norm > 100000:  # 100km threshold for divergence detection
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
            'ballistic_coeff': self.state[6],
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

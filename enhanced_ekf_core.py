import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime, timedelta
import logging
from scipy.linalg import cholesky, solve_triangular, LinAlgError
import warnings

from satellite_characterizer import SatelliteProperties
from adaptive_atmospheric_model import AdaptiveAtmosphericModel
from enhanced_srp_model import EnhancedSRPModel
from adaptive_geopotential_model import AdaptiveGeopotentialModel

class EnhancedEKFCore:
    """
    Enhanced Extended Kalman Filter core with robust numerical stabilization,
    Joseph form covariance updates, and square-root filtering options
    """
    
    def __init__(self, satellite_props: SatelliteProperties, config: Optional[Dict[str, Any]] = None):
        """Initialize enhanced EKF core"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Satellite properties
        self.satellite_props = satellite_props
        
        # State vector configuration
        self.state_dim = self.config.get('state_dimension', 9)
        self.obs_dim = self.config.get('observation_dimension', 6)
        
        # Numerical stabilization options - always use Joseph form for robustness
        self.use_joseph_form = True  # Always use Joseph form for stability
        self.use_square_root = self.config.get('use_square_root_filtering', False)
        self.use_adaptive_regularization = self.config.get('adaptive_regularization', True)
        
        # Regularization parameters - much more aggressive for stability
        self.min_eigenvalue = self.config.get('min_eigenvalue', 1e-8)
        self.regularization_factor = self.config.get('regularization_factor', 1e-4)
        self.condition_number_threshold = self.config.get('condition_threshold', 1e6)
        
        # Log throttling for warnings
        self.last_warning_times = {}
        self.warning_throttle_seconds = self.config.get('warning_throttle_seconds', 10.0)
        
        # Initialize state and covariance
        self._initialize_state_and_covariance()
        
        # Initialize force models
        self._initialize_force_models()
        
        # Numerical health monitoring
        self.numerical_health = {
            'condition_numbers': [],
            'eigenvalue_ratios': [],
            'regularization_count': 0,
            'joseph_form_count': 0,
            'square_root_count': 0
        }
        
        # Filter statistics
        self.filter_stats = {
            'total_predictions': 0,
            'total_updates': 0,
            'numerical_issues': 0,
            'successful_updates': 0
        }
        
        self.logger.info(f"Enhanced EKF core initialized for {satellite_props.satellite_type.value}")
    
    def _initialize_state_and_covariance(self):
        """Initialize state vector and covariance matrix"""
        try:
            # State vector: [position(3), velocity(3), ballistic_coeff(1), srp_coeff(1), empirical_accel(1)]
            self.state = np.zeros(self.state_dim)
            
            # Initialize from satellite properties
            # Position and velocity will be set from TLE
            self.state[6] = self.satellite_props.ballistic_coefficient
            self.state[7] = self.satellite_props.srp_coefficient
            self.state[8] = 0.0  # Empirical along-track acceleration
            
            # Initialize covariance matrix with proper scaling
            self.P = self._create_initial_covariance()
            
            # Square-root covariance (if using square-root filtering)
            if self.use_square_root:
                self.S = self._compute_square_root(self.P)
            
            self.logger.info("State and covariance initialized")
            
        except Exception as e:
            self.logger.error(f"State initialization failed: {e}")
            raise
    
    def _create_initial_covariance(self) -> np.ndarray:
        """Create initial covariance matrix with proper uncertainty bounds"""
        try:
            P = np.eye(self.state_dim)
            
            # Position uncertainty (m²) - more conservative for numerical stability
            pos_uncertainty = self.config.get('initial_position_uncertainty', 1000.0)  # 1000m - more conservative
            P[:3, :3] *= pos_uncertainty**2
            
            # Velocity uncertainty (m/s)² - more conservative
            vel_uncertainty = self.config.get('initial_velocity_uncertainty', 2.0)  # 2.0 m/s - more conservative
            P[3:6, 3:6] *= vel_uncertainty**2
            
            # Ballistic coefficient uncertainty - use more conservative bounds
            bc_base = self.satellite_props.ballistic_coefficient
            bc_std = max(bc_base * 0.1, 1e-4)  # 10% of value or minimum 1e-4
            P[6, 6] = bc_std**2
            
            # SRP coefficient uncertainty - more conservative
            srp_base = self.satellite_props.srp_coefficient
            srp_std = max(srp_base * 0.1, 0.05)  # 10% of value or minimum 0.05
            P[7, 7] = srp_std**2
            
            # Empirical acceleration uncertainty - smaller initial value
            emp_std = self.config.get('empirical_accel_std', 1e-7)  # m/s²
            P[8, 8] = emp_std**2
            
            # Ensure all diagonal elements are reasonable
            min_variance = 1e-12
            max_variance = 1e8
            
            for i in range(self.state_dim):
                P[i, i] = max(min_variance, min(P[i, i], max_variance))
            
            return P
            
        except Exception as e:
            self.logger.error(f"Initial covariance creation failed: {e}")
            return np.eye(self.state_dim) * 1e-6
    
    def _initialize_force_models(self):
        """Initialize enhanced force models"""
        try:
            # Initialize adaptive force models
            self.atmospheric_model = AdaptiveAtmosphericModel(self.config)
            self.srp_model = EnhancedSRPModel(self.config)
            self.geopotential_model = AdaptiveGeopotentialModel(self.config)
            
            # Configure models for this satellite
            self.atmospheric_model.configure_for_satellite(self.satellite_props)
            
            self.logger.info("Enhanced force models initialized")
            
        except Exception as e:
            self.logger.error(f"Force model initialization failed: {e}")
            # Create minimal fallback models
            self.atmospheric_model = None
            self.srp_model = None
            self.geopotential_model = None
    
    def predict(self, dt: float, current_time: datetime) -> bool:
        """
        Enhanced prediction step with robust numerical stabilization
        
        Args:
            dt: Time step (seconds)
            current_time: Current UTC time
            
        Returns:
            True if prediction successful, False otherwise
        """
        try:
            self.filter_stats['total_predictions'] += 1
            
            # Propagate state using enhanced force models
            new_state = self._propagate_state(self.state, dt, current_time)
            
            # Compute state transition matrix
            F = self._compute_state_transition_matrix(dt, current_time)
            
            # Compute process noise matrix
            Q = self._compute_process_noise_matrix(dt, current_time)
            
            # Covariance prediction with numerical stabilization
            if self.use_square_root:
                success = self._square_root_predict(F, Q)
            else:
                success = self._standard_predict(F, Q)
            
            if success:
                self.state = new_state
                
                # Apply adaptive regularization if needed
                if self.use_adaptive_regularization:
                    self._apply_adaptive_regularization()
                
                return True
            else:
                self.logger.warning("Prediction failed due to numerical issues")
                self.filter_stats['numerical_issues'] += 1
                return False
                
        except Exception as e:
            self.logger.error(f"Prediction step failed: {e}")
            self.filter_stats['numerical_issues'] += 1
            return False
    
    def update(self, measurement: np.ndarray, R: np.ndarray, H: Optional[np.ndarray] = None) -> bool:
        """
        Enhanced measurement update with robust numerical stabilization
        Always uses Joseph form for maximum stability
        
        Args:
            measurement: Measurement vector
            R: Measurement noise covariance
            H: Measurement matrix (optional, defaults to identity for pos/vel)
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            self.filter_stats['total_updates'] += 1
            
            # Default measurement matrix (observe position and velocity)
            if H is None:
                H = np.zeros((self.obs_dim, self.state_dim))
                H[:6, :6] = np.eye(6)
            
            # Predicted measurement
            h_pred = H @ self.state
            
            # Innovation
            innovation = measurement - h_pred
            
            # Innovation covariance
            S = H @ self.P @ H.T + R
            
            # Check for numerical issues
            if not self._check_innovation_covariance(S):
                self._throttled_warning("innovation_cov", "Innovation covariance is ill-conditioned")
                self.filter_stats['numerical_issues'] += 1
                return False
            
            # Always use Joseph form for maximum robustness
            success = self._joseph_form_update(H, innovation, S, R)
            self.numerical_health['joseph_form_count'] += 1
            
            if success:
                self.filter_stats['successful_updates'] += 1
                
                # Ensure covariance symmetry and positive definiteness
                self._enforce_covariance_symmetry()
                
                # Apply adaptive regularization after update
                if self.use_adaptive_regularization:
                    self._apply_adaptive_regularization()
                
                return True
            else:
                self._throttled_warning("measurement_update", "Measurement update failed due to numerical issues")
                self.filter_stats['numerical_issues'] += 1
                return False
                
        except Exception as e:
            self._throttled_warning("measurement_update", f"Measurement update failed: {e}")
            self.filter_stats['numerical_issues'] += 1
            return False
    
    def _propagate_state(self, state: np.ndarray, dt: float, current_time: datetime) -> np.ndarray:
        """Propagate full state vector using enhanced force models with RK4 integration"""
        try:
            # Full state integration - propagate entire 9-dimensional state vector
            def dynamics(t, y):
                """Full state dynamics including parameters"""
                r = y[:3]
                v = y[3:6]
                bc = y[6]      # Ballistic coefficient
                cr = y[7]      # SRP coefficient  
                emp_accel = y[8]  # Empirical acceleration
                
                # Two-body acceleration
                r_norm = np.linalg.norm(r)
                if r_norm < 1e3:  # Sanity check - avoid singularity
                    return np.zeros(self.state_dim)
                    
                mu = 3.986004418e14  # Earth gravitational parameter
                accel = -mu * r / (r_norm**3)
                
                # Add perturbations from enhanced force models
                try:
                    # Atmospheric drag
                    if self.atmospheric_model:
                        drag_density, _ = self.atmospheric_model.get_adaptive_density(
                            self.satellite_props, r, current_time + timedelta(seconds=t)
                        )
                        
                        # Compute drag acceleration with robustness check
                        omega_earth = 7.2921159e-5  # rad/s
                        v_rot = np.array([-omega_earth * r[1], omega_earth * r[0], 0])
                        v_rel = v - v_rot
                        v_rel_mag = np.linalg.norm(v_rel)
                        
                        if v_rel_mag > 1e-6 and drag_density > 0:  # Epsilon check for robustness
                            drag_accel = -0.5 * drag_density * bc * v_rel_mag * v_rel
                            accel += drag_accel
                    
                    # Solar radiation pressure
                    if self.srp_model:
                        srp_accel, _ = self.srp_model.compute_srp_acceleration(
                            self.satellite_props, r, v, current_time + timedelta(seconds=t)
                        )
                        accel += srp_accel
                    
                    # Geopotential perturbations
                    if self.geopotential_model:
                        geo_accel, _ = self.geopotential_model.compute_geopotential_acceleration(
                            self.satellite_props, r, current_time + timedelta(seconds=t)
                        )
                        accel += geo_accel
                    
                    # Empirical along-track acceleration with robustness
                    h = np.cross(r, v)
                    h_norm = np.linalg.norm(h)
                    if h_norm > 1e-6:  # Epsilon check to avoid division by zero
                        r_unit = r / np.linalg.norm(r)
                        along_track = np.cross(h, r)
                        along_track_norm = np.linalg.norm(along_track)
                        if along_track_norm > 1e-6:  # Additional robustness check
                            along_track = along_track / along_track_norm
                            accel += emp_accel * along_track
                        
                except Exception as e:
                    self._throttled_warning("perturbation_calc", f"Perturbation calculation failed: {e}")
                
                # Parameter derivatives (process noise handled in covariance)
                dbc_dt = 0.0        # Ballistic coefficient evolves via process noise
                dcr_dt = 0.0        # SRP coefficient evolves via process noise
                demp_dt = 0.0       # Empirical acceleration evolves via process noise
                
                # Return full state derivative
                return np.array([
                    v[0], v[1], v[2],           # Position derivatives (velocity)
                    accel[0], accel[1], accel[2],  # Velocity derivatives (acceleration)
                    dbc_dt, dcr_dt, demp_dt     # Parameter derivatives
                ])
            
            # RK4 integration for full state vector
            k1 = dynamics(0, state)
            k2 = dynamics(dt/2, state + dt/2 * k1)
            k3 = dynamics(dt/2, state + dt/2 * k2)
            k4 = dynamics(dt, state + dt * k3)
            
            new_state = state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            
            # Sanity checks on propagated state
            if not np.all(np.isfinite(new_state)):
                self._throttled_warning("state_propagation", "Non-finite values in propagated state")
                return state  # Return original state if propagation failed
            
            # Ensure parameters stay within reasonable bounds
            new_state[6] = np.clip(new_state[6], 0.001, 0.020)  # Ballistic coefficient bounds
            new_state[7] = np.clip(new_state[7], 0.5, 2.5)     # SRP coefficient bounds
            new_state[8] = np.clip(new_state[8], -1e-5, 1e-5)  # Empirical acceleration bounds
            
            return new_state
            
        except Exception as e:
            self._throttled_warning("state_propagation", f"State propagation failed: {e}")
            return state  # Return unchanged state on error
    
    def _compute_state_transition_matrix(self, dt: float, current_time: datetime) -> np.ndarray:
        """Compute state transition matrix F"""
        try:
            F = np.eye(self.state_dim)
            
            # Position-velocity coupling
            F[:3, 3:6] = np.eye(3) * dt
            
            # Velocity-position coupling (gravity gradient)
            r = self.state[:3]
            r_norm = np.linalg.norm(r)
            
            if r_norm > 0:
                mu = 3.986004418e14
                
                # Gravity gradient matrix
                I = np.eye(3)
                rr = np.outer(r, r)
                gravity_grad = -mu / (r_norm**3) * (I - 3 * rr / (r_norm**2))
                
                F[3:6, :3] = gravity_grad * dt
            
            # Parameters are constant (F[6:, 6:] already identity)
            
            return F
            
        except Exception as e:
            self.logger.error(f"State transition matrix computation failed: {e}")
            return np.eye(self.state_dim)
    
    def _compute_process_noise_matrix(self, dt: float, current_time: datetime) -> np.ndarray:
        """Compute process noise matrix Q with adaptive scaling"""
        try:
            Q = np.zeros((self.state_dim, self.state_dim))
            
            # Position and velocity process noise (RTN frame)
            r_eci = self.state[:3]
            v_eci = self.state[3:6]
            
            # RTN basis vectors
            r_hat = r_eci / np.linalg.norm(r_eci)
            h_vec = np.cross(r_eci, v_eci)
            n_hat = h_vec / np.linalg.norm(h_vec)
            t_hat = np.cross(n_hat, r_hat)
            
            # RTN to ECI transformation
            T_rtn_eci = np.array([r_hat, t_hat, n_hat]).T
            
            # Process noise spectral densities (adaptive based on satellite)
            confidence = self.satellite_props.characterization_confidence
            scale_factor = 2.0 - 0.5 * confidence  # More conservative scaling
            
            # Increased process noise for LEO satellites with atmospheric drag
            q_radial = scale_factor * 1e-6       # m²/s³ - increased for atmospheric uncertainty
            q_tangential = scale_factor * 5e-6   # m²/s³ - increased for drag uncertainty
            q_normal = scale_factor * 1e-7       # m²/s³ - increased for perturbations
            
            # Build process noise in RTN frame
            Q_pos_vel_rtn = np.array([
                [dt**3/3, dt**2/2],
                [dt**2/2, dt]
            ])
            
            Q_rtn = np.zeros((6, 6))
            for i, q_spec in enumerate([q_radial, q_tangential, q_normal]):
                Q_rtn[i:6:3, i:6:3] = Q_pos_vel_rtn * q_spec
            
            # Transform to ECI frame
            T_full = np.zeros((6, 6))
            T_full[:3, :3] = T_rtn_eci
            T_full[3:6, 3:6] = T_rtn_eci
            
            Q[:6, :6] = T_full @ Q_rtn @ T_full.T
            
            # Parameter process noise (random walk) - increased for stability
            Q[6, 6] = 1e-8 * dt * scale_factor          # Ballistic coefficient (m²/kg)²/s - increased
            Q[7, 7] = 1e-4 * dt * scale_factor          # SRP coefficient variance/s - increased
            Q[8, 8] = 1e-10 * dt * scale_factor         # Empirical acceleration (m/s²)²/s - increased
            
            return Q
            
        except Exception as e:
            self.logger.error(f"Process noise matrix computation failed: {e}")
            return np.eye(self.state_dim) * 1e-6
    
    def _standard_predict(self, F: np.ndarray, Q: np.ndarray) -> bool:
        """Standard covariance prediction"""
        try:
            # P = F * P * F^T + Q
            self.P = F @ self.P @ F.T + Q
            
            # Check for numerical issues
            return self._check_covariance_health(self.P)
            
        except Exception as e:
            self.logger.error(f"Standard prediction failed: {e}")
            return False
    
    def _square_root_predict(self, F: np.ndarray, Q: np.ndarray) -> bool:
        """Square-root covariance prediction"""
        try:
            # S * S^T = P, so we need to update S
            # This is more complex and requires specialized algorithms
            # For now, fall back to standard prediction and recompute square root
            
            success = self._standard_predict(F, Q)
            if success:
                self.S = self._compute_square_root(self.P)
                self.numerical_health['square_root_count'] += 1
            
            return success
            
        except Exception as e:
            self.logger.error(f"Square-root prediction failed: {e}")
            return False
    
    def _standard_update(self, H: np.ndarray, innovation: np.ndarray, S: np.ndarray) -> bool:
        """Standard Kalman gain update"""
        try:
            # Kalman gain
            K = self.P @ H.T @ np.linalg.inv(S)
            
            # State update
            self.state += K @ innovation
            
            # Covariance update
            I_KH = np.eye(self.state_dim) - K @ H
            self.P = I_KH @ self.P
            
            return self._check_covariance_health(self.P)
            
        except (LinAlgError, np.linalg.LinAlgError) as e:
            self.logger.warning(f"Standard update failed due to singular matrix: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Standard update failed: {e}")
            return False
    
    def _joseph_form_update(self, H: np.ndarray, innovation: np.ndarray, 
                          S: np.ndarray, R: np.ndarray) -> bool:
        """Joseph form covariance update for numerical stability"""
        try:
            # Kalman gain
            K = self.P @ H.T @ np.linalg.inv(S)
            
            # State update
            self.state += K @ innovation
            
            # Joseph form covariance update: P = (I - KH) P (I - KH)^T + K R K^T
            I_KH = np.eye(self.state_dim) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
            
            return self._check_covariance_health(self.P)
            
        except (LinAlgError, np.linalg.LinAlgError) as e:
            self.logger.warning(f"Joseph form update failed due to singular matrix: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Joseph form update failed: {e}")
            return False
    
    def _square_root_update(self, H: np.ndarray, innovation: np.ndarray,
                          S: np.ndarray, R: np.ndarray) -> bool:
        """Square-root filtering update"""
        try:
            # This is a simplified implementation
            # Full square-root filtering requires specialized algorithms like Potter's method
            
            # Fall back to Joseph form and recompute square root
            success = self._joseph_form_update(H, innovation, S, R)
            if success:
                self.S = self._compute_square_root(self.P)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Square-root update failed: {e}")
            return False
    
    def _compute_square_root(self, P: np.ndarray) -> np.ndarray:
        """Compute square root of covariance matrix"""
        try:
            # Use Cholesky decomposition
            return cholesky(P, lower=True)
            
        except LinAlgError:
            # If Cholesky fails, use eigenvalue decomposition
            try:
                eigenvals, eigenvecs = np.linalg.eigh(P)
                eigenvals = np.maximum(eigenvals, self.min_eigenvalue)
                return eigenvecs @ np.diag(np.sqrt(eigenvals))
                
            except Exception as e:
                self.logger.error(f"Square root computation failed: {e}")
                return np.eye(self.state_dim) * 1e-3
    
    def _check_covariance_health(self, P: np.ndarray) -> bool:
        """Check covariance matrix for numerical health"""
        try:
            # Check for NaN or infinite values
            if not np.all(np.isfinite(P)):
                self._throttled_warning("covariance_health", "Covariance contains NaN or infinite values")
                return False
            
            # Check symmetry and enforce it
            if not np.allclose(P, P.T, rtol=1e-10):
                self._throttled_warning("covariance_health", "Covariance matrix is not symmetric")
                # Force symmetry and reassign back to self.P
                self.P = (P + P.T) / 2
                P = self.P  # Use the corrected matrix for further checks
            
            # Check positive definiteness
            eigenvals = np.linalg.eigvals(P)
            min_eigenval = np.min(eigenvals)
            max_eigenval = np.max(eigenvals)
            
            if min_eigenval <= 0:
                self._throttled_warning("covariance_health", f"Covariance has non-positive eigenvalue: {min_eigenval}")
                return False
            
            # Check condition number
            condition_number = max_eigenval / min_eigenval
            self.numerical_health['condition_numbers'].append(condition_number)
            self.numerical_health['eigenvalue_ratios'].append(max_eigenval / min_eigenval)
            
            if condition_number > self.condition_number_threshold:
                self._throttled_warning("covariance_health", f"Covariance is ill-conditioned: {condition_number}")
                return False
            
            return True
            
        except Exception as e:
            self._throttled_warning("covariance_health", f"Covariance health check failed: {e}")
            return False
    
    def _check_innovation_covariance(self, S: np.ndarray) -> bool:
        """Check innovation covariance for numerical issues"""
        try:
            # Check condition number
            condition_number = np.linalg.cond(S)
            
            if condition_number > self.condition_number_threshold:
                self.logger.warning(f"Innovation covariance is ill-conditioned: {condition_number}")
                return False
            
            # Check positive definiteness
            eigenvals = np.linalg.eigvals(S)
            if np.any(eigenvals <= 0):
                self.logger.warning("Innovation covariance is not positive definite")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Innovation covariance check failed: {e}")
            return False
    
    def _apply_adaptive_regularization(self):
        """Apply adaptive regularization to prevent covariance collapse"""
        try:
            eigenvals, eigenvecs = np.linalg.eigh(self.P)
            
            # Check if regularization is needed
            min_eigenval = np.min(eigenvals)
            
            # Check condition number
            condition_number = np.max(eigenvals) / np.max([np.min(eigenvals), 1e-15])
            
            if min_eigenval < self.min_eigenvalue or condition_number > self.condition_number_threshold:
                # Apply aggressive regularization
                regularized_eigenvals = np.maximum(eigenvals, self.min_eigenvalue)
                
                # Add regularization proportional to maximum eigenvalue
                regularization = self.regularization_factor * np.max(eigenvals)
                regularized_eigenvals += regularization
                
                # Reconstruct covariance matrix
                self.P = eigenvecs @ np.diag(regularized_eigenvals) @ eigenvecs.T
                
                self.numerical_health['regularization_count'] += 1
                self.logger.debug(f"Applied regularization: min eigenvalue was {min_eigenval}")
                
        except Exception as e:
            self.logger.error(f"Adaptive regularization failed: {e}")
    
    def get_filter_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive filter diagnostics"""
        try:
            # Compute current covariance statistics
            eigenvals = np.linalg.eigvals(self.P)
            condition_number = np.max(eigenvals) / np.min(eigenvals)
            
            # Position and velocity uncertainties
            pos_uncertainty = np.sqrt(np.trace(self.P[:3, :3]))
            vel_uncertainty = np.sqrt(np.trace(self.P[3:6, 3:6]))
            
            diagnostics = {
                'filter_health': {
                    'condition_number': condition_number,
                    'min_eigenvalue': np.min(eigenvals),
                    'max_eigenvalue': np.max(eigenvals),
                    'position_uncertainty_m': pos_uncertainty,
                    'velocity_uncertainty_ms': vel_uncertainty
                },
                'numerical_stability': {
                    'regularization_count': self.numerical_health['regularization_count'],
                    'joseph_form_count': self.numerical_health['joseph_form_count'],
                    'square_root_count': self.numerical_health['square_root_count'],
                    'avg_condition_number': np.mean(self.numerical_health['condition_numbers']) if self.numerical_health['condition_numbers'] else 0
                },
                'filter_statistics': self.filter_stats.copy(),
                'configuration': {
                    'use_joseph_form': self.use_joseph_form,
                    'use_square_root': self.use_square_root,
                    'use_adaptive_regularization': self.use_adaptive_regularization,
                    'state_dimension': self.state_dim
                },
                'current_state': {
                    'ballistic_coefficient': self.state[6],
                    'srp_coefficient': self.state[7],
                    'empirical_acceleration': self.state[8]
                }
            }
            
            return diagnostics
            
        except Exception as e:
            self.logger.error(f"Filter diagnostics failed: {e}")
            return {'error': str(e)}
    
    def reset_filter(self, new_state: Optional[np.ndarray] = None):
        """Reset filter to initial conditions or provided state"""
        try:
            if new_state is not None:
                self.state = new_state.copy()
            else:
                # Reset to initial state
                self.state = np.zeros(self.state_dim)
                self.state[6] = self.satellite_props.ballistic_coefficient
                self.state[7] = self.satellite_props.srp_coefficient
                self.state[8] = 0.0
            
            # Reset covariance
            self.P = self._create_initial_covariance()
            
            if self.use_square_root:
                self.S = self._compute_square_root(self.P)
            
            # Reset statistics
            self.numerical_health = {
                'condition_numbers': [],
                'eigenvalue_ratios': [],
                'regularization_count': 0,
                'joseph_form_count': 0,
                'square_root_count': 0
            }
            
            self.filter_stats = {
                'total_predictions': 0,
                'total_updates': 0,
                'numerical_issues': 0,
                'successful_updates': 0
            }
            
            self.logger.info("Filter reset successfully")
            
        except Exception as e:
            self.logger.error(f"Filter reset failed: {e}")
    
    def get_state_estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current state estimate and covariance"""
        return self.state.copy(), self.P.copy()
    
    def get_position_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current position and velocity estimates"""
        return self.state[:3].copy(), self.state[3:6].copy()
    
    def get_parameter_estimates(self) -> Dict[str, float]:
        """Get current parameter estimates"""
        return {
            'ballistic_coefficient': self.state[6],
            'srp_coefficient': self.state[7],
            'empirical_acceleration': self.state[8]
        }
    
    def _throttled_warning(self, warning_key: str, message: str):
        """Log warning with throttling to prevent flooding"""
        current_time = datetime.utcnow().timestamp()
        
        if (warning_key not in self.last_warning_times or 
            current_time - self.last_warning_times[warning_key] > self.warning_throttle_seconds):
            
            self.logger.warning(message)
            self.last_warning_times[warning_key] = current_time
    
    def _enforce_covariance_symmetry(self):
        """Enforce covariance matrix symmetry after updates"""
        try:
            # Always enforce symmetry after updates for robustness
            if not np.allclose(self.P, self.P.T, rtol=1e-12):
                self.P = (self.P + self.P.T) / 2
                
        except Exception as e:
            self._throttled_warning("covariance_symmetry", f"Covariance symmetry enforcement failed: {e}")
    
    def initialize_state(self, initial_state: np.ndarray, initial_covariance: np.ndarray, 
                        initial_time: datetime):
        """
        Initialize EKF state with provided initial conditions
        
        Args:
            initial_state: Initial state vector [x, y, z, vx, vy, vz, ...]
            initial_covariance: Initial covariance matrix
            initial_time: Initial timestamp
        """
        try:
            # Ensure state vector has correct dimensions
            if len(initial_state) >= 6:
                # Copy position and velocity
                self.state[:6] = initial_state[:6]
                
                # Keep existing parameter estimates if not provided
                if len(initial_state) >= self.state_dim:
                    self.state = initial_state[:self.state_dim].copy()
            else:
                raise ValueError(f"Initial state must have at least 6 elements, got {len(initial_state)}")
            
            # Set covariance matrix
            if initial_covariance.shape == (6, 6):
                # Expand to full state dimension
                self.P = np.eye(self.state_dim) * 1e-6
                self.P[:6, :6] = initial_covariance
            elif initial_covariance.shape == (self.state_dim, self.state_dim):
                self.P = initial_covariance.copy()
            else:
                raise ValueError(f"Initial covariance shape {initial_covariance.shape} not compatible")
            
            # Store initial time
            self.current_time = initial_time
            
            # Reset filter statistics
            self.filter_stats = {
                'total_predictions': 0,
                'total_updates': 0,
                'numerical_issues': 0,
                'successful_updates': 0
            }
            
            # Initialize square-root if needed
            if self.use_square_root:
                self.S = self._compute_square_root(self.P)
            
            self.logger.info(f"EKF state initialized at {initial_time}")
            pos_km = self.state[:3]/1000
            vel_kms = self.state[3:6]/1000
            self.logger.debug(f"Initial position: [{pos_km[0]:.3f}, {pos_km[1]:.3f}, {pos_km[2]:.3f}] km")
            self.logger.debug(f"Initial velocity: [{vel_kms[0]:.6f}, {vel_kms[1]:.6f}, {vel_kms[2]:.6f}] km/s")
            
        except Exception as e:
            self.logger.error(f"State initialization failed: {e}")
            raise
    
    def propagate_to_time(self, target_time: datetime) -> np.ndarray:
        """
        Propagate EKF state to target time without measurements
        
        Args:
            target_time: Target timestamp to propagate to
            
        Returns:
            State vector at target time
        """
        try:
            if not hasattr(self, 'current_time'):
                raise ValueError("EKF not initialized - call initialize_state first")
            
            # Calculate time difference
            dt_total = (target_time - self.current_time).total_seconds()
            
            if dt_total < 0:
                raise ValueError(f"Target time {target_time} is before current time {self.current_time}")
            
            if dt_total == 0:
                return self.state.copy()
            
            # Propagate in steps for numerical stability
            max_step_size = 30.0  # Maximum 30 second steps for better stability
            current_time = self.current_time
            
            while (target_time - current_time).total_seconds() > 1e-6:
                remaining_time = (target_time - current_time).total_seconds()
                dt_step = min(max_step_size, remaining_time)
                
                # Perform prediction step
                success = self.predict(dt_step, current_time)
                
                if not success:
                    self.logger.warning(f"Prediction failed during propagation at {current_time}")
                    # Continue with best effort
                
                # Update current time
                current_time += timedelta(seconds=dt_step)
                self.current_time = current_time
            
            # Ensure we're exactly at target time
            self.current_time = target_time
            
            return self.state.copy()
            
        except Exception as e:
            self.logger.error(f"Propagation to {target_time} failed: {e}")
            raise
    
    def get_current_time(self) -> datetime:
        """Get current EKF time"""
        if hasattr(self, 'current_time'):
            return self.current_time
        else:
            raise ValueError("EKF not initialized - no current time available")
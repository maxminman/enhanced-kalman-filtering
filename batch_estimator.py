import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from scipy.optimize import least_squares
from dataclasses import dataclass

@dataclass
class BatchEstimationResult:
    """Results from batch parameter estimation"""
    parameters: np.ndarray
    covariance: np.ndarray
    residuals: np.ndarray
    cost: float
    iterations: int
    success: bool

class BatchEstimator:
    """
    Sliding-window batch parameter estimation using Levenberg-Marquardt
    for robust identification of drag coefficient (CdA) and SRP coefficient (Cr)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize batch estimator"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Estimation parameters
        self.window_size = config.get('batch_window_size', 50)  # Number of measurements
        self.overlap_ratio = config.get('batch_overlap', 0.5)   # Window overlap
        
        # Parameter bounds
        self.param_bounds = {
            'cd_a': (0.5, 5.0),     # Drag coefficient bounds
            'cr_a': (0.5, 2.5),     # SRP coefficient bounds
            'empirical': (-1e-5, 1e-5)  # Along-track acceleration bounds
        }
        
        # Estimation history
        self.estimation_history = []
        
        self.logger.info("Batch parameter estimator initialized")
    
    def estimate_parameters(self, measurement_history: List[Dict[str, Any]], 
                          state_history: List[np.ndarray]) -> np.ndarray:
        """
        Estimate parameters using sliding window batch approach
        
        Args:
            measurement_history: List of measurement dictionaries
            state_history: List of state vectors
            
        Returns:
            Estimated parameters [CdA, Cr, along_track_accel]
        """
        try:
            if len(measurement_history) < self.window_size:
                self.logger.debug("Insufficient data for batch estimation")
                return np.array([self.config.get('drag_coeff', 2.2),
                               self.config.get('srp_coeff', 1.3),
                               0.0])
            
            # Extract recent window
            window_measurements = measurement_history[-self.window_size:]
            window_states = state_history[-self.window_size:]
            
            # Prepare data for estimation
            times, observations, initial_states = self._prepare_estimation_data(
                window_measurements, window_states
            )
            
            # Initial parameter guess
            initial_params = np.array([
                self.config.get('drag_coeff', 2.2),
                self.config.get('srp_coeff', 1.3),
                0.0  # Along-track acceleration
            ])
            
            # Parameter bounds for optimization
            bounds = (
                [self.param_bounds['cd_a'][0], self.param_bounds['cr_a'][0], 
                 self.param_bounds['empirical'][0]],
                [self.param_bounds['cd_a'][1], self.param_bounds['cr_a'][1], 
                 self.param_bounds['empirical'][1]]
            )
            
            # Weighted least squares estimation
            result = self._weighted_least_squares(
                initial_params, times, observations, initial_states, bounds
            )
            
            if result.success:
                # Store estimation result
                self.estimation_history.append({
                    'timestamp': datetime.utcnow(),
                    'parameters': result.parameters.copy(),
                    'covariance': result.covariance.copy(),
                    'cost': result.cost,
                    'window_size': len(window_measurements)
                })
                
                # Keep limited history
                if len(self.estimation_history) > 100:
                    self.estimation_history = self.estimation_history[-100:]
                
                self.logger.debug(f"Batch estimation successful: CdA={result.parameters[0]:.3f}, "
                                f"Cr={result.parameters[1]:.3f}, empirical={result.parameters[2]:.2e}")
                
                return result.parameters
            else:
                self.logger.warning("Batch estimation failed to converge")
                return initial_params
                
        except Exception as e:
            self.logger.error(f"Batch estimation error: {e}")
            # Return default parameters
            return np.array([self.config.get('drag_coeff', 2.2),
                           self.config.get('srp_coeff', 1.3),
                           0.0])
    
    def _prepare_estimation_data(self, measurements: List[Dict[str, Any]], 
                               states: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for parameter estimation"""
        times = []
        observations = []
        initial_states = []
        
        for i, (meas, state) in enumerate(zip(measurements, states)):
            try:
                # Extract timestamp
                timestamp = meas['timestamp']
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                
                # Convert to seconds from first measurement
                if i == 0:
                    ref_time = timestamp
                    times.append(0.0)
                else:
                    dt = (timestamp - ref_time).total_seconds()
                    times.append(dt)
                
                # Extract measurement (position and velocity)
                obs = meas['measurement'][:6]  # Position and velocity only
                observations.append(obs)
                
                # Extract corresponding state
                initial_states.append(state[:6])  # Position and velocity only
                
            except Exception as e:
                self.logger.warning(f"Data preparation error for measurement {i}: {e}")
                continue
        
        return (np.array(times), np.array(observations), np.array(initial_states))
    
    def _weighted_least_squares(self, initial_params: np.ndarray, times: np.ndarray,
                              observations: np.ndarray, initial_states: np.ndarray,
                              bounds: Tuple) -> BatchEstimationResult:
        """Perform weighted least squares parameter estimation"""
        
        def residual_function(params):
            """Compute residuals for given parameters"""
            try:
                residuals = []
                
                for i, (t, obs, initial_state) in enumerate(zip(times, observations, initial_states)):
                    if i == 0:
                        # First measurement - use initial state
                        predicted_state = initial_state
                    else:
                        # Propagate from previous state
                        dt = t - times[i-1]
                        predicted_state = self._propagate_with_parameters(
                            initial_states[i-1], dt, params, times[i-1] + times[0]
                        )
                    
                    # Position and velocity residuals
                    residual = obs - predicted_state
                    
                    # Weight by measurement uncertainty (simplified)
                    weights = np.array([1000, 1000, 1000, 10, 10, 10])  # Position vs velocity weighting
                    weighted_residual = residual / weights
                    
                    residuals.extend(weighted_residual)
                
                return np.array(residuals)
                
            except Exception as e:
                self.logger.error(f"Residual computation error: {e}")
                return np.full(len(observations) * 6, 1e6)  # Large residuals for failure
        
        try:
            # Use scipy's least_squares with Levenberg-Marquardt
            optimization_result = least_squares(
                residual_function,
                initial_params,
                bounds=bounds,
                method='lm',  # Levenberg-Marquardt
                max_nfev=100,
                ftol=1e-8,
                xtol=1e-8
            )
            
            # Compute covariance matrix
            try:
                # Approximate covariance from Jacobian
                if optimization_result.jac is not None:
                    jac = optimization_result.jac
                    cov = np.linalg.inv(jac.T @ jac) * (optimization_result.cost / (len(observations) * 6 - len(initial_params)))
                else:
                    cov = np.eye(len(initial_params)) * 1e-3
            except np.linalg.LinAlgError:
                cov = np.eye(len(initial_params)) * 1e-3
            
            result = BatchEstimationResult(
                parameters=optimization_result.x,
                covariance=cov,
                residuals=optimization_result.fun,
                cost=optimization_result.cost,
                iterations=optimization_result.nfev,
                success=optimization_result.success
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Optimization error: {e}")
            return BatchEstimationResult(
                parameters=initial_params,
                covariance=np.eye(len(initial_params)) * 1e-3,
                residuals=np.array([]),
                cost=1e6,
                iterations=0,
                success=False
            )
    
    def _propagate_with_parameters(self, initial_state: np.ndarray, dt: float,
                                 params: np.ndarray, ref_time: float) -> np.ndarray:
        """Propagate state with given parameters"""
        try:
            from force_models import ForceModels
            
            # Create temporary force model with estimated parameters
            temp_config = self.config.copy()
            force_model = ForceModels(temp_config)
            
            def dynamics(t, y):
                """Orbital dynamics with estimated parameters"""
                r = y[:3]
                v = y[3:6]
                
                # Total acceleration
                accel = np.zeros(3)
                
                # Two-body gravitational acceleration
                r_norm = np.linalg.norm(r)
                mu = 3.986004418e14
                accel += -mu * r / (r_norm**3)
                
                # Perturbations with estimated parameters
                current_time = datetime.utcnow() + timedelta(seconds=ref_time + t)
                perturbations = force_model.compute_perturbations(
                    r, v, current_time, params[0], params[1]
                )
                accel += perturbations
                
                # Along-track empirical acceleration
                h = np.cross(r, v)
                r_unit = r / np.linalg.norm(r)
                along_track = np.cross(h, r)
                if np.linalg.norm(along_track) > 0:
                    along_track = along_track / np.linalg.norm(along_track)
                    accel += params[2] * along_track
                
                return np.concatenate([v, accel])
            
            # Integrate using RK45
            from scipy.integrate import solve_ivp
            
            result = solve_ivp(
                dynamics, [0, dt], initial_state,
                method='RK45', rtol=1e-8, atol=1e-10
            )
            
            return result.y[:, -1]
            
        except Exception as e:
            self.logger.error(f"Propagation error: {e}")
            return initial_state  # Return unchanged state on error
    
    def get_parameter_statistics(self) -> Dict[str, Any]:
        """Get statistics of estimated parameters over time"""
        if not self.estimation_history:
            return {}
        
        try:
            params_array = np.array([est['parameters'] for est in self.estimation_history])
            
            stats = {
                'cd_a': {
                    'mean': np.mean(params_array[:, 0]),
                    'std': np.std(params_array[:, 0]),
                    'min': np.min(params_array[:, 0]),
                    'max': np.max(params_array[:, 0]),
                    'current': params_array[-1, 0]
                },
                'cr_a': {
                    'mean': np.mean(params_array[:, 1]),
                    'std': np.std(params_array[:, 1]),
                    'min': np.min(params_array[:, 1]),
                    'max': np.max(params_array[:, 1]),
                    'current': params_array[-1, 1]
                },
                'empirical': {
                    'mean': np.mean(params_array[:, 2]),
                    'std': np.std(params_array[:, 2]),
                    'min': np.min(params_array[:, 2]),
                    'max': np.max(params_array[:, 2]),
                    'current': params_array[-1, 2]
                },
                'num_estimations': len(self.estimation_history)
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Parameter statistics error: {e}")
            return {}
    
    def is_parameter_stable(self, param_name: str, stability_threshold: float = 0.1) -> bool:
        """Check if parameter estimates are stable over recent estimations"""
        if len(self.estimation_history) < 5:
            return False
        
        try:
            param_index = {'cd_a': 0, 'cr_a': 1, 'empirical': 2}[param_name]
            recent_params = [est['parameters'][param_index] for est in self.estimation_history[-5:]]
            
            # Coefficient of variation
            mean_val = np.mean(recent_params)
            std_val = np.std(recent_params)
            
            if mean_val != 0:
                cv = std_val / abs(mean_val)
                return cv < stability_threshold
            else:
                return std_val < stability_threshold
                
        except Exception as e:
            self.logger.error(f"Stability check error: {e}")
            return False

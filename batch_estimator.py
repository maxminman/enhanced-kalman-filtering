import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from scipy.optimize import least_squares
from scipy.stats import chi2
from dataclasses import dataclass
import math

@dataclass
class BatchEstimationResult:
    """Results from batch parameter estimation"""
    parameters: np.ndarray
    covariance: np.ndarray
    residuals: np.ndarray
    cost: float
    iterations: int
    success: bool
    parameter_correlations: np.ndarray
    parameter_uncertainties: np.ndarray
    outlier_indices: List[int]
    convergence_metrics: Dict[str, float]
    quality_score: float

@dataclass
class ParameterQuality:
    """Parameter estimation quality metrics"""
    convergence_achieved: bool
    parameter_stability: float
    correlation_strength: float
    outlier_percentage: float
    estimation_confidence: float
    recommended_window_size: int

class EnhancedBatchEstimator:
    """
    Enhanced sliding-window batch parameter estimation with multi-parameter correlation handling,
    robust outlier rejection, and adaptive window sizing for sub-1km accuracy
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize enhanced batch estimator"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Estimation parameters
        self.base_window_size = config.get('batch_window_size', 50)
        self.min_window_size = config.get('min_window_size', 20)
        self.max_window_size = config.get('max_window_size', 100)
        self.overlap_ratio = config.get('batch_overlap', 0.5)
        
        # Adaptive window sizing
        self.adaptive_window_sizing = config.get('adaptive_window_sizing', True)
        self.window_size_history = []
        
        # Parameter bounds (enhanced)
        self.param_bounds = {
            'bc': (0.0005, 0.015),      # Extended ballistic coefficient bounds
            'cr': (0.3, 3.0),           # SRP coefficient bounds
            'empirical_along': (-2e-5, 2e-5),    # Along-track acceleration
            'empirical_cross': (-1e-5, 1e-5),    # Cross-track acceleration
            'empirical_radial': (-5e-6, 5e-6)    # Radial acceleration
        }
        
        # Robust estimation parameters
        self.outlier_threshold = config.get('outlier_threshold', 3.0)  # sigma
        self.max_outlier_percentage = config.get('max_outlier_percentage', 0.2)
        self.robust_estimation = config.get('robust_estimation', True)
        
        # Convergence criteria
        self.convergence_tolerance = config.get('convergence_tolerance', 1e-6)
        self.max_iterations = config.get('max_iterations', 100)
        self.parameter_change_threshold = config.get('parameter_change_threshold', 1e-4)
        
        # Correlation handling
        self.correlation_threshold = config.get('correlation_threshold', 0.8)
        self.handle_correlations = config.get('handle_correlations', True)
        
        # Estimation history and quality tracking
        self.estimation_history = []
        self.parameter_stability_history = []
        self.quality_metrics_history = []
        
        # Real-time adaptation parameters
        self.real_time_adaptation = config.get('real_time_adaptation', True)
        self.adaptation_rate = config.get('adaptation_rate', 0.1)
        self.stability_threshold = config.get('stability_threshold', 0.1)
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        
        # Current adapted parameters
        self.current_parameters = np.array([
            config.get('ballistic_coeff', 0.0055),
            config.get('srp_coeff', 1.3),
            0.0  # empirical acceleration
        ])
        self.parameter_uncertainties = np.ones(3) * 0.1
        self.last_adaptation_time = datetime.utcnow()
        
        self.logger.info("Enhanced batch parameter estimator initialized")
    
    def estimate_parameters(self, measurement_history: List[Dict[str, Any]], 
                          state_history: List[np.ndarray],
                          force_models: Any = None) -> BatchEstimationResult:
        """
        Enhanced parameter estimation with multi-parameter correlation handling
        and robust outlier rejection
        
        Args:
            measurement_history: List of measurement dictionaries
            state_history: List of 
            measurement_history: List of measurement dictionaries
            state_history: List of state vectors
            
        Returns:
            Estimated parameters [Bc, Cr, along_track_accel]
        """
        try:
            if len(measurement_history) < self.window_size:
                self.logger.debug("Insufficient data for batch estimation")
                return np.array([self.config.get('ballistic_coeff', 0.00540),
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
                self.config.get('ballistic_coeff', 0.00540),
                self.config.get('srp_coeff', 1.3),
                0.0  # Along-track acceleration
            ])
            
            # Parameter bounds for optimization
            bounds = (
                [self.param_bounds['bc'][0], self.param_bounds['cr_a'][0], 
                 self.param_bounds['empirical'][0]],
                [self.param_bounds['bc'][1], self.param_bounds['cr_a'][1], 
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
                
                return result
            else:
                self.logger.warning("Batch parameter estimation failed to converge")
                return BatchEstimationResult(
                    parameters=initial_params,
                    covariance=np.eye(len(initial_params)) * 1e-6,
                    residuals=np.array([]),
                    cost=np.inf,
                    iterations=0,
                    success=False,
                    parameter_correlations=np.eye(len(initial_params)),
                    parameter_uncertainties=np.ones(len(initial_params)) * 1e-3,
                    outlier_indices=[],
                    convergence_metrics={},
                    quality_score=0.0
                )
                
        except Exception as e:
            self.logger.error(f"Batch parameter estimation failed: {e}")
            return BatchEstimationResult(
                parameters=np.array([0.0055, 1.3, 0.0]),
                covariance=np.eye(3) * 1e-6,
                residuals=np.array([]),
                cost=np.inf,
                iterations=0,
                success=False,
                parameter_correlations=np.eye(3),
                parameter_uncertainties=np.ones(3) * 1e-3,
                outlier_indices=[],
                convergence_metrics={},
                quality_score=0.0
            )
    
    def _prepare_estimation_data(self, measurements: List[Dict[str, Any]], 
                               states: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for parameter estimation"""
        try:
            times = []
            observations = []
            initial_states = []
            
            for i, (meas, state) in enumerate(zip(measurements, states)):
                if 'timestamp' in meas and 'measurement' in meas:
                    times.append(i)  # Use index as time for simplicity
                    observations.append(meas['measurement'][:6])  # Position and velocity
                    initial_states.append(state[:6])
            
            return np.array(times), np.array(observations), np.array(initial_states)
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            return np.array([]), np.array([]), np.array([])
    
    def _weighted_least_squares(self, initial_params: np.ndarray, times: np.ndarray,
                              observations: np.ndarray, initial_states: np.ndarray,
                              bounds: Tuple) -> BatchEstimationResult:
        """Enhanced weighted least squares with robust outlier rejection"""
        try:
            def residual_function(params):
                """Compute residuals for parameter estimation"""
                residuals = []
                
                for i, (obs, state) in enumerate(zip(observations, initial_states)):
                    # Propagate state with current parameters
                    # This is a simplified version - in practice would use full force models
                    predicted_state = state  # Placeholder
                    
                    # Compute residual
                    residual = obs - predicted_state
                    residuals.extend(residual)
                
                return np.array(residuals)
            
            # Initial estimation without outlier rejection
            result = least_squares(
                residual_function, 
                initial_params,
                bounds=bounds,
                method='trf',
                max_nfev=self.max_iterations,
                ftol=self.convergence_tolerance,
                xtol=self.convergence_tolerance
            )
            
            if not result.success:
                return self._create_failed_result(initial_params)
            
            # Compute covariance matrix
            try:
                # Approximate Hessian from Jacobian
                J = result.jac
                covariance = np.linalg.inv(J.T @ J) * (result.cost / (len(observations) - len(initial_params)))
            except:
                covariance = np.eye(len(initial_params)) * 1e-6
            
            # Robust outlier detection and rejection
            residuals = result.fun
            outlier_indices = self._detect_outliers(residuals)
            
            # Re-estimate without outliers if necessary
            if len(outlier_indices) > 0 and len(outlier_indices) / len(residuals) < self.max_outlier_percentage:
                # Create mask for non-outliers
                mask = np.ones(len(observations), dtype=bool)
                mask[outlier_indices] = False
                
                # Re-run estimation without outliers
                filtered_observations = observations[mask]
                filtered_states = initial_states[mask]
                
                def filtered_residual_function(params):
                    residuals = []
                    for obs, state in zip(filtered_observations, filtered_states):
                        predicted_state = state  # Placeholder
                        residual = obs - predicted_state
                        residuals.extend(residual)
                    return np.array(residuals)
                
                refined_result = least_squares(
                    filtered_residual_function,
                    result.x,
                    bounds=bounds,
                    method='trf',
                    max_nfev=self.max_iterations,
                    ftol=self.convergence_tolerance
                )
                
                if refined_result.success:
                    result = refined_result
            
            # Compute parameter correlations
            correlations = self._compute_parameter_correlations(covariance)
            
            # Compute parameter uncertainties (standard deviations)
            uncertainties = np.sqrt(np.diag(covariance))
            
            # Compute convergence metrics
            convergence_metrics = {
                'cost_reduction': result.cost,
                'gradient_norm': np.linalg.norm(result.grad) if hasattr(result, 'grad') else 0.0,
                'parameter_change': np.linalg.norm(result.x - initial_params),
                'iterations': result.nfev
            }
            
            # Compute quality score
            quality_score = self._compute_quality_score(
                result, covariance, outlier_indices, convergence_metrics
            )
            
            return BatchEstimationResult(
                parameters=result.x,
                covariance=covariance,
                residuals=result.fun,
                cost=result.cost,
                iterations=result.nfev,
                success=result.success,
                parameter_correlations=correlations,
                parameter_uncertainties=uncertainties,
                outlier_indices=outlier_indices,
                convergence_metrics=convergence_metrics,
                quality_score=quality_score
            )
            
        except Exception as e:
            self.logger.error(f"Weighted least squares estimation failed: {e}")
            return self._create_failed_result(initial_params)
    
    def _detect_outliers(self, residuals: np.ndarray) -> List[int]:
        """Detect outliers using robust statistical methods"""
        try:
            if len(residuals) == 0:
                return []
            
            # Reshape residuals to per-observation
            n_obs = len(residuals) // 6  # 6 components per observation
            if n_obs == 0:
                return []
            
            obs_residuals = residuals.reshape(n_obs, 6)
            
            # Compute residual norms for each observation
            residual_norms = np.linalg.norm(obs_residuals, axis=1)
            
            # Robust outlier detection using modified Z-score
            median = np.median(residual_norms)
            mad = np.median(np.abs(residual_norms - median))  # Median Absolute Deviation
            
            if mad == 0:
                return []
            
            # Modified Z-score
            modified_z_scores = 0.6745 * (residual_norms - median) / mad
            
            # Identify outliers
            outlier_indices = np.where(np.abs(modified_z_scores) > self.outlier_threshold)[0]
            
            return outlier_indices.tolist()
            
        except Exception as e:
            self.logger.warning(f"Outlier detection failed: {e}")
            return []
    
    def _compute_parameter_correlations(self, covariance: np.ndarray) -> np.ndarray:
        """Compute parameter correlation matrix"""
        try:
            # Convert covariance to correlation
            std_devs = np.sqrt(np.diag(covariance))
            correlations = covariance / np.outer(std_devs, std_devs)
            
            # Handle numerical issues
            correlations = np.clip(correlations, -1.0, 1.0)
            np.fill_diagonal(correlations, 1.0)
            
            return correlations
            
        except Exception as e:
            self.logger.warning(f"Correlation computation failed: {e}")
            return np.eye(covariance.shape[0])
    
    def _compute_quality_score(self, result, covariance: np.ndarray, 
                             outlier_indices: List[int], 
                             convergence_metrics: Dict[str, float]) -> float:
        """Compute overall estimation quality score (0.0 to 1.0)"""
        try:
            # Convergence quality
            convergence_score = 1.0 if result.success else 0.0
            if result.success:
                # Penalize high gradient norm (poor convergence)
                grad_norm = convergence_metrics.get('gradient_norm', 0.0)
                convergence_score *= 1.0 / (1.0 + grad_norm * 1000)
            
            # Parameter uncertainty quality
            uncertainties = np.sqrt(np.diag(covariance))
            relative_uncertainties = uncertainties / (np.abs(result.x) + 1e-6)
            uncertainty_score = 1.0 / (1.0 + np.mean(relative_uncertainties))
            
            # Outlier quality
            outlier_percentage = len(outlier_indices) / max(1, len(result.fun) // 6)
            outlier_score = 1.0 - min(1.0, outlier_percentage / self.max_outlier_percentage)
            
            # Cost quality (lower is better)
            cost_score = 1.0 / (1.0 + result.cost / 1000.0)
            
            # Weighted combination
            quality_score = (0.3 * convergence_score + 
                           0.3 * uncertainty_score + 
                           0.2 * outlier_score + 
                           0.2 * cost_score)
            
            return np.clip(quality_score, 0.0, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Quality score computation failed: {e}")
            return 0.5
    
    def _create_failed_result(self, initial_params: np.ndarray) -> BatchEstimationResult:
        """Create a failed estimation result"""
        return BatchEstimationResult(
            parameters=initial_params,
            covariance=np.eye(len(initial_params)) * 1e-6,
            residuals=np.array([]),
            cost=np.inf,
            iterations=0,
            success=False,
            parameter_correlations=np.eye(len(initial_params)),
            parameter_uncertainties=np.ones(len(initial_params)) * 1e-3,
            outlier_indices=[],
            convergence_metrics={},
            quality_score=0.0
        )
    
    def assess_parameter_quality(self, estimation_result: BatchEstimationResult) -> ParameterQuality:
        """Assess parameter estimation quality and recommend window size"""
        try:
            # Convergence assessment
            convergence_achieved = (estimation_result.success and 
                                  estimation_result.convergence_metrics.get('gradient_norm', np.inf) < 1e-3)
            
            # Parameter stability (based on uncertainties)
            relative_uncertainties = (estimation_result.parameter_uncertainties / 
                                    (np.abs(estimation_result.parameters) + 1e-6))
            parameter_stability = 1.0 / (1.0 + np.mean(relative_uncertainties))
            
            # Correlation strength (maximum off-diagonal correlation)
            correlations = estimation_result.parameter_correlations
            max_correlation = np.max(np.abs(correlations - np.eye(correlations.shape[0])))
            correlation_strength = max_correlation
            
            # Outlier percentage
            outlier_percentage = len(estimation_result.outlier_indices) / max(1, len(estimation_result.residuals) // 6)
            
            # Estimation confidence
            estimation_confidence = estimation_result.quality_score
            
            # Recommend window size based on quality
            if estimation_confidence > 0.8:
                recommended_window_size = max(self.min_window_size, self.base_window_size - 10)
            elif estimation_confidence > 0.6:
                recommended_window_size = self.base_window_size
            else:
                recommended_window_size = min(self.max_window_size, self.base_window_size + 20)
            
            return ParameterQuality(
                convergence_achieved=convergence_achieved,
                parameter_stability=parameter_stability,
                correlation_strength=correlation_strength,
                outlier_percentage=outlier_percentage,
                estimation_confidence=estimation_confidence,
                recommended_window_size=recommended_window_size
            )
            
        except Exception as e:
            self.logger.error(f"Parameter quality assessment failed: {e}")
            return ParameterQuality(
                convergence_achieved=False,
                parameter_stability=0.0,
                correlation_strength=1.0,
                outlier_percentage=1.0,
                estimation_confidence=0.0,
                recommended_window_size=self.base_window_size
            )
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics for monitoring"""
        try:
            diagnostics = {
                'estimation_history_length': len(self.estimation_history),
                'current_window_size': self.base_window_size,
                'adaptive_window_sizing': self.adaptive_window_sizing,
                'robust_estimation': self.robust_estimation,
                'parameter_bounds': self.param_bounds
            }
            
            if self.estimation_history:
                latest = self.estimation_history[-1]
                diagnostics.update({
                    'latest_parameters': latest['parameters'].tolist(),
                    'latest_cost': latest['cost'],
                    'latest_window_size': latest['window_size']
                })
                
                # Parameter stability over time
                if len(self.estimation_history) >= 2:
                    recent_params = [est['parameters'] for est in self.estimation_history[-5:]]
                    param_std = np.std(recent_params, axis=0)
                    diagnostics['parameter_stability'] = param_std.tolist()
            
            return diagnostics
            
        except Exception as e:
            self.logger.error(f"Diagnostics generation failed: {e}")
            return {'error': str(e)}
    
    def update_parameters_realtime(self, estimation_result: BatchEstimationResult) -> np.ndarray:
        """
        Real-time parameter adaptation with sliding window approach
        
        Args:
            estimation_result: Latest batch estimation result
            
        Returns:
            Updated parameters for real-time use
        """
        try:
            if not self.real_time_adaptation or not estimation_result.success:
                return self.current_parameters
            
            # Assess parameter quality
            quality = self.assess_parameter_quality(estimation_result)
            
            # Only adapt if quality is sufficient
            if quality.estimation_confidence < self.confidence_threshold:
                self.logger.debug(f"Parameter quality too low for adaptation: {quality.estimation_confidence:.3f}")
                return self.current_parameters
            
            # Compute adaptation weights based on quality and stability
            adaptation_weight = self._compute_adaptation_weight(quality, estimation_result)
            
            # Adaptive parameter update with rate limiting
            new_parameters = self._adaptive_parameter_update(
                estimation_result.parameters, adaptation_weight
            )
            
            # Update parameter uncertainties
            self._update_parameter_uncertainties(estimation_result, quality)
            
            # Apply parameter bounds
            new_parameters = self._apply_parameter_bounds(new_parameters)
            
            # Update current parameters
            old_parameters = self.current_parameters.copy()
            self.current_parameters = new_parameters
            self.last_adaptation_time = datetime.utcnow()
            
            # Log significant changes
            param_change = np.linalg.norm(new_parameters - old_parameters)
            if param_change > 0.01:  # 1% change threshold
                self.logger.info(f"Real-time parameter adaptation: "
                               f"Bc={new_parameters[0]:.6f}, Cr={new_parameters[1]:.3f}, "
                               f"Emp={new_parameters[2]:.2e}, change={param_change:.4f}")
            
            # Store adaptation history
            self.parameter_stability_history.append({
                'timestamp': datetime.utcnow(),
                'parameters': new_parameters.copy(),
                'quality': quality.estimation_confidence,
                'stability': quality.parameter_stability,
                'adaptation_weight': adaptation_weight
            })
            
            # Keep limited history
            if len(self.parameter_stability_history) > 200:
                self.parameter_stability_history = self.parameter_stability_history[-100:]
            
            return self.current_parameters
            
        except Exception as e:
            self.logger.error(f"Real-time parameter adaptation failed: {e}")
            return self.current_parameters
    
    def _compute_adaptation_weight(self, quality: ParameterQuality, 
                                 estimation_result: BatchEstimationResult) -> float:
        """Compute adaptation weight based on quality metrics"""
        try:
            # Base weight from estimation confidence
            base_weight = quality.estimation_confidence
            
            # Reduce weight for high parameter correlations
            correlation_penalty = min(1.0, quality.correlation_strength / self.correlation_threshold)
            correlation_weight = 1.0 - correlation_penalty * 0.5
            
            # Reduce weight for high outlier percentage
            outlier_penalty = min(1.0, quality.outlier_percentage / self.max_outlier_percentage)
            outlier_weight = 1.0 - outlier_penalty * 0.3
            
            # Reduce weight if parameters are changing too rapidly
            stability_weight = min(1.0, quality.parameter_stability / self.stability_threshold)
            
            # Time-based weight (reduce adaptation frequency)
            time_since_last = (datetime.utcnow() - self.last_adaptation_time).total_seconds()
            min_adaptation_interval = 300.0  # 5 minutes
            time_weight = min(1.0, time_since_last / min_adaptation_interval)
            
            # Combined weight
            adaptation_weight = (base_weight * correlation_weight * outlier_weight * 
                               stability_weight * time_weight * self.adaptation_rate)
            
            return np.clip(adaptation_weight, 0.0, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Adaptation weight computation failed: {e}")
            return 0.1  # Conservative default
    
    def _adaptive_parameter_update(self, new_estimates: np.ndarray, 
                                 adaptation_weight: float) -> np.ndarray:
        """Perform adaptive parameter update with rate limiting"""
        try:
            # Exponential moving average update
            updated_params = ((1.0 - adaptation_weight) * self.current_parameters + 
                            adaptation_weight * new_estimates)
            
            # Apply maximum change limits per update
            max_change_per_update = {
                0: 0.001,   # Bc: max 0.001 change per update
                1: 0.1,     # Cr: max 0.1 change per update  
                2: 1e-6     # Empirical: max 1e-6 change per update
            }
            
            for i in range(len(updated_params)):
                change = updated_params[i] - self.current_parameters[i]
                max_change = max_change_per_update.get(i, 0.1)
                
                if abs(change) > max_change:
                    change = np.sign(change) * max_change
                    updated_params[i] = self.current_parameters[i] + change
            
            return updated_params
            
        except Exception as e:
            self.logger.error(f"Adaptive parameter update failed: {e}")
            return self.current_parameters
    
    def _update_parameter_uncertainties(self, estimation_result: BatchEstimationResult,
                                      quality: ParameterQuality):
        """Update parameter uncertainty estimates"""
        try:
            # Use estimation uncertainties if available and reliable
            if quality.estimation_confidence > 0.5:
                # Blend estimated uncertainties with current uncertainties
                blend_weight = quality.estimation_confidence * 0.3
                self.parameter_uncertainties = ((1.0 - blend_weight) * self.parameter_uncertainties + 
                                              blend_weight * estimation_result.parameter_uncertainties)
            else:
                # Increase uncertainties if estimation quality is poor
                self.parameter_uncertainties *= 1.1
            
            # Apply bounds to uncertainties
            min_uncertainties = np.array([1e-5, 0.01, 1e-8])  # Minimum uncertainties
            max_uncertainties = np.array([0.01, 1.0, 1e-5])   # Maximum uncertainties
            
            self.parameter_uncertainties = np.clip(
                self.parameter_uncertainties, min_uncertainties, max_uncertainties
            )
            
        except Exception as e:
            self.logger.warning(f"Parameter uncertainty update failed: {e}")
    
    def _apply_parameter_bounds(self, parameters: np.ndarray) -> np.ndarray:
        """Apply parameter bounds with satellite-specific adjustments"""
        try:
            bounded_params = parameters.copy()
            
            # Ballistic coefficient bounds
            bounded_params[0] = np.clip(bounded_params[0], 
                                      self.param_bounds['bc'][0], 
                                      self.param_bounds['bc'][1])
            
            # SRP coefficient bounds  
            bounded_params[1] = np.clip(bounded_params[1],
                                      self.param_bounds['cr'][0],
                                      self.param_bounds['cr'][1])
            
            # Empirical acceleration bounds
            bounded_params[2] = np.clip(bounded_params[2],
                                      self.param_bounds['empirical_along'][0],
                                      self.param_bounds['empirical_along'][1])
            
            return bounded_params
            
        except Exception as e:
            self.logger.error(f"Parameter bounds application failed: {e}")
            return parameters
    
    def get_current_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current adapted parameters and their uncertainties
        
        Returns:
            Tuple of (parameters, uncertainties)
        """
        return self.current_parameters.copy(), self.parameter_uncertainties.copy()
    
    def monitor_parameter_stability(self) -> Dict[str, Any]:
        """Monitor parameter stability over time"""
        try:
            if len(self.parameter_stability_history) < 2:
                return {'status': 'insufficient_data'}
            
            # Analyze recent parameter evolution
            recent_history = self.parameter_stability_history[-20:]  # Last 20 updates
            
            # Compute parameter trends
            timestamps = [h['timestamp'] for h in recent_history]
            parameters = np.array([h['parameters'] for h in recent_history])
            
            # Parameter stability metrics
            param_std = np.std(parameters, axis=0)
            param_mean = np.mean(parameters, axis=0)
            relative_std = param_std / (np.abs(param_mean) + 1e-6)
            
            # Quality trend
            qualities = [h['quality'] for h in recent_history]
            quality_trend = np.polyfit(range(len(qualities)), qualities, 1)[0] if len(qualities) > 1 else 0.0
            
            # Adaptation frequency
            time_span = (timestamps[-1] - timestamps[0]).total_seconds()
            adaptation_frequency = len(recent_history) / max(time_span / 3600.0, 1e-6)  # per hour
            
            stability_metrics = {
                'parameter_stability': {
                    'bc_relative_std': relative_std[0],
                    'cr_relative_std': relative_std[1], 
                    'empirical_relative_std': relative_std[2],
                    'overall_stability': 1.0 / (1.0 + np.mean(relative_std))
                },
                'quality_trend': quality_trend,
                'adaptation_frequency_per_hour': adaptation_frequency,
                'current_parameters': self.current_parameters.tolist(),
                'current_uncertainties': self.parameter_uncertainties.tolist(),
                'last_adaptation': self.last_adaptation_time.isoformat(),
                'status': 'stable' if np.mean(relative_std) < 0.1 else 'unstable'
            }
            
            return stability_metrics
            
        except Exception as e:
            self.logger.error(f"Parameter stability monitoring failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
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
                method='trf',  # Trust Region Reflective (supports bounds)
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
                'bc': {
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
            param_index = {'bc': 0, 'cr_a': 1, 'empirical': 2}[param_name]
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

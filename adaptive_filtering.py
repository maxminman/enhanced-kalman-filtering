import numpy as np
from typing import Dict, Any, List, Optional
import logging
from collections import deque

class AdaptiveFiltering:
    """
    Adaptive process and measurement noise estimation with innovation-based
    covariance matching and divergence detection
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize adaptive filtering"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Innovation monitoring
        self.innovation_window = config.get('innovation_window', 20)
        self.innovation_history = deque(maxlen=self.innovation_window)
        self.innovation_cov_history = deque(maxlen=self.innovation_window)
        
        # Adaptation parameters
        self.min_adaptation_samples = config.get('min_adaptation_samples', 10)
        self.adaptation_factor = config.get('adaptation_factor', 0.1)
        self.max_inflation_factor = config.get('max_inflation_factor', 10.0)
        
        # Process noise adaptation
        self.base_process_noise = config.get('process_noise_scale', 1.0)
        self.current_process_scale = 1.0
        
        # Measurement noise adaptation
        self.base_measurement_noise = config.get('measurement_noise_scale', 1.0)
        self.current_measurement_scale = 1.0
        
        # Divergence detection
        self.divergence_threshold = config.get('divergence_threshold', 25.0)  # Chi-square for 6 DOF
        self.divergence_count = 0
        self.max_divergence_count = config.get('max_divergence_count', 3)
        
        # Covariance inflation
        self.inflation_factor = 1.0
        self.min_inflation = 1.0
        self.max_inflation = config.get('max_covariance_inflation', 5.0)
        
        self.logger.info("Adaptive filtering initialized")
    
    def update_innovation_statistics(self, innovation: np.ndarray, 
                                   innovation_covariance: np.ndarray):
        """Update innovation statistics for adaptation"""
        try:
            # Store innovation and covariance
            self.innovation_history.append(innovation.copy())
            self.innovation_cov_history.append(innovation_covariance.copy())
            
            # Compute normalized innovation squared (NIS)
            try:
                nis = innovation.T @ np.linalg.inv(innovation_covariance) @ innovation
            except np.linalg.LinAlgError:
                nis = innovation.T @ np.linalg.pinv(innovation_covariance) @ innovation
            
            # Check for divergence
            if nis > self.divergence_threshold:
                self.divergence_count += 1
                self.logger.warning(f"Innovation divergence detected: NIS = {nis:.2f}")
            else:
                self.divergence_count = max(0, self.divergence_count - 1)
            
            return nis
            
        except Exception as e:
            self.logger.error(f"Innovation statistics update error: {e}")
            return 0.0
    
    def adapt_process_noise(self, covariance: np.ndarray, 
                           innovation_history: List[float]) -> Optional[np.ndarray]:
        """
        Adapt process noise based on innovation consistency
        
        Args:
            covariance: Current state covariance matrix
            innovation_history: Recent innovation norms
            
        Returns:
            Additional process noise matrix or None
        """
        try:
            if len(self.innovation_history) < self.min_adaptation_samples:
                return None
            
            # Compute innovation statistics
            recent_innovations = list(self.innovation_history)[-self.min_adaptation_samples:]
            recent_covariances = list(self.innovation_cov_history)[-self.min_adaptation_samples:]
            
            # Theoretical vs observed innovation covariance
            theoretical_trace = np.mean([np.trace(S) for S in recent_covariances])
            observed_trace = np.mean([
                np.linalg.norm(innov)**2 for innov in recent_innovations
            ])
            
            # Adaptation ratio
            if theoretical_trace > 0:
                adaptation_ratio = observed_trace / theoretical_trace
            else:
                adaptation_ratio = 1.0
            
            # Update process noise scale
            if adaptation_ratio > 1.5:  # Innovations too large
                self.current_process_scale *= (1 + self.adaptation_factor)
            elif adaptation_ratio < 0.5:  # Innovations too small
                self.current_process_scale *= (1 - self.adaptation_factor)
            
            # Bound the scaling
            self.current_process_scale = np.clip(
                self.current_process_scale, 0.1, self.max_inflation_factor
            )
            
            # Generate additional process noise
            state_dim = covariance.shape[0]
            Q_adaptive = np.eye(state_dim) * (self.current_process_scale - 1.0) * 1e-12
            
            return Q_adaptive
            
        except Exception as e:
            self.logger.error(f"Process noise adaptation error: {e}")
            return None
    
    def adapt_measurement_noise(self, innovation: np.ndarray,
                              innovation_covariance: np.ndarray,
                              filter_covariance: np.ndarray) -> Optional[np.ndarray]:
        """
        Adapt measurement noise based on innovation consistency
        
        Args:
            innovation: Current innovation vector
            innovation_covariance: Innovation covariance matrix
            filter_covariance: Current filter covariance
            
        Returns:
            Adapted measurement noise matrix or None
        """
        try:
            if len(self.innovation_history) < self.min_adaptation_samples:
                return None
            
            # Compute sample innovation covariance
            innovations_matrix = np.array(list(self.innovation_history))
            if innovations_matrix.shape[0] < 2:
                return None
            
            sample_innovation_cov = np.cov(innovations_matrix.T)
            
            # Compare with theoretical innovation covariance
            theoretical_cov = innovation_covariance
            
            # Compute scaling factor
            try:
                scaling_matrix = np.linalg.solve(theoretical_cov, sample_innovation_cov)
                scaling_factor = np.trace(scaling_matrix) / theoretical_cov.shape[0]
            except np.linalg.LinAlgError:
                scaling_factor = 1.0
            
            # Update measurement noise scale
            if scaling_factor > 1.2:
                self.current_measurement_scale *= (1 + self.adaptation_factor)
            elif scaling_factor < 0.8:
                self.current_measurement_scale *= (1 - self.adaptation_factor)
            
            # Bound the scaling
            self.current_measurement_scale = np.clip(
                self.current_measurement_scale, 0.1, self.max_inflation_factor
            )
            
            # Generate adapted measurement noise
            R_adapted = theoretical_cov * self.current_measurement_scale
            
            return R_adapted
            
        except Exception as e:
            self.logger.error(f"Measurement noise adaptation error: {e}")
            return None
    
    def detect_divergence(self) -> bool:
        """Detect filter divergence based on innovation statistics"""
        try:
            if len(self.innovation_history) < 5:
                return False
            
            # Check consecutive large innovations
            recent_nis = []
            for innov, cov in zip(list(self.innovation_history)[-5:], 
                                list(self.innovation_cov_history)[-5:]):
                try:
                    nis = innov.T @ np.linalg.inv(cov) @ innov
                    recent_nis.append(nis)
                except np.linalg.LinAlgError:
                    continue
            
            if len(recent_nis) >= 3:
                large_innovations = sum(1 for nis in recent_nis if nis > self.divergence_threshold)
                divergence_detected = large_innovations >= 3
                
                if divergence_detected:
                    self.logger.warning("Filter divergence detected from innovation analysis")
                
                return divergence_detected
            
            return False
            
        except Exception as e:
            self.logger.error(f"Divergence detection error: {e}")
            return False
    
    def inflate_covariance(self, covariance: np.ndarray, 
                          inflation_factor: Optional[float] = None) -> np.ndarray:
        """
        Inflate covariance matrix to prevent filter overconfidence
        
        Args:
            covariance: Current covariance matrix
            inflation_factor: Optional inflation factor override
            
        Returns:
            Inflated covariance matrix
        """
        try:
            if inflation_factor is None:
                # Adaptive inflation based on divergence count
                if self.divergence_count > 0:
                    self.inflation_factor = min(
                        self.inflation_factor * (1 + 0.1 * self.divergence_count),
                        self.max_inflation
                    )
                else:
                    # Gradually reduce inflation if no divergence
                    self.inflation_factor = max(
                        self.inflation_factor * 0.99,
                        self.min_inflation
                    )
                
                inflation_factor = self.inflation_factor
            
            # Apply inflation
            inflated_cov = covariance * inflation_factor
            
            return inflated_cov
            
        except Exception as e:
            self.logger.error(f"Covariance inflation error: {e}")
            return covariance
    
    def reset_adaptation(self):
        """Reset adaptive parameters to initial values"""
        try:
            self.current_process_scale = 1.0
            self.current_measurement_scale = 1.0
            self.inflation_factor = 1.0
            self.divergence_count = 0
            self.innovation_history.clear()
            self.innovation_cov_history.clear()
            
            self.logger.info("Adaptive filtering parameters reset")
            
        except Exception as e:
            self.logger.error(f"Adaptation reset error: {e}")
    
    def get_adaptation_status(self) -> Dict[str, Any]:
        """Get current adaptation status"""
        try:
            status = {
                'process_noise_scale': self.current_process_scale,
                'measurement_noise_scale': self.current_measurement_scale,
                'covariance_inflation': self.inflation_factor,
                'divergence_count': self.divergence_count,
                'innovation_samples': len(self.innovation_history),
                'adaptation_active': len(self.innovation_history) >= self.min_adaptation_samples
            }
            
            if len(self.innovation_history) > 0:
                # Recent innovation statistics
                recent_innovations = list(self.innovation_history)[-5:]
                recent_norms = [np.linalg.norm(innov) for innov in recent_innovations]
                
                status.update({
                    'recent_innovation_mean': np.mean(recent_norms),
                    'recent_innovation_std': np.std(recent_norms),
                    'recent_innovation_max': np.max(recent_norms)
                })
            
            return status
            
        except Exception as e:
            self.logger.error(f"Adaptation status error: {e}")
            return {}
    
    def apply_adaptive_corrections(self, state: np.ndarray, covariance: np.ndarray,
                                 innovation: np.ndarray, innovation_cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply all adaptive corrections to state and covariance
        
        Args:
            state: Current state estimate
            covariance: Current covariance matrix
            innovation: Current innovation
            innovation_cov: Innovation covariance
            
        Returns:
            Tuple of (corrected_state, corrected_covariance)
        """
        try:
            # Update innovation statistics
            self.update_innovation_statistics(innovation, innovation_cov)
            
            # Apply covariance inflation if needed
            corrected_covariance = covariance
            if self.detect_divergence():
                corrected_covariance = self.inflate_covariance(corrected_covariance, 2.0)
                self.logger.warning("Applied divergence correction - covariance inflated")
            else:
                corrected_covariance = self.inflate_covariance(corrected_covariance)
            
            # State remains unchanged in this implementation
            corrected_state = state.copy()
            
            return corrected_state, corrected_covariance
            
        except Exception as e:
            self.logger.error(f"Adaptive corrections error: {e}")
            return state, covariance

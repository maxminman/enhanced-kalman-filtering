#!/usr/bin/env python3
"""
Adaptive Filter Tuner for Enhanced EKF
Implements innovation-based adaptive tuning of process and measurement noise matrices
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from collections import deque
import math

@dataclass
class InnovationStatistics:
    """Statistics computed from innovation sequences"""
    mean: np.ndarray
    covariance: np.ndarray
    normalized_innovation_squared: float
    consistency_ratio: float
    bias_indicator: float
    quality_metric: float
    sample_count: int
    time_span: float  # seconds

@dataclass
class AdaptationConfig:
    """Configuration for adaptive filter tuning"""
    # Innovation window parameters
    innovation_window_size: int = 50
    min_samples_for_adaptation: int = 10
    max_adaptation_rate: float = 0.1
    
    # Bias detection thresholds
    bias_detection_threshold: float = 3.0  # sigma
    consistency_threshold_low: float = 0.5
    consistency_threshold_high: float = 2.0
    
    # Adaptation rates
    process_noise_adaptation_rate: float = 0.05
    measurement_noise_adaptation_rate: float = 0.1
    
    # TLE quality parameters
    tle_age_threshold_hours: float = 24.0
    tle_quality_degradation_rate: float = 0.1  # per hour
    
    # Bounds for noise scaling
    min_process_noise_scale: float = 0.1
    max_process_noise_scale: float = 10.0
    min_measurement_noise_scale: float = 0.5
    max_measurement_noise_scale: float = 20.0

class AdaptiveFilterTuner:
    """
    Adaptive filter tuner that adjusts process and measurement noise matrices
    based on innovation statistics and TLE quality assessment
    """
    
    def __init__(self, config: Optional[AdaptationConfig] = None):
        """Initialize adaptive filter tuner"""
        self.config = config or AdaptationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Innovation history storage
        self.innovation_history = deque(maxlen=self.config.innovation_window_size)
        self.innovation_covariance_history = deque(maxlen=self.config.innovation_window_size)
        self.timestamp_history = deque(maxlen=self.config.innovation_window_size)
        
        # Current adaptation factors
        self.process_noise_scale = 1.0
        self.measurement_noise_scale = 1.0
        
        # Base noise matrices (will be set by EKF)
        self.base_process_noise = None
        self.base_measurement_noise = None
        
        # Statistics tracking
        self.last_statistics = None
        self.adaptation_history = []
        
        # TLE quality tracking
        self.last_tle_epoch = None
        self.tle_quality_factor = 1.0
        
        self.logger.info("Adaptive filter tuner initialized")
    
    def set_base_noise_matrices(self, Q_base: np.ndarray, R_base: np.ndarray):
        """Set base process and measurement noise matrices"""
        self.base_process_noise = Q_base.copy()
        self.base_measurement_noise = R_base.copy()
        self.logger.info(f"Base noise matrices set: Q shape {Q_base.shape}, R shape {R_base.shape}")
    
    def update_innovation(self, innovation: np.ndarray, innovation_covariance: np.ndarray, 
                         timestamp: datetime, tle_epoch: Optional[datetime] = None):
        """
        Update innovation history and compute adaptive tuning
        
        Args:
            innovation: Innovation vector (measurement - prediction)
            innovation_covariance: Innovation covariance matrix S
            timestamp: Current time
            tle_epoch: TLE epoch time for quality assessment
        """
        try:
            # Store innovation data
            self.innovation_history.append(innovation.copy())
            self.innovation_covariance_history.append(innovation_covariance.copy())
            self.timestamp_history.append(timestamp)
            
            # Update TLE quality assessment
            if tle_epoch is not None:
                self._update_tle_quality(tle_epoch, timestamp)
            
            # Compute innovation statistics if we have enough samples
            if len(self.innovation_history) >= self.config.min_samples_for_adaptation:
                statistics = self._compute_innovation_statistics()
                self.last_statistics = statistics
                
                # Perform adaptive tuning
                self._adapt_noise_matrices(statistics)
                
                # Log adaptation if significant change
                if len(self.adaptation_history) == 0 or self._significant_adaptation_change():
                    self._log_adaptation_status(statistics)
            
        except Exception as e:
            self.logger.error(f"Innovation update failed: {e}")
    
    def get_adapted_noise_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current adapted process and measurement noise matrices
        
        Returns:
            Tuple of (adapted_Q, adapted_R)
        """
        if self.base_process_noise is None or self.base_measurement_noise is None:
            raise ValueError("Base noise matrices not set")
        
        # Apply scaling factors
        adapted_Q = self.base_process_noise * self.process_noise_scale
        adapted_R = self.base_measurement_noise * self.measurement_noise_scale
        
        return adapted_Q, adapted_R
    
    def _update_tle_quality(self, tle_epoch: datetime, current_time: datetime):
        """Update TLE quality factor based on age"""
        try:
            self.last_tle_epoch = tle_epoch
            
            # Calculate TLE age in hours
            tle_age_hours = (current_time - tle_epoch).total_seconds() / 3600.0
            
            # Quality degrades exponentially with age
            if tle_age_hours <= self.config.tle_age_threshold_hours:
                # Good quality within threshold
                age_factor = 1.0
            else:
                # Exponential degradation beyond threshold
                excess_hours = tle_age_hours - self.config.tle_age_threshold_hours
                age_factor = math.exp(-self.config.tle_quality_degradation_rate * excess_hours)
            
            self.tle_quality_factor = max(0.1, age_factor)  # Minimum quality factor
            
        except Exception as e:
            self.logger.warning(f"TLE quality update failed: {e}")
            self.tle_quality_factor = 0.5  # Conservative default
    
    def _compute_innovation_statistics(self) -> InnovationStatistics:
        """Compute comprehensive innovation statistics"""
        try:
            innovations = np.array(list(self.innovation_history))
            covariances = list(self.innovation_covariance_history)
            timestamps = list(self.timestamp_history)
            
            n_samples, n_dim = innovations.shape
            
            # Basic statistics
            mean_innovation = np.mean(innovations, axis=0)
            sample_covariance = np.cov(innovations.T)
            
            # Ensure sample covariance is positive definite
            if sample_covariance.ndim == 0:
                sample_covariance = np.array([[sample_covariance]])
            elif sample_covariance.ndim == 1:
                sample_covariance = np.diag(sample_covariance)
            
            # Add small regularization to diagonal
            sample_covariance += np.eye(sample_covariance.shape[0]) * 1e-12
            
            # Normalized Innovation Squared (NIS) test
            # Average the theoretical covariance matrices
            avg_theoretical_cov = np.mean(covariances, axis=0)
            
            # Compute NIS for each innovation
            nis_values = []
            for i, innov in enumerate(innovations):
                try:
                    theoretical_cov = covariances[i]
                    # Add regularization for numerical stability
                    reg_cov = theoretical_cov + np.eye(theoretical_cov.shape[0]) * 1e-10
                    nis = innov.T @ np.linalg.solve(reg_cov, innov)
                    nis_values.append(nis)
                except np.linalg.LinAlgError:
                    # Fallback for singular matrices
                    nis_values.append(np.sum(innov**2))
            
            avg_nis = np.mean(nis_values)
            
            # Consistency ratio (should be close to measurement dimension)
            expected_nis = n_dim
            consistency_ratio = avg_nis / expected_nis if expected_nis > 0 else 1.0
            
            # Bias indicator (normalized mean innovation)
            try:
                bias_indicator = np.sqrt(mean_innovation.T @ np.linalg.solve(
                    sample_covariance / n_samples, mean_innovation
                ))
            except np.linalg.LinAlgError:
                bias_indicator = np.linalg.norm(mean_innovation)
            
            # Quality metric (combination of consistency and bias)
            quality_metric = self._compute_quality_metric(consistency_ratio, bias_indicator)
            
            # Time span
            time_span = (timestamps[-1] - timestamps[0]).total_seconds()
            
            return InnovationStatistics(
                mean=mean_innovation,
                covariance=sample_covariance,
                normalized_innovation_squared=avg_nis,
                consistency_ratio=consistency_ratio,
                bias_indicator=bias_indicator,
                quality_metric=quality_metric,
                sample_count=n_samples,
                time_span=time_span
            )
            
        except Exception as e:
            self.logger.error(f"Innovation statistics computation failed: {e}")
            # Return default statistics
            n_dim = len(self.innovation_history[0])
            return InnovationStatistics(
                mean=np.zeros(n_dim),
                covariance=np.eye(n_dim),
                normalized_innovation_squared=n_dim,
                consistency_ratio=1.0,
                bias_indicator=0.0,
                quality_metric=0.5,
                sample_count=len(self.innovation_history),
                time_span=60.0
            )
    
    def _compute_quality_metric(self, consistency_ratio: float, bias_indicator: float) -> float:
        """Compute overall quality metric from consistency and bias indicators"""
        try:
            # Consistency component (penalize deviation from 1.0)
            consistency_score = 1.0 / (1.0 + abs(consistency_ratio - 1.0))
            
            # Bias component (penalize high bias)
            bias_score = 1.0 / (1.0 + bias_indicator)
            
            # TLE quality component
            tle_score = self.tle_quality_factor
            
            # Combined quality (weighted average)
            quality = 0.4 * consistency_score + 0.3 * bias_score + 0.3 * tle_score
            
            return np.clip(quality, 0.0, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Quality metric computation failed: {e}")
            return 0.5
    
    def _adapt_noise_matrices(self, statistics: InnovationStatistics):
        """Adapt process and measurement noise matrices based on statistics"""
        try:
            # Process noise adaptation based on consistency ratio
            if statistics.consistency_ratio > self.config.consistency_threshold_high:
                # Innovations too large - increase process noise
                process_adaptation = 1.0 + self.config.process_noise_adaptation_rate * \
                                   (statistics.consistency_ratio - 1.0)
            elif statistics.consistency_ratio < self.config.consistency_threshold_low:
                # Innovations too small - decrease process noise
                process_adaptation = 1.0 - self.config.process_noise_adaptation_rate * \
                                   (1.0 - statistics.consistency_ratio)
            else:
                # Consistency is good - small adjustment toward 1.0
                process_adaptation = 1.0 + 0.01 * (1.0 - statistics.consistency_ratio)
            
            # Measurement noise adaptation based on bias and TLE quality
            measurement_adaptation = 1.0
            
            # Increase measurement noise for high bias (unreliable measurements)
            if statistics.bias_indicator > self.config.bias_detection_threshold:
                bias_factor = 1.0 + self.config.measurement_noise_adaptation_rate * \
                             (statistics.bias_indicator - self.config.bias_detection_threshold)
                measurement_adaptation *= bias_factor
            
            # Increase measurement noise for poor TLE quality
            tle_factor = 1.0 / max(0.1, self.tle_quality_factor)
            measurement_adaptation *= (1.0 + 0.5 * (tle_factor - 1.0))
            
            # Apply adaptations with rate limiting
            max_rate = self.config.max_adaptation_rate
            
            # Rate-limited process noise update
            target_process_scale = self.process_noise_scale * process_adaptation
            process_change = np.clip(
                target_process_scale - self.process_noise_scale,
                -max_rate * self.process_noise_scale,
                max_rate * self.process_noise_scale
            )
            self.process_noise_scale += process_change
            
            # Rate-limited measurement noise update
            target_measurement_scale = self.measurement_noise_scale * measurement_adaptation
            measurement_change = np.clip(
                target_measurement_scale - self.measurement_noise_scale,
                -max_rate * self.measurement_noise_scale,
                max_rate * self.measurement_noise_scale
            )
            self.measurement_noise_scale += measurement_change
            
            # Apply bounds
            self.process_noise_scale = np.clip(
                self.process_noise_scale,
                self.config.min_process_noise_scale,
                self.config.max_process_noise_scale
            )
            
            self.measurement_noise_scale = np.clip(
                self.measurement_noise_scale,
                self.config.min_measurement_noise_scale,
                self.config.max_measurement_noise_scale
            )
            
            # Store adaptation history
            self.adaptation_history.append({
                'timestamp': datetime.utcnow(),
                'process_noise_scale': self.process_noise_scale,
                'measurement_noise_scale': self.measurement_noise_scale,
                'consistency_ratio': statistics.consistency_ratio,
                'bias_indicator': statistics.bias_indicator,
                'quality_metric': statistics.quality_metric,
                'tle_quality_factor': self.tle_quality_factor
            })
            
            # Keep only recent history
            if len(self.adaptation_history) > 1000:
                self.adaptation_history = self.adaptation_history[-500:]
            
        except Exception as e:
            self.logger.error(f"Noise matrix adaptation failed: {e}")
    
    def _significant_adaptation_change(self) -> bool:
        """Check if adaptation has changed significantly since last log"""
        if len(self.adaptation_history) < 2:
            return True
        
        current = self.adaptation_history[-1]
        previous = self.adaptation_history[-2]
        
        process_change = abs(current['process_noise_scale'] - previous['process_noise_scale'])
        measurement_change = abs(current['measurement_noise_scale'] - previous['measurement_noise_scale'])
        
        return process_change > 0.05 or measurement_change > 0.05
    
    def _log_adaptation_status(self, statistics: InnovationStatistics):
        """Log current adaptation status"""
        try:
            self.logger.info(f"Adaptive Filter Tuning Status:")
            self.logger.info(f"  Innovation samples: {statistics.sample_count}")
            self.logger.info(f"  Consistency ratio: {statistics.consistency_ratio:.3f}")
            self.logger.info(f"  Bias indicator: {statistics.bias_indicator:.3f}")
            self.logger.info(f"  Quality metric: {statistics.quality_metric:.3f}")
            self.logger.info(f"  TLE quality factor: {self.tle_quality_factor:.3f}")
            self.logger.info(f"  Process noise scale: {self.process_noise_scale:.3f}")
            self.logger.info(f"  Measurement noise scale: {self.measurement_noise_scale:.3f}")
            
            # Diagnostic messages
            if statistics.consistency_ratio > self.config.consistency_threshold_high:
                self.logger.warning("High innovation consistency - increasing process noise")
            elif statistics.consistency_ratio < self.config.consistency_threshold_low:
                self.logger.warning("Low innovation consistency - decreasing process noise")
            
            if statistics.bias_indicator > self.config.bias_detection_threshold:
                self.logger.warning(f"High bias detected ({statistics.bias_indicator:.2f}) - increasing measurement noise")
            
            if self.tle_quality_factor < 0.5:
                self.logger.warning(f"Poor TLE quality ({self.tle_quality_factor:.2f}) - increasing measurement noise")
                
        except Exception as e:
            self.logger.error(f"Adaptation status logging failed: {e}")
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics for monitoring"""
        try:
            diagnostics = {
                'process_noise_scale': self.process_noise_scale,
                'measurement_noise_scale': self.measurement_noise_scale,
                'tle_quality_factor': self.tle_quality_factor,
                'innovation_samples': len(self.innovation_history),
                'adaptation_history_length': len(self.adaptation_history)
            }
            
            if self.last_statistics is not None:
                diagnostics.update({
                    'consistency_ratio': self.last_statistics.consistency_ratio,
                    'bias_indicator': self.last_statistics.bias_indicator,
                    'quality_metric': self.last_statistics.quality_metric,
                    'normalized_innovation_squared': self.last_statistics.normalized_innovation_squared
                })
            
            if self.last_tle_epoch is not None:
                diagnostics['tle_age_hours'] = (datetime.utcnow() - self.last_tle_epoch).total_seconds() / 3600.0
            
            return diagnostics
            
        except Exception as e:
            self.logger.error(f"Diagnostics generation failed: {e}")
            return {'error': str(e)}
    
    def reset_adaptation(self):
        """Reset adaptation to initial state"""
        try:
            self.innovation_history.clear()
            self.innovation_covariance_history.clear()
            self.timestamp_history.clear()
            
            self.process_noise_scale = 1.0
            self.measurement_noise_scale = 1.0
            self.tle_quality_factor = 1.0
            
            self.last_statistics = None
            self.adaptation_history.clear()
            
            self.logger.info("Adaptive filter tuning reset to initial state")
            
        except Exception as e:
            self.logger.error(f"Adaptation reset failed: {e}")
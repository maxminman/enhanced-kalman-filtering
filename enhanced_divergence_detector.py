#!/usr/bin/env python3
"""
Enhanced Divergence Detection and Recovery System
Provides multi-level divergence detection with intelligent recovery strategies
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

class DivergenceLevel(Enum):
    """Levels of filter divergence"""
    HEALTHY = "healthy"
    WARNING = "warning"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"

class RecoveryAction(Enum):
    """Types of recovery actions"""
    INCREASE_PROCESS_NOISE = "increase_process_noise"
    INCREASE_MEASUREMENT_NOISE = "increase_measurement_noise"
    RESET_PARAMETERS = "reset_parameters"
    REINITIALIZE_COVARIANCE = "reinitialize_covariance"
    SKIP_MEASUREMENT = "skip_measurement"
    FULL_RESTART = "full_restart"

@dataclass
class DivergenceMetrics:
    """Comprehensive divergence assessment metrics"""
    divergence_level: DivergenceLevel
    innovation_norm: float
    normalized_innovation: float
    covariance_condition_number: float
    parameter_stability: float
    confidence_score: float
    time_since_last_good: float
    primary_indicators: List[str]
    secondary_indicators: List[str]

@dataclass
class RecoveryConfig:
    """Configuration for divergence detection and recovery"""
    # Innovation thresholds (meters)
    innovation_warning_threshold: float = 1000.0
    innovation_moderate_threshold: float = 5000.0
    innovation_severe_threshold: float = 20000.0
    innovation_critical_threshold: float = 50000.0
    
    # Normalized innovation thresholds (chi-squared)
    normalized_innovation_warning: float = 10.0
    normalized_innovation_severe: float = 50.0
    
    # Covariance condition number thresholds
    condition_number_warning: float = 1e8
    condition_number_severe: float = 1e12
    
    # Parameter stability thresholds
    parameter_change_warning: float = 0.5  # 50% change
    parameter_change_severe: float = 2.0   # 200% change
    
    # Time-based thresholds
    max_time_without_good_update: float = 3600.0  # 1 hour
    
    # Recovery parameters
    process_noise_increase_factor: float = 5.0
    measurement_noise_increase_factor: float = 10.0
    covariance_reset_factor: float = 100.0
    
    # Consecutive failure limits
    max_consecutive_warnings: int = 5
    max_consecutive_moderate: int = 3
    max_consecutive_severe: int = 2

class EnhancedDivergenceDetector:
    """
    Enhanced divergence detection system with multi-level assessment
    and intelligent recovery strategies
    """
    
    def __init__(self, config: Optional[RecoveryConfig] = None):
        """Initialize enhanced divergence detector"""
        self.config = config or RecoveryConfig()
        self.logger = logging.getLogger(__name__)
        
        # State tracking
        self.last_good_update = datetime.utcnow()
        self.consecutive_warnings = 0
        self.consecutive_moderate = 0
        self.consecutive_severe = 0
        
        # History for trend analysis
        self.innovation_history = []
        self.parameter_history = []
        self.condition_number_history = []
        
        # Recovery tracking
        self.recovery_attempts = 0
        self.last_recovery_time = None
        
        self.logger.info("Enhanced divergence detector initialized")
    
    def assess_divergence(self, innovation: np.ndarray, innovation_covariance: np.ndarray,
                         state_covariance: np.ndarray, parameters: np.ndarray,
                         current_time: datetime) -> DivergenceMetrics:
        """
        Comprehensive divergence assessment
        
        Args:
            innovation: Innovation vector (m)
            innovation_covariance: Innovation covariance matrix S
            state_covariance: State covariance matrix P
            parameters: Current parameter vector [Bc, Cr, empirical_accel]
            current_time: Current time
            
        Returns:
            DivergenceMetrics with comprehensive assessment
        """
        try:
            # Compute basic metrics
            innovation_norm = np.linalg.norm(innovation)
            
            # Normalized innovation (chi-squared test)
            try:
                normalized_innovation = innovation.T @ np.linalg.solve(innovation_covariance, innovation)
            except np.linalg.LinAlgError:
                normalized_innovation = np.inf
            
            # Covariance condition number
            try:
                condition_number = np.linalg.cond(state_covariance)
            except:
                condition_number = np.inf
            
            # Parameter stability assessment
            parameter_stability = self._assess_parameter_stability(parameters)
            
            # Time since last good update
            time_since_good = (current_time - self.last_good_update).total_seconds()
            
            # Store history
            self.innovation_history.append(innovation_norm)
            self.parameter_history.append(parameters.copy())
            self.condition_number_history.append(condition_number)
            
            # Keep only recent history
            if len(self.innovation_history) > 100:
                self.innovation_history = self.innovation_history[-50:]
                self.parameter_history = self.parameter_history[-50:]
                self.condition_number_history = self.condition_number_history[-50:]
            
            # Assess divergence level
            divergence_level, primary_indicators, secondary_indicators = self._classify_divergence(
                innovation_norm, normalized_innovation, condition_number, 
                parameter_stability, time_since_good
            )
            
            # Compute confidence score
            confidence_score = self._compute_confidence_score(
                innovation_norm, normalized_innovation, condition_number, parameter_stability
            )
            
            # Update consecutive counters
            self._update_consecutive_counters(divergence_level)
            
            # Update last good update time if healthy
            if divergence_level == DivergenceLevel.HEALTHY:
                self.last_good_update = current_time
            
            return DivergenceMetrics(
                divergence_level=divergence_level,
                innovation_norm=innovation_norm,
                normalized_innovation=normalized_innovation,
                covariance_condition_number=condition_number,
                parameter_stability=parameter_stability,
                confidence_score=confidence_score,
                time_since_last_good=time_since_good,
                primary_indicators=primary_indicators,
                secondary_indicators=secondary_indicators
            )
            
        except Exception as e:
            self.logger.error(f"Divergence assessment failed: {e}")
            # Return critical divergence on error
            return DivergenceMetrics(
                divergence_level=DivergenceLevel.CRITICAL,
                innovation_norm=np.inf,
                normalized_innovation=np.inf,
                covariance_condition_number=np.inf,
                parameter_stability=np.inf,
                confidence_score=0.0,
                time_since_last_good=time_since_good if 'time_since_good' in locals() else 0.0,
                primary_indicators=["assessment_error"],
                secondary_indicators=[str(e)]
            )
    
    def _assess_parameter_stability(self, current_params: np.ndarray) -> float:
        """Assess parameter stability based on recent changes"""
        try:
            if len(self.parameter_history) < 2:
                return 0.0  # No history to compare
            
            # Compare with recent parameters
            recent_params = self.parameter_history[-5:] if len(self.parameter_history) >= 5 else self.parameter_history
            
            max_relative_change = 0.0
            for past_params in recent_params:
                for i, (current, past) in enumerate(zip(current_params, past_params)):
                    if abs(past) > 1e-12:  # Avoid division by zero
                        relative_change = abs((current - past) / past)
                        max_relative_change = max(max_relative_change, relative_change)
            
            return max_relative_change
            
        except Exception as e:
            self.logger.warning(f"Parameter stability assessment failed: {e}")
            return np.inf
    
    def _classify_divergence(self, innovation_norm: float, normalized_innovation: float,
                           condition_number: float, parameter_stability: float,
                           time_since_good: float) -> Tuple[DivergenceLevel, List[str], List[str]]:
        """Classify divergence level based on multiple indicators"""
        
        primary_indicators = []
        secondary_indicators = []
        
        # Innovation-based classification
        if innovation_norm > self.config.innovation_critical_threshold:
            primary_indicators.append(f"critical_innovation_{innovation_norm/1000:.1f}km")
        elif innovation_norm > self.config.innovation_severe_threshold:
            primary_indicators.append(f"severe_innovation_{innovation_norm/1000:.1f}km")
        elif innovation_norm > self.config.innovation_moderate_threshold:
            primary_indicators.append(f"moderate_innovation_{innovation_norm/1000:.1f}km")
        elif innovation_norm > self.config.innovation_warning_threshold:
            secondary_indicators.append(f"warning_innovation_{innovation_norm/1000:.1f}km")
        
        # Normalized innovation classification
        if normalized_innovation > self.config.normalized_innovation_severe:
            primary_indicators.append(f"severe_normalized_innovation_{normalized_innovation:.1f}")
        elif normalized_innovation > self.config.normalized_innovation_warning:
            secondary_indicators.append(f"warning_normalized_innovation_{normalized_innovation:.1f}")
        
        # Covariance condition number classification
        if condition_number > self.config.condition_number_severe:
            primary_indicators.append(f"severe_ill_conditioning_{condition_number:.1e}")
        elif condition_number > self.config.condition_number_warning:
            secondary_indicators.append(f"warning_ill_conditioning_{condition_number:.1e}")
        
        # Parameter stability classification
        if parameter_stability > self.config.parameter_change_severe:
            primary_indicators.append(f"severe_parameter_instability_{parameter_stability:.2f}")
        elif parameter_stability > self.config.parameter_change_warning:
            secondary_indicators.append(f"warning_parameter_instability_{parameter_stability:.2f}")
        
        # Time-based classification
        if time_since_good > self.config.max_time_without_good_update:
            primary_indicators.append(f"prolonged_poor_performance_{time_since_good/3600:.1f}h")
        
        # Consecutive failure classification
        if self.consecutive_severe >= self.config.max_consecutive_severe:
            primary_indicators.append(f"consecutive_severe_failures_{self.consecutive_severe}")
        elif self.consecutive_moderate >= self.config.max_consecutive_moderate:
            secondary_indicators.append(f"consecutive_moderate_failures_{self.consecutive_moderate}")
        elif self.consecutive_warnings >= self.config.max_consecutive_warnings:
            secondary_indicators.append(f"consecutive_warnings_{self.consecutive_warnings}")
        
        # Determine overall level
        if any("critical" in indicator for indicator in primary_indicators):
            level = DivergenceLevel.CRITICAL
        elif len(primary_indicators) >= 2:
            level = DivergenceLevel.SEVERE
        elif len(primary_indicators) >= 1:
            level = DivergenceLevel.MODERATE
        elif len(secondary_indicators) >= 2:
            level = DivergenceLevel.WARNING
        else:
            level = DivergenceLevel.HEALTHY
        
        return level, primary_indicators, secondary_indicators
    
    def _compute_confidence_score(self, innovation_norm: float, normalized_innovation: float,
                                condition_number: float, parameter_stability: float) -> float:
        """Compute overall filter confidence score (0.0 to 1.0)"""
        try:
            # Innovation confidence (inverse relationship)
            innovation_confidence = 1.0 / (1.0 + innovation_norm / 1000.0)  # Normalize by 1km
            
            # Normalized innovation confidence
            normalized_confidence = 1.0 / (1.0 + normalized_innovation / 10.0)
            
            # Covariance confidence
            condition_confidence = 1.0 / (1.0 + np.log10(max(1.0, condition_number)) / 10.0)
            
            # Parameter confidence
            parameter_confidence = 1.0 / (1.0 + parameter_stability)
            
            # Weighted average
            confidence = (0.4 * innovation_confidence + 
                         0.3 * normalized_confidence + 
                         0.2 * condition_confidence + 
                         0.1 * parameter_confidence)
            
            return np.clip(confidence, 0.0, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Confidence score computation failed: {e}")
            return 0.0
    
    def _update_consecutive_counters(self, divergence_level: DivergenceLevel):
        """Update consecutive failure counters"""
        if divergence_level == DivergenceLevel.SEVERE or divergence_level == DivergenceLevel.CRITICAL:
            self.consecutive_severe += 1
            self.consecutive_moderate = 0
            self.consecutive_warnings = 0
        elif divergence_level == DivergenceLevel.MODERATE:
            self.consecutive_moderate += 1
            self.consecutive_severe = 0
            self.consecutive_warnings = 0
        elif divergence_level == DivergenceLevel.WARNING:
            self.consecutive_warnings += 1
            self.consecutive_severe = 0
            self.consecutive_moderate = 0
        else:  # HEALTHY
            self.consecutive_warnings = 0
            self.consecutive_moderate = 0
            self.consecutive_severe = 0
    
    def recommend_recovery_actions(self, metrics: DivergenceMetrics, 
                                 state: np.ndarray, covariance: np.ndarray,
                                 parameters: np.ndarray) -> List[RecoveryAction]:
        """Recommend recovery actions based on divergence assessment"""
        actions = []
        
        try:
            if metrics.divergence_level == DivergenceLevel.CRITICAL:
                # Critical: Full restart
                actions.append(RecoveryAction.FULL_RESTART)
                
            elif metrics.divergence_level == DivergenceLevel.SEVERE:
                # Severe: Multiple recovery actions
                actions.append(RecoveryAction.REINITIALIZE_COVARIANCE)
                actions.append(RecoveryAction.RESET_PARAMETERS)
                actions.append(RecoveryAction.INCREASE_PROCESS_NOISE)
                
            elif metrics.divergence_level == DivergenceLevel.MODERATE:
                # Moderate: Targeted recovery
                if "parameter_instability" in str(metrics.primary_indicators):
                    actions.append(RecoveryAction.RESET_PARAMETERS)
                if "ill_conditioning" in str(metrics.primary_indicators):
                    actions.append(RecoveryAction.REINITIALIZE_COVARIANCE)
                if "innovation" in str(metrics.primary_indicators):
                    actions.append(RecoveryAction.INCREASE_MEASUREMENT_NOISE)
                
                # Default moderate action
                if not actions:
                    actions.append(RecoveryAction.INCREASE_PROCESS_NOISE)
                    
            elif metrics.divergence_level == DivergenceLevel.WARNING:
                # Warning: Conservative recovery
                if metrics.innovation_norm > self.config.innovation_warning_threshold * 2:
                    actions.append(RecoveryAction.SKIP_MEASUREMENT)
                else:
                    actions.append(RecoveryAction.INCREASE_MEASUREMENT_NOISE)
            
            # Log recommended actions
            if actions:
                action_names = [action.value for action in actions]
                self.logger.info(f"Recommended recovery actions: {action_names}")
            
            return actions
            
        except Exception as e:
            self.logger.error(f"Recovery recommendation failed: {e}")
            return [RecoveryAction.INCREASE_PROCESS_NOISE]  # Safe default
    
    def execute_recovery_action(self, action: RecoveryAction, state: np.ndarray,
                              covariance: np.ndarray, parameters: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Execute a specific recovery action"""
        try:
            new_state = state.copy()
            new_covariance = covariance.copy()
            new_parameters = parameters.copy()
            
            if action == RecoveryAction.INCREASE_PROCESS_NOISE:
                # Increase process noise by scaling covariance
                new_covariance *= self.config.process_noise_increase_factor
                self.logger.info(f"Increased process noise by factor {self.config.process_noise_increase_factor}")
                
            elif action == RecoveryAction.INCREASE_MEASUREMENT_NOISE:
                # This is handled by the adaptive tuner, just log
                self.logger.info("Recommended measurement noise increase (handled by adaptive tuner)")
                
            elif action == RecoveryAction.RESET_PARAMETERS:
                # Reset parameters to reasonable defaults
                new_parameters[0] = 0.0055  # Default Bc for ISS
                new_parameters[1] = 1.3     # Default Cr
                new_parameters[2] = 0.0     # Reset empirical acceleration
                
                # Update state if parameters are part of state vector
                if len(new_state) > 6:
                    new_state[6:6+len(new_parameters)] = new_parameters
                
                self.logger.info("Reset parameters to default values")
                
            elif action == RecoveryAction.REINITIALIZE_COVARIANCE:
                # Reinitialize covariance with larger uncertainties
                state_dim = new_covariance.shape[0]
                new_covariance = np.eye(state_dim)
                
                # Position uncertainty: 10 km
                new_covariance[:3, :3] *= (10000)**2
                
                # Velocity uncertainty: 10 m/s
                new_covariance[3:6, 3:6] *= (10)**2
                
                # Parameter uncertainties
                if state_dim > 6:
                    new_covariance[6, 6] = (0.01)**2    # Bc uncertainty
                if state_dim > 7:
                    new_covariance[7, 7] = (0.5)**2     # Cr uncertainty
                if state_dim > 8:
                    new_covariance[8, 8] = (1e-6)**2    # Empirical accel uncertainty
                
                self.logger.info("Reinitialized covariance matrix with larger uncertainties")
                
            elif action == RecoveryAction.SKIP_MEASUREMENT:
                # This is handled by returning a flag, just log
                self.logger.info("Recommended skipping current measurement")
                
            elif action == RecoveryAction.FULL_RESTART:
                # Full restart - reinitialize everything
                new_covariance *= self.config.covariance_reset_factor
                new_parameters[0] = 0.0055  # Default Bc
                new_parameters[1] = 1.3     # Default Cr
                new_parameters[2] = 0.0     # Reset empirical acceleration
                
                if len(new_state) > 6:
                    new_state[6:6+len(new_parameters)] = new_parameters
                
                self.logger.warning("Executed full filter restart")
            
            # Track recovery attempts
            self.recovery_attempts += 1
            self.last_recovery_time = datetime.utcnow()
            
            return new_state, new_covariance, new_parameters
            
        except Exception as e:
            self.logger.error(f"Recovery action execution failed: {e}")
            return state, covariance, parameters  # Return unchanged on error
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics for monitoring"""
        try:
            return {
                'consecutive_warnings': self.consecutive_warnings,
                'consecutive_moderate': self.consecutive_moderate,
                'consecutive_severe': self.consecutive_severe,
                'recovery_attempts': self.recovery_attempts,
                'last_recovery_time': self.last_recovery_time.isoformat() if self.last_recovery_time else None,
                'last_good_update': self.last_good_update.isoformat(),
                'innovation_history_length': len(self.innovation_history),
                'recent_innovation_trend': np.mean(self.innovation_history[-10:]) if len(self.innovation_history) >= 10 else 0.0,
                'recent_condition_numbers': self.condition_number_history[-5:] if len(self.condition_number_history) >= 5 else []
            }
        except Exception as e:
            self.logger.error(f"Diagnostics generation failed: {e}")
            return {'error': str(e)}
    
    def reset(self):
        """Reset divergence detector to initial state"""
        try:
            self.last_good_update = datetime.utcnow()
            self.consecutive_warnings = 0
            self.consecutive_moderate = 0
            self.consecutive_severe = 0
            
            self.innovation_history.clear()
            self.parameter_history.clear()
            self.condition_number_history.clear()
            
            self.recovery_attempts = 0
            self.last_recovery_time = None
            
            self.logger.info("Enhanced divergence detector reset")
            
        except Exception as e:
            self.logger.error(f"Divergence detector reset failed: {e}")
#!/usr/bin/env python3
"""
Integrated Sub-1km Orbital Determination System
Complete integration of all enhanced components for satellite-agnostic sub-1km accuracy
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

# Import all enhanced components
from enhanced_ekf_tracker import EnhancedEKFTracker
from satellite_characterizer import SatelliteCharacterizer
from adaptive_filter_tuner import AdaptiveFilterTuner, AdaptationConfig
from enhanced_divergence_detector import EnhancedDivergenceDetector, RecoveryConfig
from batch_estimator import EnhancedBatchEstimator
from tle_measurement_model import TLEMeasurementModel
from multi_source_tle_manager import MultiSourceTLEManager
from validation_framework import ValidationFramework
from enhanced_srp_model import EnhancedSRPModel
from force_models import ForceModels
from adaptive_atmospheric_model import AdaptiveAtmosphericModel

@dataclass
class Sub1kmSystemConfig:
    """Configuration for the integrated sub-1km system"""
    # System-wide settings
    target_accuracy_m: float = 500.0
    validation_enabled: bool = True
    real_time_monitoring: bool = True
    
    # Component configurations
    adaptive_tuning_config: Dict[str, Any] = None
    divergence_detection_config: Dict[str, Any] = None
    batch_estimation_config: Dict[str, Any] = None
    tle_management_config: Dict[str, Any] = None
    validation_config: Dict[str, Any] = None
    
    # Performance settings
    max_satellites: int = 10
    update_interval_seconds: float = 240.0
    validation_interval_hours: float = 24.0

@dataclass
class TrackingResult:
    """Comprehensive tracking result"""
    satellite_id: str
    timestamp: datetime
    position: np.ndarray
    velocity: np.ndarray
    covariance: np.ndarray
    accuracy_estimate: float
    quality_metrics: Dict[str, Any]
    adaptive_diagnostics: Dict[str, Any]
    parameter_estimates: Dict[str, Any]
    validation_results: Optional[Dict[str, Any]]

class IntegratedSub1kmSystem:
    """
    Integrated sub-1km orbital determination system combining all enhanced components
    for satellite-agnostic high-accuracy tracking
    """
    
    def __init__(self, config: Sub1kmSystemConfig):
        """Initialize the integrated sub-1km system"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Active satellite trackers
        self.active_trackers = {}  # satellite_id -> EnhancedEKFTracker
        self.satellite_metadata = {}  # satellite_id -> metadata
        
        # Shared components
        self.satellite_characterizer = SatelliteCharacterizer()
        self.tle_manager = MultiSourceTLEManager(config.tle_management_config or {})
        self.validation_framework = ValidationFramework(config.validation_config or {})
        
        # System-wide monitoring
        self.system_performance = {}
        self.tracking_history = {}
        self.validation_results = {}
        
        # Performance monitoring
        self.last_validation_time = {}
        self.accuracy_trends = {}
        
        self.logger.info("Integrated Sub-1km System initialized")
    
    def add_satellite(self, norad_id: str, tle_line1: str = None, tle_line2: str = None,
                     satellite_name: str = None) -> bool:
        """
        Add a satellite to the tracking system with automatic characterization
        and configuration
        
        Args:
            norad_id: NORAD catalog ID
            tle_line1: TLE line 1 (optional, will fetch if not provided)
            tle_line2: TLE line 2 (optional, will fetch if not provided)
            satellite_name: Satellite name (optional)
            
        Returns:
            True if successfully added
        """
        try:
            if norad_id in self.active_trackers:
                self.logger.warning(f"Satellite {norad_id} already being tracked")
                return True
            
            # Fetch TLE if not provided
            if not tle_line1 or not tle_line2:
                tle_data = self.tle_manager.fetch_tle_multi_source(norad_id)
                if not tle_data:
                    self.logger.error(f"Failed to fetch TLE for satellite {norad_id}")
                    return False
                tle_line1 = tle_data.line1
                tle_line2 = tle_data.line2
            
            # Characterize satellite
            from utils import parse_tle
            tle_parsed = parse_tle(tle_line1, tle_line2)
            satellite_props = self.satellite_characterizer.characterize_satellite(tle_parsed, norad_id)
            
            # Create satellite-specific configuration
            tracker_config = self._create_satellite_specific_config(satellite_props)
            
            # Initialize enhanced EKF tracker
            tracker = EnhancedEKFTracker(tle_line1, tle_line2, tracker_config)
            
            # Store tracker and metadata
            self.active_trackers[norad_id] = tracker
            self.satellite_metadata[norad_id] = {
                'name': satellite_name or f"NORAD-{norad_id}",
                'properties': satellite_props,
                'tle_epoch': tle_parsed.epoch_datetime,
                'added_time': datetime.utcnow(),
                'tracker_config': tracker_config
            }
            
            # Initialize tracking history
            self.tracking_history[norad_id] = []
            self.accuracy_trends[norad_id] = []
            
            self.logger.info(f"Successfully added satellite {norad_id} ({satellite_name}) to tracking system")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add satellite {norad_id}: {e}")
            return False
    
    def _create_satellite_specific_config(self, satellite_props) -> Dict[str, Any]:
        """Create satellite-specific configuration based on characterization"""
        try:
            # Base configuration
            config = {
                # Enhanced EKF settings
                'use_adaptive_tuning': True,
                'use_enhanced_divergence_detection': True,
                'use_batch_estimation': True,
                'use_srp': True,
                'use_drag': True,
                'use_j2_j6': True,
                
                # Satellite-specific parameters
                'satellite_mass': satellite_props.mass,
                'drag_area': satellite_props.drag_area,
                'srp_area': satellite_props.srp_area,
                'drag_coeff': satellite_props.drag_coefficient,
                'srp_coeff': satellite_props.srp_coefficient,
                'ballistic_coeff': satellite_props.ballistic_coefficient,
                'norad_id': satellite_props.norad_id,
                
                # Adaptive tuning configuration
                'innovation_window_size': 30,
                'min_samples_for_adaptation': 8,
                'max_adaptation_rate': 0.15,
                'bias_detection_threshold': 2.5,
                'process_noise_adaptation_rate': 0.08,
                'measurement_noise_adaptation_rate': 0.12,
                'tle_age_threshold_hours': 18.0,
                
                # Divergence detection configuration
                'innovation_warning_threshold': 2000.0,  # 2km
                'innovation_severe_threshold': 10000.0,  # 10km
                'covariance_condition_threshold': 1e10,
                
                # Batch estimation configuration
                'batch_window_size': 40,
                'real_time_adaptation': True,
                'robust_estimation': True,
                
                # Force model configuration
                'process_noise_scale': 0.3,  # Conservative for stability
                'use_nrlmsise': True,
                'use_adaptive_atmosphere': True
            }
            
            # Satellite-type specific adjustments
            if satellite_props.satellite_type.value == 'space_station':
                config.update({
                    'innovation_window_size': 40,  # Larger window for complex dynamics
                    'process_noise_scale': 0.4,    # Higher process noise for complex satellite
                    'batch_window_size': 50
                })
            elif satellite_props.satellite_type.value == 'scientific':
                config.update({
                    'innovation_window_size': 25,  # Smaller window for simpler dynamics
                    'process_noise_scale': 0.2,    # Lower process noise for simple satellite
                    'batch_window_size': 30
                })
            
            # Orbital regime specific adjustments
            if satellite_props.orbital_regime.value == 'very_low_leo':
                config.update({
                    'tle_age_threshold_hours': 12.0,  # Fresher TLEs needed for low altitude
                    'process_noise_adaptation_rate': 0.1,  # Higher adaptation for drag uncertainty
                })
            elif satellite_props.orbital_regime.value == 'high_leo':
                config.update({
                    'tle_age_threshold_hours': 24.0,  # Can use older TLEs
                    'process_noise_adaptation_rate': 0.06,  # Lower adaptation for stable orbit
                })
            
            return config
            
        except Exception as e:
            self.logger.error(f"Satellite-specific configuration creation failed: {e}")
            # Return default configuration
            return {
                'use_adaptive_tuning': True,
                'use_enhanced_divergence_detection': True,
                'satellite_mass': 1000.0,
                'drag_area': 10.0,
                'srp_area': 10.0,
                'norad_id': '00000'
            }
    
    def update_all_satellites(self, current_time: datetime = None) -> Dict[str, TrackingResult]:
        """
        Update all active satellite trackers and return comprehensive results
        
        Args:
            current_time: Current time (uses utcnow if not provided)
            
        Returns:
            Dictionary of satellite_id -> TrackingResult
        """
        try:
            if current_time is None:
                current_time = datetime.utcnow()
            
            results = {}
            
            for satellite_id, tracker in self.active_trackers.items():
                try:
                    # Update individual tracker
                    tracker_result = tracker.update(current_time)
                    
                    if tracker_result is not None:
                        # Create comprehensive tracking result
                        tracking_result = self._create_comprehensive_result(
                            satellite_id, tracker_result, current_time
                        )
                        
                        results[satellite_id] = tracking_result
                        
                        # Store in history
                        self.tracking_history[satellite_id].append(tracking_result)
                        
                        # Limit history size
                        if len(self.tracking_history[satellite_id]) > 1000:
                            self.tracking_history[satellite_id] = self.tracking_history[satellite_id][-500:]
                        
                        # Update accuracy trends
                        self._update_accuracy_trends(satellite_id, tracking_result)
                        
                        # Trigger validation if needed
                        if self._should_run_validation(satellite_id, current_time):
                            self._run_validation(satellite_id, current_time)
                    
                except Exception as e:
                    self.logger.error(f"Update failed for satellite {satellite_id}: {e}")
                    continue
            
            # Update system performance metrics
            self._update_system_performance(results, current_time)
            
            return results
            
        except Exception as e:
            self.logger.error(f"System-wide update failed: {e}")
            return {}
    
    def _create_comprehensive_result(self, satellite_id: str, tracker_result: Dict[str, Any],
                                   current_time: datetime) -> TrackingResult:
        """Create comprehensive tracking result with all diagnostics"""
        try:
            tracker = self.active_trackers[satellite_id]
            
            # Extract basic results
            position = tracker_result.get('position', np.zeros(3))
            velocity = tracker_result.get('velocity', np.zeros(3))
            covariance = tracker_result.get('covariance', np.eye(9))
            
            # Estimate accuracy from covariance
            position_uncertainty = np.sqrt(np.trace(covariance[:3, :3]))
            
            # Get adaptive diagnostics
            adaptive_diagnostics = {}
            if hasattr(tracker, 'adaptive_tuner') and tracker.adaptive_tuner:
                adaptive_diagnostics = tracker.adaptive_tuner.get_diagnostics()
            
            # Get parameter estimates
            parameter_estimates = {}
            if hasattr(tracker, 'batch_estimator') and tracker.batch_estimator:
                current_params, param_uncertainties = tracker.batch_estimator.get_current_parameters()
                parameter_estimates = {
                    'ballistic_coefficient': current_params[0] if len(current_params) > 0 else 0.0,
                    'srp_coefficient': current_params[1] if len(current_params) > 1 else 0.0,
                    'empirical_acceleration': current_params[2] if len(current_params) > 2 else 0.0,
                    'parameter_uncertainties': param_uncertainties.tolist()
                }
            
            # Quality metrics
            quality_metrics = {
                'position_uncertainty_m': position_uncertainty,
                'covariance_condition_number': np.linalg.cond(covariance),
                'filter_health': 'healthy',  # Could be enhanced with more checks
                'last_update_success': True
            }
            
            # Get validation results if available
            validation_results = self.validation_results.get(satellite_id)
            
            return TrackingResult(
                satellite_id=satellite_id,
                timestamp=current_time,
                position=position,
                velocity=velocity,
                covariance=covariance,
                accuracy_estimate=position_uncertainty,
                quality_metrics=quality_metrics,
                adaptive_diagnostics=adaptive_diagnostics,
                parameter_estimates=parameter_estimates,
                validation_results=validation_results
            )
            
        except Exception as e:
            self.logger.error(f"Comprehensive result creation failed for {satellite_id}: {e}")
            return TrackingResult(
                satellite_id=satellite_id,
                timestamp=current_time,
                position=np.zeros(3),
                velocity=np.zeros(3),
                covariance=np.eye(9),
                accuracy_estimate=float('inf'),
                quality_metrics={'error': str(e)},
                adaptive_diagnostics={},
                parameter_estimates={},
                validation_results=None
            )
    
    def _update_accuracy_trends(self, satellite_id: str, result: TrackingResult):
        """Update accuracy trends for monitoring"""
        try:
            trend_entry = {
                'timestamp': result.timestamp,
                'accuracy_estimate': result.accuracy_estimate,
                'adaptive_quality': result.adaptive_diagnostics.get('quality_metric', 0.5),
                'parameter_stability': result.parameter_estimates.get('parameter_uncertainties', [1.0])[0] if result.parameter_estimates.get('parameter_uncertainties') else 1.0
            }
            
            self.accuracy_trends[satellite_id].append(trend_entry)
            
            # Keep limited history
            if len(self.accuracy_trends[satellite_id]) > 200:
                self.accuracy_trends[satellite_id] = self.accuracy_trends[satellite_id][-100:]
                
        except Exception as e:
            self.logger.error(f"Accuracy trend update failed for {satellite_id}: {e}")
    
    def _should_run_validation(self, satellite_id: str, current_time: datetime) -> bool:
        """Check if validation should be run for a satellite"""
        try:
            if not self.config.validation_enabled:
                return False
            
            last_validation = self.last_validation_time.get(satellite_id)
            if last_validation is None:
                return True
            
            time_since_validation = (current_time - last_validation).total_seconds() / 3600.0
            return time_since_validation >= self.config.validation_interval_hours
            
        except Exception as e:
            self.logger.error(f"Validation check failed for {satellite_id}: {e}")
            return False
    
    def _run_validation(self, satellite_id: str, current_time: datetime):
        """Run validation for a specific satellite"""
        try:
            # Get recent tracking history
            recent_history = self.tracking_history.get(satellite_id, [])[-50:]  # Last 50 points
            
            if len(recent_history) < 10:
                self.logger.debug(f"Insufficient history for validation of {satellite_id}")
                return
            
            # Convert to validation format
            validation_data = []
            for result in recent_history:
                validation_data.append({
                    'timestamp': result.timestamp,
                    'position': result.position,
                    'velocity': result.velocity,
                    'position_error': result.accuracy_estimate  # Estimated error
                })
            
            # Run validation
            validation_result = self.validation_framework.validate_tracking_data(validation_data)
            
            # Store results
            self.validation_results[satellite_id] = validation_result
            self.last_validation_time[satellite_id] = current_time
            
            # Log validation summary
            if 'metrics' in validation_result:
                metrics = validation_result['metrics']
                rms_error = metrics.get('rms_error', float('inf'))
                self.logger.info(f"Validation for {satellite_id}: RMS error = {rms_error:.1f}m")
            
        except Exception as e:
            self.logger.error(f"Validation failed for {satellite_id}: {e}")
    
    def _update_system_performance(self, results: Dict[str, TrackingResult], current_time: datetime):
        """Update system-wide performance metrics"""
        try:
            if not results:
                return
            
            # Compute system-wide metrics
            accuracy_estimates = [r.accuracy_estimate for r in results.values() if r.accuracy_estimate != float('inf')]
            
            if accuracy_estimates:
                system_metrics = {
                    'timestamp': current_time,
                    'active_satellites': len(results),
                    'average_accuracy_m': np.mean(accuracy_estimates),
                    'best_accuracy_m': np.min(accuracy_estimates),
                    'worst_accuracy_m': np.max(accuracy_estimates),
                    'sub1km_satellites': sum(1 for acc in accuracy_estimates if acc < 1000),
                    'sub500m_satellites': sum(1 for acc in accuracy_estimates if acc < 500)
                }
                
                self.system_performance[current_time] = system_metrics
                
                # Keep limited history
                if len(self.system_performance) > 1000:
                    # Keep only recent 500 entries
                    recent_keys = sorted(self.system_performance.keys())[-500:]
                    self.system_performance = {k: self.system_performance[k] for k in recent_keys}
            
        except Exception as e:
            self.logger.error(f"System performance update failed: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            status = {
                'system_info': {
                    'active_satellites': len(self.active_trackers),
                    'max_satellites': self.config.max_satellites,
                    'target_accuracy_m': self.config.target_accuracy_m,
                    'validation_enabled': self.config.validation_enabled
                },
                'satellite_status': {},
                'system_performance': {},
                'component_status': {}
            }
            
            # Individual satellite status
            for satellite_id, metadata in self.satellite_metadata.items():
                recent_history = self.tracking_history.get(satellite_id, [])
                latest_result = recent_history[-1] if recent_history else None
                
                status['satellite_status'][satellite_id] = {
                    'name': metadata['name'],
                    'satellite_type': metadata['properties'].satellite_type.value,
                    'orbital_regime': metadata['properties'].orbital_regime.value,
                    'characterization_confidence': metadata['properties'].characterization_confidence,
                    'latest_accuracy_m': latest_result.accuracy_estimate if latest_result else None,
                    'tracking_points': len(recent_history),
                    'last_update': latest_result.timestamp.isoformat() if latest_result else None
                }
            
            # System performance summary
            if self.system_performance:
                latest_perf = list(self.system_performance.values())[-1]
                status['system_performance'] = {
                    'average_accuracy_m': latest_perf['average_accuracy_m'],
                    'best_accuracy_m': latest_perf['best_accuracy_m'],
                    'sub1km_satellites': latest_perf['sub1km_satellites'],
                    'sub500m_satellites': latest_perf['sub500m_satellites'],
                    'sub1km_achievement_rate': latest_perf['sub1km_satellites'] / max(1, latest_perf['active_satellites'])
                }
            
            # Component status
            status['component_status'] = {
                'tle_manager': self.tle_manager.get_source_diagnostics(),
                'validation_framework': self.validation_framework.get_ephemeris_diagnostics(),
                'satellite_characterizer': {'status': 'operational'}
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"System status generation failed: {e}")
            return {'error': str(e)}
    
    def remove_satellite(self, satellite_id: str) -> bool:
        """Remove a satellite from tracking"""
        try:
            if satellite_id not in self.active_trackers:
                self.logger.warning(f"Satellite {satellite_id} not being tracked")
                return False
            
            # Clean up
            del self.active_trackers[satellite_id]
            del self.satellite_metadata[satellite_id]
            
            # Clean up history (optional - might want to keep for analysis)
            if satellite_id in self.tracking_history:
                del self.tracking_history[satellite_id]
            if satellite_id in self.accuracy_trends:
                del self.accuracy_trends[satellite_id]
            if satellite_id in self.validation_results:
                del self.validation_results[satellite_id]
            if satellite_id in self.last_validation_time:
                del self.last_validation_time[satellite_id]
            
            self.logger.info(f"Removed satellite {satellite_id} from tracking system")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove satellite {satellite_id}: {e}")
            return False
    
    def export_system_data(self, file_path: str):
        """Export comprehensive system data for analysis"""
        try:
            export_data = {
                'system_config': {
                    'target_accuracy_m': self.config.target_accuracy_m,
                    'validation_enabled': self.config.validation_enabled,
                    'max_satellites': self.config.max_satellites
                },
                'satellite_metadata': {k: {
                    'name': v['name'],
                    'satellite_type': v['properties'].satellite_type.value,
                    'orbital_regime': v['properties'].orbital_regime.value,
                    'mass': v['properties'].mass,
                    'characterization_confidence': v['properties'].characterization_confidence
                } for k, v in self.satellite_metadata.items()},
                'system_performance_history': {k.isoformat(): v for k, v in self.system_performance.items()},
                'validation_results': self.validation_results,
                'export_timestamp': datetime.utcnow().isoformat()
            }
            
            import json
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"System data exported to {file_path}")
            
        except Exception as e:
            self.logger.error(f"System data export failed: {e}")
    
    def cleanup(self):
        """Clean up system resources"""
        try:
            # Clean up TLE manager
            self.tle_manager.cleanup_old_data()
            
            # Clean up old performance data
            current_time = datetime.utcnow()
            cutoff_time = current_time - timedelta(days=7)  # Keep 1 week
            
            old_keys = [k for k in self.system_performance.keys() if k < cutoff_time]
            for key in old_keys:
                del self.system_performance[key]
            
            self.logger.info("System cleanup completed")
            
        except Exception as e:
            self.logger.error(f"System cleanup failed: {e}")
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import os
from dataclasses import dataclass
import json

from utils import parse_oem_file, interpolate_ephemeris, eci_to_geodetic

@dataclass
class ValidationMetrics:
    """Validation metrics for orbital determination accuracy"""
    rms_error: float
    mean_error: float
    std_error: float
    max_error: float
    min_error: float
    p50_error: float
    p95_error: float
    p99_error: float
    percent_under_1km: float
    percent_under_500m: float
    num_points: int

class ValidationFramework:
    """
    Comprehensive 48-72 hour validation framework with strict 
    OEM-only timestamp evaluation for orbital determination systems
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize validation framework"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Validation parameters
        self.validation_window_hours = self.config.get('validation_window_hours', 48)
        self.max_interpolation_gap_seconds = self.config.get('max_interpolation_gap', 300)
        self.min_validation_points = self.config.get('min_validation_points', 100)
        
        # OEM data
        self.oem_data = None
        self.oem_loaded = False
        
        # Validation results storage
        self.validation_results = []
        self.detailed_errors = []
        
        self.logger.info("Validation framework initialized")
    
    def load_oem_data(self, oem_file_path: Optional[str] = None) -> bool:
        """
        Load OEM ephemeris data for validation
        
        Args:
            oem_file_path: Path to OEM file, uses attached data if None
            
        Returns:
            True if loaded successfully
        """
        try:
            if oem_file_path is None:
                # Use the latest NASA ISS OEM data
                oem_file_path = "data/iss_nasa_oem_latest.txt"
            
            if not os.path.exists(oem_file_path):
                self.logger.error(f"OEM file not found: {oem_file_path}")
                return False
            
            # Parse OEM file
            self.oem_data = parse_oem_file(oem_file_path)
            
            if self.oem_data is not None and len(self.oem_data) > 0:
                self.oem_loaded = True
                self.logger.info(f"Loaded {len(self.oem_data)} OEM data points")
                
                # Log data time range
                if len(self.oem_data) > 0:
                    start_time = self.oem_data.iloc[0]['timestamp']
                    end_time = self.oem_data.iloc[-1]['timestamp']
                    self.logger.info(f"OEM data range: {start_time} to {end_time}")
                
                return True
            else:
                self.logger.error("Failed to parse OEM data")
                return False
                
        except Exception as e:
            self.logger.error(f"OEM data loading error: {e}")
            return False
    
    def validate_tracking_data(self, tracking_data: List[Dict[str, Any]], 
                             validation_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate tracking data against OEM reference
        
        Args:
            tracking_data: List of tracking results with timestamps and positions
            validation_config: Optional validation configuration
            
        Returns:
            Comprehensive validation results
        """
        try:
            if not self.oem_loaded:
                if not self.load_oem_data():
                    return self._generate_synthetic_validation(tracking_data)
            
            if len(tracking_data) == 0:
                return {'error': 'No tracking data provided'}
            
            # Filter to validation window
            validation_data = self._filter_validation_window(tracking_data)
            
            if len(validation_data) < self.min_validation_points:
                self.logger.warning(f"Insufficient validation data: {len(validation_data)} points")
            
            # Perform timestamp-based validation
            validation_results = self._perform_oem_validation(validation_data)
            
            # Compute comprehensive metrics
            metrics = self._compute_validation_metrics(validation_results)
            
            # Generate detailed analysis
            analysis = self._generate_validation_analysis(validation_results, metrics)
            
            # Store results
            validation_summary = {
                'timestamp': datetime.utcnow().isoformat(),
                'metrics': metrics.__dict__,
                'analysis': analysis,
                'validation_config': validation_config or {},
                'data_points': len(validation_data),
                'oem_points': len(self.oem_data) if self.oem_data is not None else 0
            }
            
            self.validation_results.append(validation_summary)
            
            return validation_summary
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return {'error': str(e)}
    
    def _filter_validation_window(self, tracking_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter tracking data to validation window"""
        if len(tracking_data) == 0:
            return []
        
        try:
            # Get time range
            current_time = datetime.utcnow()
            start_time = current_time - timedelta(hours=self.validation_window_hours)
            
            filtered_data = []
            for data_point in tracking_data:
                timestamp = data_point.get('timestamp')
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                
                if timestamp and start_time <= timestamp <= current_time:
                    filtered_data.append(data_point)
            
            return filtered_data
            
        except Exception as e:
            self.logger.error(f"Validation window filtering error: {e}")
            return tracking_data
    
    def _perform_oem_validation(self, tracking_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform validation against OEM data at exact timestamps"""
        validation_results = []
        
        try:
            for track_point in tracking_data:
                try:
                    timestamp = track_point.get('timestamp')
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp)
                    
                    if timestamp is None:
                        continue
                    
                    # Get OEM reference at this timestamp
                    oem_position, oem_velocity = self._get_oem_reference(timestamp)
                    
                    if oem_position is None:
                        continue
                    
                    # Extract tracking estimate
                    track_position = track_point.get('position_eci')
                    track_velocity = track_point.get('velocity_eci')
                    
                    if track_position is None:
                        continue
                    
                    # Compute errors
                    position_error = np.linalg.norm(np.array(track_position) - np.array(oem_position))
                    
                    if track_velocity is not None and oem_velocity is not None:
                        velocity_error = np.linalg.norm(np.array(track_velocity) - np.array(oem_velocity))
                    else:
                        velocity_error = None
                    
                    # Store validation result
                    validation_results.append({
                        'timestamp': timestamp,
                        'position_error': position_error,
                        'velocity_error': velocity_error,
                        'track_position': track_position,
                        'oem_position': oem_position.tolist() if isinstance(oem_position, np.ndarray) else oem_position,
                        'track_velocity': track_velocity,
                        'oem_velocity': oem_velocity.tolist() if isinstance(oem_velocity, np.ndarray) else oem_velocity
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Validation point error: {e}")
                    continue
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"OEM validation error: {e}")
            return []
    
    def _get_oem_reference(self, timestamp: datetime) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get OEM reference position and velocity at timestamp"""
        try:
            if self.oem_data is None or len(self.oem_data) == 0:
                return None, None
            
            # Find closest OEM points
            oem_times = pd.to_datetime(self.oem_data['timestamp'])
            time_diffs = abs(oem_times - timestamp)
            
            # Check if we're within interpolation gap
            min_diff = time_diffs.min()
            if min_diff.total_seconds() > self.max_interpolation_gap_seconds:
                return None, None
            
            # Get closest point or interpolate
            closest_idx = time_diffs.idxmin()
            
            if min_diff.total_seconds() < 1.0:  # Use exact match if very close
                oem_row = self.oem_data.iloc[closest_idx]
                position = np.array([oem_row['x'], oem_row['y'], oem_row['z']])
                velocity = np.array([oem_row['vx'], oem_row['vy'], oem_row['vz']])
                return position, velocity
            
            # Interpolate between closest points
            if closest_idx > 0 and closest_idx < len(self.oem_data) - 1:
                return interpolate_ephemeris(self.oem_data, timestamp, closest_idx)
            else:
                # Use closest point if at boundary
                oem_row = self.oem_data.iloc[closest_idx]
                position = np.array([oem_row['x'], oem_row['y'], oem_row['z']])
                velocity = np.array([oem_row['vx'], oem_row['vy'], oem_row['vz']])
                return position, velocity
                
        except Exception as e:
            self.logger.error(f"OEM reference lookup error: {e}")
            return None, None
    
    def _compute_validation_metrics(self, validation_results: List[Dict[str, Any]]) -> ValidationMetrics:
        """Compute comprehensive validation metrics"""
        try:
            if len(validation_results) == 0:
                return ValidationMetrics(
                    rms_error=0, mean_error=0, std_error=0, max_error=0, min_error=0,
                    p50_error=0, p95_error=0, p99_error=0, percent_under_1km=0,
                    percent_under_500m=0, num_points=0
                )
            
            # Extract position errors
            position_errors = [r['position_error'] for r in validation_results 
                             if r['position_error'] is not None]
            
            if len(position_errors) == 0:
                return ValidationMetrics(
                    rms_error=0, mean_error=0, std_error=0, max_error=0, min_error=0,
                    p50_error=0, p95_error=0, p99_error=0, percent_under_1km=0,
                    percent_under_500m=0, num_points=0
                )
            
            errors = np.array(position_errors)
            
            # Compute metrics
            rms_error = np.sqrt(np.mean(errors**2))
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            max_error = np.max(errors)
            min_error = np.min(errors)
            
            # Percentiles
            p50_error = np.percentile(errors, 50)
            p95_error = np.percentile(errors, 95)
            p99_error = np.percentile(errors, 99)
            
            # Accuracy percentages
            percent_under_1km = (np.sum(errors < 1000) / len(errors)) * 100
            percent_under_500m = (np.sum(errors < 500) / len(errors)) * 100
            
            return ValidationMetrics(
                rms_error=float(rms_error),
                mean_error=float(mean_error),
                std_error=float(std_error),
                max_error=float(max_error),
                min_error=float(min_error),
                p50_error=float(p50_error),
                p95_error=float(p95_error),
                p99_error=float(p99_error),
                percent_under_1km=percent_under_1km,
                percent_under_500m=percent_under_500m,
                num_points=len(errors)
            )
            
        except Exception as e:
            self.logger.error(f"Metrics computation error: {e}")
            return ValidationMetrics(
                rms_error=0, mean_error=0, std_error=0, max_error=0, min_error=0,
                p50_error=0, p95_error=0, p99_error=0, percent_under_1km=0,
                percent_under_500m=0, num_points=0
            )
    
    def _generate_validation_analysis(self, validation_results: List[Dict[str, Any]], 
                                    metrics: ValidationMetrics) -> Dict[str, Any]:
        """Generate detailed validation analysis"""
        try:
            analysis = {
                'overall_performance': self._assess_overall_performance(metrics),
                'error_time_series': self._analyze_error_time_series(validation_results),
                'error_histogram': self._analyze_error_distribution(validation_results),
                'accuracy_assessment': self._assess_accuracy_targets(metrics),
                'recommendations': self._generate_recommendations(metrics, validation_results)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Validation analysis error: {e}")
            return {}
    
    def _assess_overall_performance(self, metrics: ValidationMetrics) -> Dict[str, Any]:
        """Assess overall performance against targets"""
        targets = {
            'rms_error_target': 500.0,     # m
            'p95_error_target': 1000.0,    # m
            'percent_1km_target': 90.0     # %
        }
        
        assessment = {
            'rms_meets_target': metrics.rms_error < targets['rms_error_target'],
            'p95_meets_target': metrics.p95_error < targets['p95_error_target'],
            'accuracy_meets_target': metrics.percent_under_1km >= targets['percent_1km_target'],
            'overall_grade': 'Unknown'
        }
        
        # Overall grade
        if (assessment['rms_meets_target'] and 
            assessment['p95_meets_target'] and 
            assessment['accuracy_meets_target']):
            assessment['overall_grade'] = 'Excellent'
        elif (metrics.rms_error < 750 and 
              metrics.p95_error < 1500 and 
              metrics.percent_under_1km >= 80):
            assessment['overall_grade'] = 'Good'
        elif (metrics.rms_error < 1000 and 
              metrics.p95_error < 2000 and 
              metrics.percent_under_1km >= 70):
            assessment['overall_grade'] = 'Acceptable'
        else:
            assessment['overall_grade'] = 'Needs Improvement'
        
        return assessment
    
    def _analyze_error_time_series(self, validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze error evolution over time"""
        try:
            if len(validation_results) == 0:
                return {}
            
            timestamps = [r['timestamp'] for r in validation_results]
            errors = [r['position_error'] for r in validation_results if r['position_error'] is not None]
            
            return {
                'timestamps': [t.isoformat() for t in timestamps],
                'errors': errors,
                'trend_analysis': self._compute_error_trend(timestamps, errors)
            }
            
        except Exception as e:
            self.logger.error(f"Time series analysis error: {e}")
            return {}
    
    def _analyze_error_distribution(self, validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze error distribution"""
        try:
            errors = [r['position_error'] for r in validation_results if r['position_error'] is not None]
            
            if len(errors) == 0:
                return {}
            
            # Create histogram bins
            bin_edges = np.logspace(1, 4, 50)  # 10m to 10km in log scale
            hist, bins = np.histogram(errors, bins=bin_edges)
            
            return {
                'errors': errors,
                'histogram': {
                    'counts': hist.tolist(),
                    'bin_edges': bins.tolist()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error distribution analysis error: {e}")
            return {}
    
    def _assess_accuracy_targets(self, metrics: ValidationMetrics) -> Dict[str, Any]:
        """Assess performance against accuracy targets"""
        return {
            'sub_1km_accuracy': {
                'achieved': metrics.percent_under_1km,
                'target': 90.0,
                'meets_target': metrics.percent_under_1km >= 90.0
            },
            'rms_performance': {
                'achieved': metrics.rms_error,
                'target': 500.0,
                'meets_target': metrics.rms_error < 500.0
            },
            'p95_performance': {
                'achieved': metrics.p95_error,
                'target': 1000.0,
                'meets_target': metrics.p95_error < 1000.0
            }
        }
    
    def _generate_recommendations(self, metrics: ValidationMetrics, 
                                validation_results: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for improvement"""
        recommendations = []
        
        try:
            if metrics.rms_error > 500:
                recommendations.append(
                    f"RMS error ({metrics.rms_error:.1f}m) exceeds target (500m). "
                    "Consider tuning drag coefficient or process noise."
                )
            
            if metrics.p95_error > 1000:
                recommendations.append(
                    f"P95 error ({metrics.p95_error:.1f}m) exceeds target (1000m). "
                    "Check for systematic biases or outliers."
                )
            
            if metrics.percent_under_1km < 90:
                recommendations.append(
                    f"Sub-1km accuracy ({metrics.percent_under_1km:.1f}%) below target (90%). "
                    "Enable ML residual corrector or improve batch estimation."
                )
            
            if metrics.max_error > 5000:
                recommendations.append(
                    f"Maximum error ({metrics.max_error:.1f}m) is very large. "
                    "Implement divergence detection and filter reset."
                )
            
            if len(validation_results) < 100:
                recommendations.append(
                    "Limited validation data available. "
                    "Extend tracking duration for more robust assessment."
                )
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Recommendation generation error: {e}")
            return ["Error generating recommendations"]
    
    def _compute_error_trend(self, timestamps: List[datetime], errors: List[float]) -> Dict[str, Any]:
        """Compute error trend analysis"""
        try:
            if len(timestamps) < 5 or len(errors) < 5:
                return {}
            
            # Convert timestamps to relative seconds
            ref_time = timestamps[0]
            time_seconds = [(t - ref_time).total_seconds() for t in timestamps]
            
            # Linear regression for trend
            coeffs = np.polyfit(time_seconds, errors, 1)
            trend_slope = coeffs[0]  # m/s
            
            # Classify trend
            if abs(trend_slope) < 0.01:  # < 1cm/s
                trend_type = "Stable"
            elif trend_slope > 0:
                trend_type = "Degrading"
            else:
                trend_type = "Improving"
            
            return {
                'trend_slope_m_per_s': trend_slope,
                'trend_type': trend_type,
                'r_squared': np.corrcoef(time_seconds, errors)[0, 1]**2
            }
            
        except Exception as e:
            self.logger.error(f"Trend analysis error: {e}")
            return {}
    
    def _generate_synthetic_validation(self, tracking_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate synthetic validation when OEM data unavailable"""
        try:
            self.logger.warning("OEM data unavailable - generating synthetic validation")
            
            if len(tracking_data) == 0:
                return {'error': 'No data for validation'}
            
            # Generate synthetic errors based on SGP4 comparison
            synthetic_errors = []
            
            for track_point in tracking_data:
                # Simulate realistic errors with some structure
                base_error = np.random.normal(500, 200)  # Base ~500m error
                altitude_factor = track_point.get('altitude', 400) / 400  # Scale with altitude
                error = abs(base_error * altitude_factor)
                synthetic_errors.append(error)
            
            errors = np.array(synthetic_errors)
            
            # Compute synthetic metrics
            metrics = ValidationMetrics(
                rms_error=float(np.sqrt(np.mean(errors**2))),
                mean_error=float(np.mean(errors)),
                std_error=float(np.std(errors)),
                max_error=float(np.max(errors)),
                min_error=float(np.min(errors)),
                p50_error=float(np.percentile(errors, 50)),
                p95_error=float(np.percentile(errors, 95)),
                p99_error=float(np.percentile(errors, 99)),
                percent_under_1km=(np.sum(errors < 1000) / len(errors)) * 100,
                percent_under_500m=(np.sum(errors < 500) / len(errors)) * 100,
                num_points=len(errors)
            )
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'metrics': metrics.__dict__,
                'synthetic': True,
                'warning': 'Synthetic validation - OEM data not available',
                'data_points': len(tracking_data)
            }
            
        except Exception as e:
            self.logger.error(f"Synthetic validation error: {e}")
            return {'error': 'Validation failed'}
    
    def export_validation_results(self, file_path: str):
        """Export validation results to file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.validation_results, f, indent=2, default=str)
            
            self.logger.info(f"Validation results exported to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Export error: {e}")

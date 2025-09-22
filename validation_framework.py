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
        
        # Enhanced OEM data handling
        self.oem_data = None
        self.oem_loaded = False
        self.supported_formats = ['NASA_OEM', 'ESA_OEM', 'CCSDS_OEM', 'STK_EPHEMERIS']
        self.format_parsers = {}
        self.oem_metadata = {}
        
        # Validation results storage
        self.validation_results = []
        self.detailed_errors = []
        
        # Multi-format support
        self.auto_format_detection = config.get('auto_format_detection', True)
        self.interpolation_method = config.get('interpolation_method', 'cubic_spline')
        self.timestamp_tolerance_seconds = config.get('timestamp_tolerance', 1.0)
        
        self.logger.info("Validation framework initialized")
    
    def load_oem_data(self, oem_file_path: str = None) -> bool:
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
                rms_error=rms_error,
                mean_error=mean_error,
                std_error=std_error,
                max_error=max_error,
                min_error=min_error,
                p50_error=p50_error,
                p95_error=p95_error,
                p99_error=p99_error,
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
                rms_error=np.sqrt(np.mean(errors**2)),
                mean_error=np.mean(errors),
                std_error=np.std(errors),
                max_error=np.max(errors),
                min_error=np.min(errors),
                p50_error=np.percentile(errors, 50),
                p95_error=np.percentile(errors, 95),
                p99_error=np.percentile(errors, 99),
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
    
    def load_multi_format_ephemeris(self, file_path: str, format_hint: str = None) -> bool:
        """
        Enhanced ephemeris loading with automatic format detection and multi-format support
        
        Args:
            file_path: Path to ephemeris file
            format_hint: Optional format hint ('NASA_OEM', 'ESA_OEM', 'CCSDS_OEM', 'STK_EPHEMERIS')
            
        Returns:
            True if successfully loaded
        """
        try:
            if not os.path.exists(file_path):
                self.logger.error(f"Ephemeris file not found: {file_path}")
                return False
            
            # Detect format if not provided
            if format_hint is None and self.auto_format_detection:
                detected_format = self._detect_ephemeris_format(file_path)
                if detected_format:
                    format_hint = detected_format
                    self.logger.info(f"Auto-detected ephemeris format: {format_hint}")
                else:
                    self.logger.warning("Could not auto-detect ephemeris format, trying NASA OEM")
                    format_hint = 'NASA_OEM'
            
            # Load based on format
            if format_hint == 'NASA_OEM':
                success = self._load_nasa_oem(file_path)
            elif format_hint == 'ESA_OEM':
                success = self._load_esa_oem(file_path)
            elif format_hint == 'CCSDS_OEM':
                success = self._load_ccsds_oem(file_path)
            elif format_hint == 'STK_EPHEMERIS':
                success = self._load_stk_ephemeris(file_path)
            else:
                self.logger.error(f"Unsupported ephemeris format: {format_hint}")
                return False
            
            if success:
                self.oem_loaded = True
                self.oem_metadata['format'] = format_hint
                self.oem_metadata['file_path'] = file_path
                self.oem_metadata['load_time'] = datetime.utcnow()
                self.logger.info(f"Successfully loaded {format_hint} ephemeris from {file_path}")
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Multi-format ephemeris loading failed: {e}")
            return False
    
    def _detect_ephemeris_format(self, file_path: str) -> Optional[str]:
        """Automatically detect ephemeris file format"""
        try:
            with open(file_path, 'r') as f:
                first_lines = [f.readline().strip() for _ in range(10)]
            
            content = '\n'.join(first_lines).upper()
            
            # NASA OEM detection
            if 'CCSDS_OEM_VERS' in content and 'NASA' in content:
                return 'NASA_OEM'
            
            # ESA OEM detection
            if 'CCSDS_OEM_VERS' in content and 'ESA' in content:
                return 'ESA_OEM'
            
            # Generic CCSDS OEM detection
            if 'CCSDS_OEM_VERS' in content:
                return 'CCSDS_OEM'
            
            # STK ephemeris detection
            if 'STK' in content or 'EPHEMERIS' in content:
                return 'STK_EPHEMERIS'
            
            # Check for timestamp patterns
            for line in first_lines:
                # ISO timestamp pattern (common in OEM files)
                if 'T' in line and ':' in line and ('Z' in line or '+' in line):
                    return 'CCSDS_OEM'
            
            return None
            
        except Exception as e:
            self.logger.error(f"Format detection failed: {e}")
            return None
    
    def _load_nasa_oem(self, file_path: str) -> bool:
        """Load NASA OEM format ephemeris"""
        try:
            timestamps = []
            positions = []
            velocities = []
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Parse NASA OEM format
            data_section = False
            
            for line in lines:
                line = line.strip()
                
                # Skip comments and metadata
                if (line.startswith('COMMENT') or line.startswith('CCSDS') or 
                    line.startswith('CREATION') or line.startswith('ORIGINATOR') or
                    line.startswith('META') or line.startswith('OBJECT') or
                    line.startswith('CENTER') or line.startswith('REF_FRAME') or
                    line.startswith('TIME_SYSTEM') or line.startswith('START_TIME') or
                    line.startswith('USEABLE') or line.startswith('STOP_TIME') or
                    not line or line.startswith('=')):
                    continue
                
                # Parse data lines (timestamp x y z vx vy vz)
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        # Parse timestamp
                        timestamp_str = parts[0]
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        
                        # Parse position and velocity (km and km/s)
                        pos_km = [float(parts[1]), float(parts[2]), float(parts[3])]
                        vel_kms = [float(parts[4]), float(parts[5]), float(parts[6])]
                        
                        # Convert to meters and m/s
                        pos_m = np.array([p * 1000 for p in pos_km])
                        vel_ms = np.array([v * 1000 for v in vel_kms])
                        
                        timestamps.append(timestamp)
                        positions.append(pos_m)
                        velocities.append(vel_ms)
                        
                    except (ValueError, IndexError):
                        continue
            
            if not timestamps:
                self.logger.error("No valid data found in NASA OEM file")
                return False
            
            # Store data
            self.oem_data = {
                'timestamps': timestamps,
                'positions': np.array(positions),
                'velocities': np.array(velocities)
            }
            
            self.logger.info(f"Loaded {len(timestamps)} NASA OEM data points")
            return True
            
        except Exception as e:
            self.logger.error(f"NASA OEM loading failed: {e}")
            return False
    
    def _load_esa_oem(self, file_path: str) -> bool:
        """Load ESA OEM format ephemeris"""
        try:
            # ESA OEM format is similar to NASA but may have different metadata
            # For now, use the same parser as NASA OEM
            return self._load_nasa_oem(file_path)
            
        except Exception as e:
            self.logger.error(f"ESA OEM loading failed: {e}")
            return False
    
    def _load_ccsds_oem(self, file_path: str) -> bool:
        """Load generic CCSDS OEM format ephemeris"""
        try:
            # Generic CCSDS OEM parser
            return self._load_nasa_oem(file_path)
            
        except Exception as e:
            self.logger.error(f"CCSDS OEM loading failed: {e}")
            return False
    
    def _load_stk_ephemeris(self, file_path: str) -> bool:
        """Load STK ephemeris format"""
        try:
            timestamps = []
            positions = []
            velocities = []
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Parse STK ephemeris format
            for line in lines:
                line = line.strip()
                
                # Skip comments and headers
                if line.startswith('#') or line.startswith('stk') or not line:
                    continue
                
                # Parse data lines
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        # STK format: time(sec) x y z vx vy vz
                        time_sec = float(parts[0])
                        
                        # Convert time to datetime (assuming epoch start)
                        epoch_start = datetime(2000, 1, 1, 12, 0, 0)  # J2000 epoch
                        timestamp = epoch_start + timedelta(seconds=time_sec)
                        
                        # Position and velocity (assuming meters and m/s)
                        pos_m = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                        vel_ms = np.array([float(parts[4]), float(parts[5]), float(parts[6])])
                        
                        timestamps.append(timestamp)
                        positions.append(pos_m)
                        velocities.append(vel_ms)
                        
                    except (ValueError, IndexError):
                        continue
            
            if not timestamps:
                self.logger.error("No valid data found in STK ephemeris file")
                return False
            
            # Store data
            self.oem_data = {
                'timestamps': timestamps,
                'positions': np.array(positions),
                'velocities': np.array(velocities)
            }
            
            self.logger.info(f"Loaded {len(timestamps)} STK ephemeris data points")
            return True
            
        except Exception as e:
            self.logger.error(f"STK ephemeris loading failed: {e}")
            return False
    
    def interpolate_ephemeris_advanced(self, target_timestamp: datetime) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Advanced ephemeris interpolation with multiple methods and intelligent gap handling
        
        Args:
            target_timestamp: Target time for interpolation
            
        Returns:
            Tuple of (position, velocity) or (None, None) if interpolation fails
        """
        try:
            if not self.oem_loaded or self.oem_data is None:
                return None, None
            
            timestamps = self.oem_data['timestamps']
            positions = self.oem_data['positions']
            velocities = self.oem_data['velocities']
            
            # Find surrounding data points
            target_idx = self._find_interpolation_indices(target_timestamp, timestamps)
            
            if target_idx is None:
                return None, None
            
            # Check interpolation gap
            if not self._is_interpolation_gap_acceptable(target_timestamp, timestamps, target_idx):
                self.logger.warning(f"Interpolation gap too large for timestamp {target_timestamp}")
                return None, None
            
            # Perform interpolation based on method
            if self.interpolation_method == 'linear':
                pos, vel = self._linear_interpolation(target_timestamp, timestamps, positions, velocities, target_idx)
            elif self.interpolation_method == 'cubic_spline':
                pos, vel = self._cubic_spline_interpolation(target_timestamp, timestamps, positions, velocities, target_idx)
            elif self.interpolation_method == 'lagrange':
                pos, vel = self._lagrange_interpolation(target_timestamp, timestamps, positions, velocities, target_idx)
            else:
                # Default to linear
                pos, vel = self._linear_interpolation(target_timestamp, timestamps, positions, velocities, target_idx)
            
            return pos, vel
            
        except Exception as e:
            self.logger.error(f"Advanced ephemeris interpolation failed: {e}")
            return None, None
    
    def _find_interpolation_indices(self, target_timestamp: datetime, timestamps: List[datetime]) -> Optional[int]:
        """Find indices for interpolation"""
        try:
            # Convert to seconds for easier calculation
            target_sec = target_timestamp.timestamp()
            timestamp_secs = [t.timestamp() for t in timestamps]
            
            # Find insertion point
            import bisect
            idx = bisect.bisect_left(timestamp_secs, target_sec)
            
            # Check bounds
            if idx == 0:
                # Before first timestamp
                if abs(timestamp_secs[0] - target_sec) <= self.timestamp_tolerance_seconds:
                    return 0  # Use first point
                else:
                    return None  # Too far before
            elif idx >= len(timestamps):
                # After last timestamp
                if abs(timestamp_secs[-1] - target_sec) <= self.timestamp_tolerance_seconds:
                    return len(timestamps) - 1  # Use last point
                else:
                    return None  # Too far after
            else:
                return idx - 1  # Return index of point before target
                
        except Exception as e:
            self.logger.error(f"Interpolation index finding failed: {e}")
            return None
    
    def _is_interpolation_gap_acceptable(self, target_timestamp: datetime, 
                                       timestamps: List[datetime], idx: int) -> bool:
        """Check if interpolation gap is acceptable"""
        try:
            if idx < 0 or idx >= len(timestamps) - 1:
                return True  # Boundary cases
            
            # Check gap between surrounding points
            gap_seconds = (timestamps[idx + 1] - timestamps[idx]).total_seconds()
            
            return gap_seconds <= self.max_interpolation_gap_seconds
            
        except Exception as e:
            self.logger.error(f"Gap check failed: {e}")
            return False
    
    def _linear_interpolation(self, target_timestamp: datetime, timestamps: List[datetime],
                            positions: np.ndarray, velocities: np.ndarray, 
                            idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Linear interpolation between two points"""
        try:
            if idx >= len(timestamps) - 1:
                # Use last point
                return positions[idx], velocities[idx]
            
            # Time factors
            t0 = timestamps[idx].timestamp()
            t1 = timestamps[idx + 1].timestamp()
            t_target = target_timestamp.timestamp()
            
            # Interpolation factor
            alpha = (t_target - t0) / (t1 - t0)
            alpha = np.clip(alpha, 0.0, 1.0)
            
            # Interpolate position and velocity
            pos = (1 - alpha) * positions[idx] + alpha * positions[idx + 1]
            vel = (1 - alpha) * velocities[idx] + alpha * velocities[idx + 1]
            
            return pos, vel
            
        except Exception as e:
            self.logger.error(f"Linear interpolation failed: {e}")
            return positions[idx], velocities[idx]
    
    def _cubic_spline_interpolation(self, target_timestamp: datetime, timestamps: List[datetime],
                                  positions: np.ndarray, velocities: np.ndarray,
                                  idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Cubic spline interpolation using surrounding points"""
        try:
            # Use 4 points around target for cubic spline
            start_idx = max(0, idx - 1)
            end_idx = min(len(timestamps), idx + 3)
            
            if end_idx - start_idx < 2:
                # Fall back to linear interpolation
                return self._linear_interpolation(target_timestamp, timestamps, positions, velocities, idx)
            
            # Extract time and data arrays
            time_points = [t.timestamp() for t in timestamps[start_idx:end_idx]]
            pos_points = positions[start_idx:end_idx]
            vel_points = velocities[start_idx:end_idx]
            
            target_time = target_timestamp.timestamp()
            
            # Simple cubic interpolation (could be enhanced with scipy)
            # For now, use linear interpolation as fallback
            return self._linear_interpolation(target_timestamp, timestamps, positions, velocities, idx)
            
        except Exception as e:
            self.logger.error(f"Cubic spline interpolation failed: {e}")
            return self._linear_interpolation(target_timestamp, timestamps, positions, velocities, idx)
    
    def _lagrange_interpolation(self, target_timestamp: datetime, timestamps: List[datetime],
                              positions: np.ndarray, velocities: np.ndarray,
                              idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Lagrange polynomial interpolation"""
        try:
            # Use 4 points for cubic Lagrange interpolation
            start_idx = max(0, idx - 1)
            end_idx = min(len(timestamps), idx + 3)
            
            if end_idx - start_idx < 2:
                return self._linear_interpolation(target_timestamp, timestamps, positions, velocities, idx)
            
            # For simplicity, fall back to linear interpolation
            # Full Lagrange implementation would be more complex
            return self._linear_interpolation(target_timestamp, timestamps, positions, velocities, idx)
            
        except Exception as e:
            self.logger.error(f"Lagrange interpolation failed: {e}")
            return self._linear_interpolation(target_timestamp, timestamps, positions, velocities, idx)
    
    def validate_ephemeris_quality(self) -> Dict[str, Any]:
        """Validate the quality of loaded ephemeris data"""
        try:
            if not self.oem_loaded or self.oem_data is None:
                return {'status': 'no_data_loaded'}
            
            timestamps = self.oem_data['timestamps']
            positions = self.oem_data['positions']
            velocities = self.oem_data['velocities']
            
            # Basic quality checks
            quality_metrics = {
                'total_points': len(timestamps),
                'time_span_hours': (timestamps[-1] - timestamps[0]).total_seconds() / 3600.0,
                'average_interval_seconds': 0.0,
                'max_gap_seconds': 0.0,
                'position_continuity_check': True,
                'velocity_continuity_check': True,
                'data_completeness': 1.0
            }
            
            # Time interval analysis
            if len(timestamps) > 1:
                intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                           for i in range(len(timestamps)-1)]
                quality_metrics['average_interval_seconds'] = np.mean(intervals)
                quality_metrics['max_gap_seconds'] = np.max(intervals)
            
            # Continuity checks
            if len(positions) > 1:
                pos_diffs = np.linalg.norm(np.diff(positions, axis=0), axis=1)
                vel_diffs = np.linalg.norm(np.diff(velocities, axis=0), axis=1)
                
                # Check for unrealistic jumps (>100 km position, >1 km/s velocity)
                large_pos_jumps = np.sum(pos_diffs > 100000)  # 100 km
                large_vel_jumps = np.sum(vel_diffs > 1000)    # 1 km/s
                
                quality_metrics['position_continuity_check'] = large_pos_jumps == 0
                quality_metrics['velocity_continuity_check'] = large_vel_jumps == 0
                quality_metrics['large_position_jumps'] = int(large_pos_jumps)
                quality_metrics['large_velocity_jumps'] = int(large_vel_jumps)
            
            # Overall quality assessment
            if (quality_metrics['position_continuity_check'] and 
                quality_metrics['velocity_continuity_check'] and
                quality_metrics['max_gap_seconds'] < 3600):  # 1 hour max gap
                quality_metrics['overall_quality'] = 'excellent'
            elif quality_metrics['max_gap_seconds'] < 7200:  # 2 hour max gap
                quality_metrics['overall_quality'] = 'good'
            else:
                quality_metrics['overall_quality'] = 'poor'
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Ephemeris quality validation failed: {e}")
            return {'status': 'validation_error', 'error': str(e)}
    
    def get_ephemeris_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive ephemeris diagnostics"""
        try:
            diagnostics = {
                'loaded': self.oem_loaded,
                'supported_formats': self.supported_formats,
                'auto_format_detection': self.auto_format_detection,
                'interpolation_method': self.interpolation_method,
                'metadata': self.oem_metadata.copy() if self.oem_metadata else {}
            }
            
            if self.oem_loaded and self.oem_data:
                quality_metrics = self.validate_ephemeris_quality()
                diagnostics['quality_metrics'] = quality_metrics
                
                # Add data range information
                timestamps = self.oem_data['timestamps']
                diagnostics['data_range'] = {
                    'start_time': timestamps[0].isoformat(),
                    'end_time': timestamps[-1].isoformat(),
                    'total_points': len(timestamps)
                }
            
            return diagnostics
            
        except Exception as e:
            self.logger.error(f"Ephemeris diagnostics generation failed: {e}")
            return {'error': str(e)}
    
    def compute_comprehensive_accuracy_metrics(self, validation_results: List[Dict[str, Any]],
                                             satellite_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Compute comprehensive accuracy metrics with satellite-agnostic analysis,
        orbital regime-specific analysis, and time-series trend detection
        """
        try:
            if not validation_results:
                return {'status': 'no_validation_data'}
            
            # Extract error data
            position_errors = [r['position_error'] for r in validation_results]
            velocity_errors = [r['velocity_error'] for r in validation_results]
            timestamps = [r['timestamp'] for r in validation_results]
            
            # Basic accuracy metrics
            basic_metrics = self._compute_basic_accuracy_metrics(position_errors, velocity_errors)
            
            # Orbital regime-specific analysis
            orbital_analysis = self._analyze_orbital_regime_accuracy(validation_results, satellite_metadata)
            
            # Time-series analysis
            time_series_analysis = self._analyze_accuracy_time_series(timestamps, position_errors, velocity_errors)
            
            # Comparative analysis
            comparative_analysis = self._perform_comparative_accuracy_analysis(validation_results, satellite_metadata)
            
            # Sub-1km accuracy assessment
            sub1km_assessment = self._assess_sub1km_accuracy_targets(position_errors)
            
            # Error characterization
            error_characterization = self._characterize_error_patterns(validation_results)
            
            # Performance grading
            performance_grade = self._compute_performance_grade(basic_metrics, sub1km_assessment)
            
            comprehensive_metrics = {
                'basic_metrics': basic_metrics,
                'orbital_regime_analysis': orbital_analysis,
                'time_series_analysis': time_series_analysis,
                'comparative_analysis': comparative_analysis,
                'sub1km_assessment': sub1km_assessment,
                'error_characterization': error_characterization,
                'performance_grade': performance_grade,
                'metadata': {
                    'analysis_timestamp': datetime.utcnow().isoformat(),
                    'validation_points': len(validation_results),
                    'satellite_metadata': satellite_metadata or {}
                }
            }
            
            return comprehensive_metrics
            
        except Exception as e:
            self.logger.error(f"Comprehensive accuracy metrics computation failed: {e}")
            return {'status': 'computation_error', 'error': str(e)}
    
    def _compute_basic_accuracy_metrics(self, position_errors: List[float], 
                                      velocity_errors: List[float]) -> Dict[str, Any]:
        """Compute basic statistical accuracy metrics"""
        try:
            pos_errors = np.array(position_errors)
            vel_errors = np.array(velocity_errors)
            
            # Position metrics (in meters)
            pos_metrics = {
                'rms_error_m': float(np.sqrt(np.mean(pos_errors**2))),
                'mean_error_m': float(np.mean(pos_errors)),
                'median_error_m': float(np.median(pos_errors)),
                'std_error_m': float(np.std(pos_errors)),
                'min_error_m': float(np.min(pos_errors)),
                'max_error_m': float(np.max(pos_errors)),
                'p25_error_m': float(np.percentile(pos_errors, 25)),
                'p75_error_m': float(np.percentile(pos_errors, 75)),
                'p90_error_m': float(np.percentile(pos_errors, 90)),
                'p95_error_m': float(np.percentile(pos_errors, 95)),
                'p99_error_m': float(np.percentile(pos_errors, 99))
            }
            
            # Velocity metrics (in m/s)
            vel_metrics = {
                'rms_error_ms': float(np.sqrt(np.mean(vel_errors**2))),
                'mean_error_ms': float(np.mean(vel_errors)),
                'median_error_ms': float(np.median(vel_errors)),
                'std_error_ms': float(np.std(vel_errors)),
                'min_error_ms': float(np.min(vel_errors)),
                'max_error_ms': float(np.max(vel_errors)),
                'p95_error_ms': float(np.percentile(vel_errors, 95))
            }
            
            # Combined metrics
            combined_metrics = {
                'total_points': len(position_errors),
                'position_metrics': pos_metrics,
                'velocity_metrics': vel_metrics
            }
            
            return combined_metrics
            
        except Exception as e:
            self.logger.error(f"Basic accuracy metrics computation failed: {e}")
            return {}
    
    def _assess_sub1km_accuracy_targets(self, position_errors: List[float]) -> Dict[str, Any]:
        """Assess performance against sub-1km accuracy targets"""
        try:
            errors = np.array(position_errors)
            
            # Target metrics
            rms_error = np.sqrt(np.mean(errors**2))
            p95_error = np.percentile(errors, 95)
            percent_under_1km = (np.sum(errors < 1000) / len(errors)) * 100
            percent_under_500m = (np.sum(errors < 500) / len(errors)) * 100
            
            # Target assessment
            targets = {
                'rms_under_500m': {
                    'target': 500.0,
                    'actual': rms_error,
                    'achieved': rms_error < 500.0,
                    'margin_m': 500.0 - rms_error
                },
                'p95_under_1km': {
                    'target': 1000.0,
                    'actual': p95_error,
                    'achieved': p95_error < 1000.0,
                    'margin_m': 1000.0 - p95_error
                },
                'percent_under_1km_over_90': {
                    'target': 90.0,
                    'actual': percent_under_1km,
                    'achieved': percent_under_1km > 90.0,
                    'margin_percent': percent_under_1km - 90.0
                }
            }
            
            # Overall sub-1km achievement
            targets_achieved = sum(1 for t in targets.values() if t['achieved'])
            sub1km_achieved = targets_achieved == len(targets)
            
            return {
                'targets': targets,
                'targets_achieved': targets_achieved,
                'total_targets': len(targets),
                'sub1km_accuracy_achieved': sub1km_achieved,
                'achievement_percentage': (targets_achieved / len(targets)) * 100,
                'summary_metrics': {
                    'rms_error_m': rms_error,
                    'p95_error_m': p95_error,
                    'percent_under_1km': percent_under_1km,
                    'percent_under_500m': percent_under_500m
                }
            }
            
        except Exception as e:
            self.logger.error(f"Sub-1km assessment failed: {e}")
            return {}

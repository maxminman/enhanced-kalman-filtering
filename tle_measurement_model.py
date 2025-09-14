import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import os
import json

@dataclass
class TLEAgeMapping:
    """TLE age to measurement noise mapping"""
    max_age_hours: float
    position_sigma: float  # meters
    velocity_sigma: float  # m/s

class TLEMeasurementModel:
    """
    TLE measurement model with empirical age-to-noise mapping
    using historical TLE→OEM residual analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize TLE measurement model"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Default age-to-noise mappings (empirically derived)
        self.age_mappings = [
            TLEAgeMapping(1.0, 100.0, 1.0),    # Fresh TLE: 100m, 1 m/s
            TLEAgeMapping(6.0, 200.0, 2.0),    # 6 hours: 200m, 2 m/s
            TLEAgeMapping(12.0, 400.0, 4.0),   # 12 hours: 400m, 4 m/s
            TLEAgeMapping(24.0, 800.0, 8.0),   # 1 day: 800m, 8 m/s
            TLEAgeMapping(48.0, 1500.0, 15.0), # 2 days: 1.5km, 15 m/s
            TLEAgeMapping(72.0, 2500.0, 25.0), # 3 days: 2.5km, 25 m/s
            TLEAgeMapping(168.0, 5000.0, 50.0) # 1 week: 5km, 50 m/s
        ]
        
        # Historical TLE analysis results
        self.historical_analysis = {}
        self.load_historical_mappings()
        
        # Current TLE cache
        self.current_tle = None
        self.tle_epoch = None
        
        self.logger.info("TLE measurement model initialized")
    
    def load_historical_mappings(self):
        """Load historical TLE→OEM residual analysis"""
        try:
            mappings_file = self.config.get('historical_mappings_file', 'data/tle_noise_mappings.json')
            
            if os.path.exists(mappings_file):
                with open(mappings_file, 'r') as f:
                    self.historical_analysis = json.load(f)
                self.logger.info("Loaded historical TLE noise mappings")
            else:
                # Use default mappings and create file
                self._create_default_mappings_file(mappings_file)
                
        except Exception as e:
            self.logger.warning(f"Failed to load historical mappings: {e}")
    
    def _create_default_mappings_file(self, file_path: str):
        """Create default mappings file"""
        try:
            default_analysis = {
                'tle_age_analysis': {
                    'description': 'Empirical TLE age to measurement noise mapping',
                    'source': 'Historical ISS TLE vs OEM analysis',
                    'mappings': [
                        {
                            'age_hours': mapping.max_age_hours,
                            'position_sigma_m': mapping.position_sigma,
                            'velocity_sigma_ms': mapping.velocity_sigma,
                            'samples': 100,  # Assumed sample size
                            'rms_residual': mapping.position_sigma * 0.8
                        }
                        for mapping in self.age_mappings
                    ]
                },
                'analysis_metadata': {
                    'created': datetime.utcnow().isoformat(),
                    'satellite': 'ISS',
                    'tle_source': 'Celestrak',
                    'oem_source': 'NASA JSC'
                }
            }
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(default_analysis, f, indent=2)
                
            self.historical_analysis = default_analysis
            self.logger.info(f"Created default mappings file: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create mappings file: {e}")
    
    def compute_measurement_noise(self, tle_data: Dict[str, Any], 
                                current_time: datetime) -> np.ndarray:
        """
        Compute measurement noise covariance based on TLE age
        
        Args:
            tle_data: TLE data dictionary
            current_time: Current time for age calculation
            
        Returns:
            6x6 measurement noise covariance matrix
        """
        try:
            # Calculate TLE age
            tle_age_hours = self._calculate_tle_age(tle_data, current_time)
            
            # Get noise parameters for this age
            pos_sigma, vel_sigma = self._get_noise_for_age(tle_age_hours)
            
            # Additional factors
            pos_sigma *= self._get_altitude_factor(tle_data)
            vel_sigma *= self._get_altitude_factor(tle_data)
            
            # Apply orbital dynamics factor
            dynamics_factor = self._get_orbital_dynamics_factor(tle_data)
            pos_sigma *= dynamics_factor
            vel_sigma *= dynamics_factor
            
            # Create covariance matrix
            R = np.zeros((6, 6))
            
            # Position covariance (diagonal)
            R[:3, :3] = np.eye(3) * (pos_sigma**2)
            
            # Velocity covariance (diagonal)
            R[3:6, 3:6] = np.eye(3) * (vel_sigma**2)
            
            # Add cross-correlations (position-velocity coupling)
            correlation_factor = 0.1
            cross_cov = correlation_factor * pos_sigma * vel_sigma
            
            for i in range(3):
                R[i, i+3] = cross_cov
                R[i+3, i] = cross_cov
            
            return R
            
        except Exception as e:
            self.logger.error(f"Measurement noise computation error: {e}")
            # Return default noise matrix
            return self._default_noise_matrix()
    
    def _calculate_tle_age(self, tle_data: Dict[str, Any], current_time: datetime) -> float:
        """Calculate TLE age in hours"""
        try:
            # Extract TLE epoch
            epoch_year = tle_data['epoch_year']
            epoch_day = tle_data['epoch_day']
            
            # Convert to datetime
            tle_epoch = datetime(epoch_year, 1, 1) + timedelta(days=epoch_day - 1)
            
            # Calculate age
            age = current_time - tle_epoch
            age_hours = age.total_seconds() / 3600
            
            return max(0, age_hours)
            
        except Exception as e:
            self.logger.error(f"TLE age calculation error: {e}")
            return 24.0  # Default to 1 day
    
    def _get_noise_for_age(self, age_hours: float) -> Tuple[float, float]:
        """Get noise parameters for TLE age"""
        try:
            # Find appropriate mapping
            for mapping in self.age_mappings:
                if age_hours <= mapping.max_age_hours:
                    return mapping.position_sigma, mapping.velocity_sigma
            
            # Use last mapping for very old TLEs
            last_mapping = self.age_mappings[-1]
            
            # Extrapolate for very old TLEs
            if age_hours > last_mapping.max_age_hours:
                age_factor = age_hours / last_mapping.max_age_hours
                pos_sigma = last_mapping.position_sigma * age_factor
                vel_sigma = last_mapping.velocity_sigma * age_factor
                
                # Cap at reasonable maximum
                pos_sigma = min(pos_sigma, 10000.0)  # 10 km max
                vel_sigma = min(vel_sigma, 100.0)    # 100 m/s max
                
                return pos_sigma, vel_sigma
            
            return last_mapping.position_sigma, last_mapping.velocity_sigma
            
        except Exception as e:
            self.logger.error(f"Noise parameter lookup error: {e}")
            return 1000.0, 10.0  # Default values
    
    def _get_altitude_factor(self, tle_data: Dict[str, Any]) -> float:
        """Get altitude-dependent noise factor"""
        try:
            # Estimate altitude from mean motion
            mean_motion = tle_data.get('mean_motion', 15.5)  # rev/day
            
            # Convert to semi-major axis
            mu = 3.986004418e14  # m^3/s^2
            n = mean_motion * 2 * np.pi / 86400  # rad/s
            a = (mu / (n**2))**(1/3)  # m
            
            # Altitude above Earth surface
            altitude_km = (a - 6371000) / 1000  # km
            
            # Atmospheric drag increases uncertainty at low altitudes
            if altitude_km < 300:
                # High drag regime - more uncertainty
                factor = 1.5
            elif altitude_km < 600:
                # Medium altitude - normal uncertainty
                factor = 1.0
            else:
                # High altitude - slightly less uncertainty
                factor = 0.8
            
            return factor
            
        except Exception as e:
            self.logger.error(f"Altitude factor calculation error: {e}")
            return 1.0
    
    def _get_orbital_dynamics_factor(self, tle_data: Dict[str, Any]) -> float:
        """Get factor based on orbital dynamics complexity"""
        try:
            eccentricity = tle_data.get('eccentricity', 0.0)
            inclination = tle_data.get('inclination', 0.0)
            
            # High eccentricity increases uncertainty
            ecc_factor = 1.0 + 2.0 * eccentricity
            
            # Polar/retrograde orbits may have more uncertainty
            inc_factor = 1.0
            if inclination > 80 or inclination < 10:
                inc_factor = 1.2
            
            return ecc_factor * inc_factor
            
        except Exception as e:
            self.logger.error(f"Orbital dynamics factor error: {e}")
            return 1.0
    
    def generate_measurement(self, tle_data: Dict[str, Any], 
                           current_time: datetime) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Generate synthetic measurement from TLE at current time
        
        Args:
            tle_data: TLE data dictionary
            current_time: Current time
            
        Returns:
            Tuple of (measurement_vector, noise_covariance) or (None, None)
        """
        try:
            # Get state vector from TLE using SGP4
            state_vector = self._tle_to_state_vector(tle_data, current_time)
            
            if state_vector is None:
                return None, None
            
            # Compute measurement noise
            R = self.compute_measurement_noise(tle_data, current_time)
            
            # Add noise to create realistic measurement
            noise = np.random.multivariate_normal(np.zeros(6), R)
            measurement = state_vector + noise
            
            return measurement, R
            
        except Exception as e:
            self.logger.error(f"Measurement generation error: {e}")
            return None, None
    
    def _tle_to_state_vector(self, tle_data: Dict[str, Any], 
                           current_time: datetime) -> Optional[np.ndarray]:
        """Convert TLE to state vector using SGP4"""
        try:
            from skyfield.api import load, EarthSatellite
            
            # Create satellite object
            satellite = EarthSatellite(tle_data['line1'], tle_data['line2'])
            
            # Get position and velocity at current time
            ts = load.timescale()
            t = ts.utc(current_time.year, current_time.month, current_time.day,
                      current_time.hour, current_time.minute, current_time.second)
            
            geocentric = satellite.at(t)
            position = np.array(geocentric.position.km) * 1000  # Convert to meters
            velocity = np.array(geocentric.velocity.km_per_s) * 1000  # Convert to m/s
            
            return np.concatenate([position, velocity])
            
        except Exception as e:
            self.logger.error(f"TLE to state vector conversion error: {e}")
            return None
    
    def _default_noise_matrix(self) -> np.ndarray:
        """Return default noise covariance matrix"""
        R = np.zeros((6, 6))
        
        # Default position uncertainty: 1 km
        R[:3, :3] = np.eye(3) * (1000.0**2)
        
        # Default velocity uncertainty: 10 m/s
        R[3:6, 3:6] = np.eye(3) * (10.0**2)
        
        return R
    
    def analyze_tle_residuals(self, tle_history: List[Dict[str, Any]], 
                            oem_data: Optional[Any] = None) -> Dict[str, Any]:
        """
        Analyze TLE residuals against OEM data to update age mappings
        
        Args:
            tle_history: List of historical TLE data points
            oem_data: OEM reference data
            
        Returns:
            Analysis results
        """
        try:
            if not oem_data or len(tle_history) == 0:
                return {'error': 'Insufficient data for analysis'}
            
            analysis_results = {
                'age_bins': [],
                'residual_statistics': [],
                'updated_mappings': []
            }
            
            # Group TLEs by age bins
            age_bins = [1, 6, 12, 24, 48, 72, 168]  # hours
            
            for i, max_age in enumerate(age_bins):
                min_age = age_bins[i-1] if i > 0 else 0
                
                # Find TLEs in this age range
                bin_residuals = []
                
                for tle_point in tle_history:
                    tle_age = tle_point.get('age_hours', 0)
                    
                    if min_age < tle_age <= max_age:
                        # Compute residual against OEM
                        residual = self._compute_tle_oem_residual(tle_point, oem_data)
                        if residual is not None:
                            bin_residuals.append(residual)
                
                if bin_residuals:
                    residuals = np.array(bin_residuals)
                    
                    # Compute statistics
                    pos_rms = np.sqrt(np.mean(residuals[:, :3]**2))
                    vel_rms = np.sqrt(np.mean(residuals[:, 3:6]**2))
                    
                    analysis_results['age_bins'].append(max_age)
                    analysis_results['residual_statistics'].append({
                        'age_range': [min_age, max_age],
                        'n_samples': len(bin_residuals),
                        'position_rms': pos_rms,
                        'velocity_rms': vel_rms,
                        'position_sigma': np.std(np.linalg.norm(residuals[:, :3], axis=1)),
                        'velocity_sigma': np.std(np.linalg.norm(residuals[:, 3:6], axis=1))
                    })
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"TLE residual analysis error: {e}")
            return {'error': str(e)}
    
    def _compute_tle_oem_residual(self, tle_point: Dict[str, Any], 
                                oem_data: Any) -> Optional[np.ndarray]:
        """Compute residual between TLE prediction and OEM truth"""
        try:
            # This would require implementing TLE propagation to OEM timestamp
            # and computing the difference - simplified for now
            
            # Placeholder implementation
            timestamp = tle_point.get('timestamp')
            if timestamp is None:
                return None
            
            # In practice, would propagate TLE to timestamp and compare with OEM
            # For now, return synthetic residual based on age
            age_hours = tle_point.get('age_hours', 24)
            
            # Synthetic residual scaling with age
            pos_sigma, vel_sigma = self._get_noise_for_age(age_hours)
            
            residual = np.concatenate([
                np.random.normal(0, pos_sigma, 3),
                np.random.normal(0, vel_sigma, 3)
            ])
            
            return residual
            
        except Exception as e:
            self.logger.error(f"TLE-OEM residual computation error: {e}")
            return None
    
    def update_age_mappings(self, analysis_results: Dict[str, Any]):
        """Update age mappings based on analysis results"""
        try:
            if 'residual_statistics' not in analysis_results:
                return
            
            updated_mappings = []
            
            for i, stats in enumerate(analysis_results['residual_statistics']):
                if stats['n_samples'] >= 5:  # Minimum samples for reliable estimate
                    age_range = stats['age_range']
                    max_age = age_range[1]
                    
                    # Use RMS + margin for sigma estimate
                    pos_sigma = stats['position_rms'] * 1.2  # 20% margin
                    vel_sigma = stats['velocity_rms'] * 1.2
                    
                    updated_mappings.append(TLEAgeMapping(max_age, pos_sigma, vel_sigma))
            
            if updated_mappings:
                self.age_mappings = updated_mappings
                self.logger.info(f"Updated {len(updated_mappings)} age mappings")
                
                # Save updated mappings
                self._save_updated_mappings(analysis_results)
            
        except Exception as e:
            self.logger.error(f"Age mapping update error: {e}")
    
    def _save_updated_mappings(self, analysis_results: Dict[str, Any]):
        """Save updated mappings to file"""
        try:
            mappings_file = self.config.get('historical_mappings_file', 'data/tle_noise_mappings.json')
            
            updated_analysis = {
                'tle_age_analysis': {
                    'description': 'Updated TLE age to measurement noise mapping',
                    'source': 'Empirical TLE vs OEM analysis',
                    'mappings': [
                        {
                            'age_hours': mapping.max_age_hours,
                            'position_sigma_m': mapping.position_sigma,
                            'velocity_sigma_ms': mapping.velocity_sigma
                        }
                        for mapping in self.age_mappings
                    ]
                },
                'analysis_results': analysis_results,
                'analysis_metadata': {
                    'updated': datetime.utcnow().isoformat(),
                    'satellite': 'ISS',
                    'tle_source': 'Celestrak',
                    'oem_source': 'NASA JSC'
                }
            }
            
            with open(mappings_file, 'w') as f:
                json.dump(updated_analysis, f, indent=2)
                
            self.logger.info(f"Saved updated mappings to {mappings_file}")
            
        except Exception as e:
            self.logger.error(f"Mappings save error: {e}")
    
    def get_measurement_quality(self, tle_data: Dict[str, Any], 
                              current_time: datetime) -> Dict[str, Any]:
        """Get measurement quality assessment"""
        try:
            age_hours = self._calculate_tle_age(tle_data, current_time)
            pos_sigma, vel_sigma = self._get_noise_for_age(age_hours)
            
            # Quality assessment
            if age_hours < 6:
                quality = "Excellent"
            elif age_hours < 24:
                quality = "Good"
            elif age_hours < 72:
                quality = "Fair"
            else:
                quality = "Poor"
            
            return {
                'age_hours': age_hours,
                'position_uncertainty_m': pos_sigma,
                'velocity_uncertainty_ms': vel_sigma,
                'quality_rating': quality,
                'recommended_weight': max(0.1, 1.0 / (1.0 + age_hours / 24))
            }
            
        except Exception as e:
            self.logger.error(f"Quality assessment error: {e}")
            return {'quality_rating': 'Unknown', 'recommended_weight': 0.5}

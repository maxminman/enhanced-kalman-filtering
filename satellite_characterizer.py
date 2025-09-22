import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from enum import Enum
import json
import os

class OrbitalRegime(Enum):
    """Classification of orbital regimes for LEO satellites"""
    VERY_LOW_LEO = "very_low_leo"      # 200-400 km
    LOW_LEO = "low_leo"                # 400-600 km  
    MID_LEO = "mid_leo"                # 600-800 km
    HIGH_LEO = "high_leo"              # 800-1000 km

class SatelliteType(Enum):
    """Classification of satellite types based on characteristics"""
    SPACE_STATION = "space_station"    # Large, high drag (ISS, Tiangong)
    EARTH_OBSERVATION = "earth_obs"    # Medium size, moderate drag (Sentinel, Landsat)
    COMMUNICATION = "communication"    # Various sizes, moderate drag (Starlink, OneWeb)
    SCIENTIFIC = "scientific"          # Small to medium, low drag (CubeSats, research)
    DEBRIS = "debris"                  # Unknown properties, high uncertainty
    UNKNOWN = "unknown"                # Cannot classify

@dataclass
class SatelliteProperties:
    """Comprehensive satellite properties with uncertainty bounds"""
    norad_id: str
    name: Optional[str]
    
    # Physical properties
    mass: float  # kg
    drag_area: float  # m²
    srp_area: float  # m²
    drag_coefficient: float
    srp_coefficient: float
    
    # Derived properties
    ballistic_coefficient: float  # Cd*A/m (m²/kg)
    area_to_mass_ratio: float    # A/m (m²/kg)
    
    # Classification
    orbital_regime: OrbitalRegime
    satellite_type: SatelliteType
    
    # Uncertainty bounds (parameter_name -> (lower_bound, upper_bound))
    uncertainty_bounds: Dict[str, Tuple[float, float]]
    
    # Confidence metrics
    characterization_confidence: float  # 0.0 to 1.0
    parameter_confidence: Dict[str, float]  # Individual parameter confidence
    
    # Metadata
    characterization_timestamp: datetime
    data_sources: List[str]

class SatelliteCharacterizer:
    """
    Automated satellite characterization engine that estimates satellite properties
    from TLE data and orbital behavior for satellite-agnostic tracking
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize satellite characterizer"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Load satellite database if available
        self.satellite_db = self._load_satellite_database()
        
        # Characterization parameters
        self.min_tle_history = self.config.get('min_tle_history', 5)  # Minimum TLE points for analysis
        self.decay_analysis_window = self.config.get('decay_analysis_window', 30)  # Days
        
        # Physical constants
        self.earth_radius = 6378137.0  # m
        self.mu = 3.986004418e14  # m³/s²
        
        # Default satellite parameters by type
        self._initialize_default_parameters()
        
        self.logger.info("Satellite characterizer initialized")
    
    def characterize_satellite(self, tle_data, norad_id: str, 
                             tle_history: Optional[List] = None) -> SatelliteProperties:
        """
        Characterize satellite from TLE data and optional historical TLE data
        
        Args:
            tle_data: Current TLE data object
            norad_id: NORAD catalog ID
            tle_history: Optional list of historical TLE data for decay analysis
            
        Returns:
            SatelliteProperties with estimated parameters and uncertainties
        """
        try:
            self.logger.info(f"Characterizing satellite {norad_id}")
            
            # Check satellite database first
            db_props = self._lookup_satellite_database(norad_id)
            if db_props and db_props.characterization_confidence > 0.9:
                self.logger.info(f"High-confidence database match for {norad_id}")
                return db_props
            
            # Classify orbital regime
            orbital_regime = self._classify_orbital_regime(tle_data)
            
            # Estimate ballistic coefficient from TLE decay analysis
            ballistic_coeff, bc_confidence = self._estimate_ballistic_coefficient(
                tle_data, tle_history
            )
            
            # Classify satellite type based on orbital characteristics
            satellite_type, type_confidence = self._classify_satellite_type(
                tle_data, ballistic_coeff, orbital_regime
            )
            
            # Estimate physical parameters based on classification
            physical_params = self._estimate_physical_parameters(
                satellite_type, orbital_regime, ballistic_coeff
            )
            
            # Calculate uncertainty bounds
            uncertainty_bounds = self._calculate_uncertainty_bounds(
                physical_params, bc_confidence, type_confidence
            )
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(
                bc_confidence, type_confidence, db_props is not None
            )
            
            # Create satellite properties object
            satellite_props = SatelliteProperties(
                norad_id=norad_id,
                name=getattr(tle_data, 'satellite_name', None),
                mass=physical_params['mass'],
                drag_area=physical_params['drag_area'],
                srp_area=physical_params['srp_area'],
                drag_coefficient=physical_params['drag_coefficient'],
                srp_coefficient=physical_params['srp_coefficient'],
                ballistic_coefficient=ballistic_coeff,
                area_to_mass_ratio=physical_params['drag_area'] / physical_params['mass'],
                orbital_regime=orbital_regime,
                satellite_type=satellite_type,
                uncertainty_bounds=uncertainty_bounds,
                characterization_confidence=overall_confidence,
                parameter_confidence={
                    'ballistic_coefficient': bc_confidence,
                    'satellite_type': type_confidence,
                    'mass': physical_params.get('mass_confidence', 0.5),
                    'drag_area': physical_params.get('area_confidence', 0.5)
                },
                characterization_timestamp=datetime.utcnow(),
                data_sources=['tle_analysis', 'orbital_classification']
            )
            
            # Update satellite database with new characterization
            self._update_satellite_database(satellite_props)
            
            self.logger.info(f"Satellite {norad_id} characterized: {satellite_type.value}, "
                           f"confidence={overall_confidence:.2f}")
            
            return satellite_props
            
        except Exception as e:
            self.logger.error(f"Satellite characterization failed for {norad_id}: {e}")
            return self._create_default_properties(norad_id, tle_data)
    
    def _classify_orbital_regime(self, tle_data) -> OrbitalRegime:
        """Classify orbital regime based on altitude"""
        try:
            # Calculate semi-major axis from mean motion
            n = tle_data.mean_motion * 2 * np.pi / 86400  # rad/s
            a = (self.mu / (n**2))**(1/3)  # m
            
            # Calculate approximate altitude (assuming circular orbit)
            altitude = a - self.earth_radius  # m
            altitude_km = altitude / 1000  # km
            
            if altitude_km < 400:
                return OrbitalRegime.VERY_LOW_LEO
            elif altitude_km < 600:
                return OrbitalRegime.LOW_LEO
            elif altitude_km < 800:
                return OrbitalRegime.MID_LEO
            else:
                return OrbitalRegime.HIGH_LEO
                
        except Exception as e:
            self.logger.warning(f"Orbital regime classification failed: {e}")
            return OrbitalRegime.LOW_LEO  # Default
    
    def _estimate_ballistic_coefficient(self, tle_data, tle_history: Optional[List]) -> Tuple[float, float]:
        """
        Estimate ballistic coefficient from TLE decay analysis
        
        Returns:
            Tuple of (ballistic_coefficient, confidence)
        """
        try:
            if not tle_history or len(tle_history) < self.min_tle_history:
                # Use single TLE estimate based on orbital regime
                return self._estimate_bc_from_single_tle(tle_data)
            
            # Analyze decay rate from TLE history
            decay_rate, decay_confidence = self._analyze_decay_rate(tle_history)
            
            if decay_rate is None or decay_confidence < 0.3:
                return self._estimate_bc_from_single_tle(tle_data)
            
            # Convert decay rate to ballistic coefficient
            # This is a simplified model - full implementation would use atmospheric density models
            altitude = self._get_altitude_from_tle(tle_data)
            
            # Approximate atmospheric density at altitude (kg/m³)
            rho = self._approximate_atmospheric_density(altitude)
            
            # Orbital velocity (m/s)
            velocity = np.sqrt(self.mu / (altitude + self.earth_radius))
            
            # Ballistic coefficient from decay rate
            # decay_rate ≈ -1.5 * rho * Bc * velocity / (2 * altitude)
            if rho > 0 and velocity > 0 and altitude > 0:
                bc = abs(decay_rate) * 2 * altitude / (1.5 * rho * velocity)
                bc = np.clip(bc, 0.001, 0.020)  # Physical bounds
                
                return bc, decay_confidence
            else:
                return self._estimate_bc_from_single_tle(tle_data)
                
        except Exception as e:
            self.logger.warning(f"Ballistic coefficient estimation failed: {e}")
            return self._estimate_bc_from_single_tle(tle_data)
    
    def _estimate_bc_from_single_tle(self, tle_data) -> Tuple[float, float]:
        """Estimate ballistic coefficient from single TLE based on orbital characteristics"""
        try:
            # Get orbital parameters
            altitude = self._get_altitude_from_tle(tle_data)
            inclination = tle_data.inclination
            eccentricity = tle_data.eccentricity
            
            # Estimate based on altitude and inclination patterns
            if altitude < 400000:  # Very low orbit
                bc_base = 0.008  # High drag
            elif altitude < 600000:  # Low orbit
                bc_base = 0.005
            else:  # Higher orbit
                bc_base = 0.003
            
            # Adjust for inclination (sun-synchronous orbits often have different characteristics)
            if 95 < inclination < 105:  # Sun-synchronous
                bc_base *= 0.8
            
            # Adjust for eccentricity
            if eccentricity > 0.01:  # Eccentric orbit
                bc_base *= 1.2
            
            bc = np.clip(bc_base, 0.001, 0.015)
            confidence = 0.3  # Low confidence for single TLE estimate
            
            return bc, confidence
            
        except Exception as e:
            self.logger.warning(f"Single TLE BC estimation failed: {e}")
            return 0.005, 0.2  # Default with very low confidence
    
    def _analyze_decay_rate(self, tle_history: List) -> Tuple[Optional[float], float]:
        """Analyze orbital decay rate from TLE history"""
        try:
            if len(tle_history) < 3:
                return None, 0.0
            
            # Extract altitudes and times
            times = []
            altitudes = []
            
            for tle in tle_history:
                try:
                    altitude = self._get_altitude_from_tle(tle)
                    epoch = tle.epoch_datetime
                    
                    times.append(epoch.timestamp())
                    altitudes.append(altitude)
                except:
                    continue
            
            if len(times) < 3:
                return None, 0.0
            
            # Convert to numpy arrays
            times = np.array(times)
            altitudes = np.array(altitudes)
            
            # Sort by time
            sort_idx = np.argsort(times)
            times = times[sort_idx]
            altitudes = altitudes[sort_idx]
            
            # Linear regression for decay rate
            time_days = (times - times[0]) / 86400  # Convert to days
            
            if len(time_days) < 3 or np.max(time_days) < 1:  # Need at least 1 day span
                return None, 0.0
            
            # Fit linear trend
            coeffs = np.polyfit(time_days, altitudes, 1)
            decay_rate = coeffs[0]  # m/day
            
            # Calculate R-squared for confidence
            altitude_pred = np.polyval(coeffs, time_days)
            ss_res = np.sum((altitudes - altitude_pred)**2)
            ss_tot = np.sum((altitudes - np.mean(altitudes))**2)
            
            if ss_tot > 0:
                r_squared = 1 - (ss_res / ss_tot)
                confidence = np.clip(r_squared, 0.0, 1.0)
            else:
                confidence = 0.0
            
            # Convert to m/s
            decay_rate_ms = decay_rate / 86400
            
            return decay_rate_ms, confidence
            
        except Exception as e:
            self.logger.warning(f"Decay rate analysis failed: {e}")
            return None, 0.0
    
    def _classify_satellite_type(self, tle_data, ballistic_coeff: float, 
                               orbital_regime: OrbitalRegime) -> Tuple[SatelliteType, float]:
        """Classify satellite type based on orbital characteristics and ballistic coefficient"""
        try:
            altitude = self._get_altitude_from_tle(tle_data) / 1000  # km
            inclination = tle_data.inclination
            
            # Classification logic based on orbital characteristics
            confidence = 0.6  # Base confidence
            
            # Space stations (large, low altitude, high drag)
            # ISS characteristics: ~400km altitude, ~51.6° inclination, high ballistic coefficient
            if (altitude < 450 and 45 < inclination < 60):
                # High ballistic coefficient indicates large satellite
                if ballistic_coeff > 0.004:
                    return SatelliteType.SPACE_STATION, 0.8
                else:
                    # Could still be space station with lower estimate
                    return SatelliteType.SPACE_STATION, 0.6
            
            # Earth observation satellites (sun-synchronous orbits)
            if (95 < inclination < 105 and 600 < altitude < 800):
                return SatelliteType.EARTH_OBSERVATION, 0.7
            
            # Communication satellites (various patterns)
            if (altitude > 500 and ballistic_coeff < 0.004):
                return SatelliteType.COMMUNICATION, 0.6
            
            # Scientific satellites (often small, various orbits)
            if (ballistic_coeff < 0.003 and altitude > 400):
                return SatelliteType.SCIENTIFIC, 0.5
            
            # Additional check for ISS-like characteristics even with lower BC estimate
            if (altitude < 450 and 50 < inclination < 53 and orbital_regime == OrbitalRegime.LOW_LEO):
                return SatelliteType.SPACE_STATION, 0.7
            
            # Default to unknown with low confidence
            return SatelliteType.UNKNOWN, 0.3
            
        except Exception as e:
            self.logger.warning(f"Satellite type classification failed: {e}")
            return SatelliteType.UNKNOWN, 0.2
    
    def _estimate_physical_parameters(self, satellite_type: SatelliteType, 
                                    orbital_regime: OrbitalRegime, 
                                    ballistic_coeff: float) -> Dict[str, float]:
        """Estimate physical parameters based on satellite classification"""
        try:
            # Get default parameters for satellite type
            defaults = self.default_params.get(satellite_type, self.default_params[SatelliteType.UNKNOWN])
            
            # Estimate mass from ballistic coefficient and assumed drag area
            drag_area = defaults['drag_area']
            drag_coeff = defaults['drag_coefficient']
            
            # mass = Cd * A / Bc
            estimated_mass = drag_coeff * drag_area / ballistic_coeff
            
            # Apply reasonable bounds
            mass_bounds = defaults['mass_bounds']
            estimated_mass = np.clip(estimated_mass, mass_bounds[0], mass_bounds[1])
            
            # Adjust parameters based on orbital regime
            regime_factors = {
                OrbitalRegime.VERY_LOW_LEO: {'drag_factor': 1.2, 'srp_factor': 0.8},
                OrbitalRegime.LOW_LEO: {'drag_factor': 1.0, 'srp_factor': 1.0},
                OrbitalRegime.MID_LEO: {'drag_factor': 0.8, 'srp_factor': 1.2},
                OrbitalRegime.HIGH_LEO: {'drag_factor': 0.6, 'srp_factor': 1.4}
            }
            
            factors = regime_factors.get(orbital_regime, {'drag_factor': 1.0, 'srp_factor': 1.0})
            
            return {
                'mass': estimated_mass,
                'drag_area': drag_area * factors['drag_factor'],
                'srp_area': defaults['srp_area'] * factors['srp_factor'],
                'drag_coefficient': drag_coeff,
                'srp_coefficient': defaults['srp_coefficient'],
                'mass_confidence': 0.6,
                'area_confidence': 0.5
            }
            
        except Exception as e:
            self.logger.warning(f"Physical parameter estimation failed: {e}")
            return self.default_params[SatelliteType.UNKNOWN]
    
    def _calculate_uncertainty_bounds(self, physical_params: Dict[str, float], 
                                    bc_confidence: float, type_confidence: float) -> Dict[str, Tuple[float, float]]:
        """Calculate uncertainty bounds for all parameters"""
        try:
            # Base uncertainty factors (higher uncertainty = lower confidence)
            base_uncertainty = 1.0 - (bc_confidence * type_confidence)
            
            uncertainty_factors = {
                'mass': 0.3 + 0.4 * base_uncertainty,
                'drag_area': 0.2 + 0.3 * base_uncertainty,
                'srp_area': 0.25 + 0.35 * base_uncertainty,
                'drag_coefficient': 0.15 + 0.25 * base_uncertainty,
                'srp_coefficient': 0.2 + 0.3 * base_uncertainty,
                'ballistic_coefficient': 0.2 + 0.3 * base_uncertainty
            }
            
            bounds = {}
            for param, value in physical_params.items():
                if param.endswith('_confidence'):
                    continue
                    
                factor = uncertainty_factors.get(param, 0.3)
                lower = value * (1 - factor)
                upper = value * (1 + factor)
                bounds[param] = (lower, upper)
            
            return bounds
            
        except Exception as e:
            self.logger.warning(f"Uncertainty bounds calculation failed: {e}")
            return {}
    
    def _calculate_overall_confidence(self, bc_confidence: float, type_confidence: float, 
                                    has_db_match: bool) -> float:
        """Calculate overall characterization confidence"""
        try:
            # Weighted combination of confidence factors
            confidence = 0.4 * bc_confidence + 0.4 * type_confidence
            
            # Bonus for database match
            if has_db_match:
                confidence += 0.2
            
            # Ensure bounds
            confidence = np.clip(confidence, 0.0, 1.0)
            
            return confidence
            
        except Exception as e:
            self.logger.warning(f"Confidence calculation failed: {e}")
            return 0.3
    
    def _get_altitude_from_tle(self, tle_data) -> float:
        """Calculate altitude from TLE data"""
        try:
            n = tle_data.mean_motion * 2 * np.pi / 86400  # rad/s
            a = (self.mu / (n**2))**(1/3)  # m
            altitude = a - self.earth_radius  # m
            return altitude
        except:
            return 400000.0  # Default 400 km
    
    def _approximate_atmospheric_density(self, altitude: float) -> float:
        """Approximate atmospheric density at altitude (very simplified)"""
        try:
            # Simplified exponential atmosphere model
            h = altitude / 1000  # km
            if h < 200:
                return 2.5e-11  # kg/m³
            elif h < 300:
                return 1.8e-11 * np.exp(-(h - 200) / 50)
            elif h < 500:
                return 6.0e-12 * np.exp(-(h - 300) / 60)
            else:
                return 1.0e-12 * np.exp(-(h - 500) / 80)
        except:
            return 1e-12  # Default very low density
    
    def _initialize_default_parameters(self):
        """Initialize default parameters for different satellite types"""
        self.default_params = {
            SatelliteType.SPACE_STATION: {
                'mass': 450000.0,  # kg (ISS-like)
                'drag_area': 1500.0,  # m²
                'srp_area': 1200.0,  # m²
                'drag_coefficient': 2.2,
                'srp_coefficient': 1.3,
                'mass_bounds': (200000, 600000),
                'mass_confidence': 0.7,
                'area_confidence': 0.6
            },
            SatelliteType.EARTH_OBSERVATION: {
                'mass': 2300.0,  # kg (Sentinel-like)
                'drag_area': 12.0,  # m²
                'srp_area': 10.0,  # m²
                'drag_coefficient': 2.0,
                'srp_coefficient': 1.2,
                'mass_bounds': (1000, 5000),
                'mass_confidence': 0.6,
                'area_confidence': 0.5
            },
            SatelliteType.COMMUNICATION: {
                'mass': 260.0,  # kg (Starlink-like)
                'drag_area': 8.0,  # m²
                'srp_area': 6.0,  # m²
                'drag_coefficient': 2.1,
                'srp_coefficient': 1.1,
                'mass_bounds': (100, 1000),
                'mass_confidence': 0.5,
                'area_confidence': 0.4
            },
            SatelliteType.SCIENTIFIC: {
                'mass': 150.0,  # kg (CubeSat/small sat)
                'drag_area': 2.0,  # m²
                'srp_area': 1.5,  # m²
                'drag_coefficient': 2.3,
                'srp_coefficient': 1.0,
                'mass_bounds': (10, 500),
                'mass_confidence': 0.4,
                'area_confidence': 0.3
            },
            SatelliteType.UNKNOWN: {
                'mass': 500.0,  # kg (conservative estimate)
                'drag_area': 10.0,  # m²
                'srp_area': 8.0,  # m²
                'drag_coefficient': 2.2,
                'srp_coefficient': 1.2,
                'mass_bounds': (50, 2000),
                'mass_confidence': 0.3,
                'area_confidence': 0.3
            }
        }
    
    def _load_satellite_database(self) -> Dict[str, SatelliteProperties]:
        """Load satellite database from file if available"""
        try:
            db_path = self.config.get('satellite_db_path', 'data/satellite_database.json')
            if os.path.exists(db_path):
                with open(db_path, 'r') as f:
                    db_data = json.load(f)
                
                # Convert to SatelliteProperties objects
                satellite_db = {}
                for norad_id, data in db_data.items():
                    # This would need proper deserialization
                    # For now, return empty dict
                    pass
                
                self.logger.info(f"Loaded satellite database with {len(satellite_db)} entries")
                return satellite_db
            else:
                self.logger.info("No satellite database found, starting with empty database")
                return {}
                
        except Exception as e:
            self.logger.warning(f"Failed to load satellite database: {e}")
            return {}
    
    def _lookup_satellite_database(self, norad_id: str) -> Optional[SatelliteProperties]:
        """Look up satellite in database"""
        return self.satellite_db.get(norad_id)
    
    def _update_satellite_database(self, satellite_props: SatelliteProperties):
        """Update satellite database with new characterization"""
        try:
            self.satellite_db[satellite_props.norad_id] = satellite_props
            
            # Optionally save to file
            if self.config.get('save_characterizations', True):
                self._save_satellite_database()
                
        except Exception as e:
            self.logger.warning(f"Failed to update satellite database: {e}")
    
    def _save_satellite_database(self):
        """Save satellite database to file"""
        try:
            db_path = self.config.get('satellite_db_path', 'data/satellite_database.json')
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            # Convert to serializable format
            db_data = {}
            for norad_id, props in self.satellite_db.items():
                # This would need proper serialization
                # For now, just save basic info
                db_data[norad_id] = {
                    'characterization_confidence': props.characterization_confidence,
                    'satellite_type': props.satellite_type.value,
                    'orbital_regime': props.orbital_regime.value,
                    'timestamp': props.characterization_timestamp.isoformat()
                }
            
            with open(db_path, 'w') as f:
                json.dump(db_data, f, indent=2)
                
            self.logger.debug(f"Saved satellite database to {db_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save satellite database: {e}")
    
    def _create_default_properties(self, norad_id: str, tle_data) -> SatelliteProperties:
        """Create default satellite properties when characterization fails"""
        try:
            orbital_regime = self._classify_orbital_regime(tle_data)
            defaults = self.default_params[SatelliteType.UNKNOWN]
            
            return SatelliteProperties(
                norad_id=norad_id,
                name=getattr(tle_data, 'satellite_name', None),
                mass=defaults['mass'],
                drag_area=defaults['drag_area'],
                srp_area=defaults['srp_area'],
                drag_coefficient=defaults['drag_coefficient'],
                srp_coefficient=defaults['srp_coefficient'],
                ballistic_coefficient=0.005,  # Default
                area_to_mass_ratio=defaults['drag_area'] / defaults['mass'],
                orbital_regime=orbital_regime,
                satellite_type=SatelliteType.UNKNOWN,
                uncertainty_bounds={
                    'mass': (defaults['mass'] * 0.5, defaults['mass'] * 2.0),
                    'drag_area': (defaults['drag_area'] * 0.7, defaults['drag_area'] * 1.5),
                    'ballistic_coefficient': (0.002, 0.010)
                },
                characterization_confidence=0.2,
                parameter_confidence={
                    'ballistic_coefficient': 0.2,
                    'satellite_type': 0.1,
                    'mass': 0.2,
                    'drag_area': 0.2
                },
                characterization_timestamp=datetime.utcnow(),
                data_sources=['default_fallback']
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create default properties: {e}")
            raise
    
    def get_characterization_summary(self, satellite_props: SatelliteProperties) -> Dict[str, Any]:
        """Get human-readable characterization summary"""
        return {
            'norad_id': satellite_props.norad_id,
            'name': satellite_props.name,
            'satellite_type': satellite_props.satellite_type.value,
            'orbital_regime': satellite_props.orbital_regime.value,
            'confidence': f"{satellite_props.characterization_confidence:.1%}",
            'estimated_mass': f"{satellite_props.mass:.0f} kg",
            'ballistic_coefficient': f"{satellite_props.ballistic_coefficient:.6f} m²/kg",
            'characterization_age': (datetime.utcnow() - satellite_props.characterization_timestamp).total_seconds() / 3600,
            'data_sources': satellite_props.data_sources
        }